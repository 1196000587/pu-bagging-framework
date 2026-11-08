"""One orchestration layer for independent data, model, optimizer and export adapters."""
from pathlib import Path
import importlib.metadata
import logging
import platform
import sys
import traceback
import numpy as np
import pandas as pd
import joblib
from .config import load_config, resolve_path
from .data import load_data
from .evaluation import evaluate, training_draw, split_folds
from .models import build_model, fit_classifier
from .sampling import sample_count, rounds_for_coverage, expected_coverage, seed_for
from .optimizers import optimize, candidate_settings, validate_space
from .ensemble import RunningMoments, batched_probability
from .outputs import export_tables, export_geotiffs, capture_curve, write_json
from .plots import plot_results
from .explain import ExplanationCollector


def planned_rounds(config, n_pos, n_unlabeled, ratio):
    count = sample_count(n_pos, n_unlabeled, ratio)
    rounds = config["sampling"]["rounds"] or rounds_for_coverage(n_unlabeled, count, config["sampling"]["coverage"])
    if rounds > config["sampling"]["max_rounds"]:
        raise ValueError(f"Required {rounds} rounds exceeds max_rounds; adjust sampling settings explicitly.")
    return count, rounds


def validate(config):
    data = load_data(config)
    ratio = float(config["sampling"]["ratio"])
    count, rounds = planned_rounds(config, len(data.positive), len(data.unlabeled), ratio)
    X, y, groups, _ = training_draw(data, ratio, seed_for(config["seed"], 10))
    split_folds(X, y, groups, config["cv"], seed_for(config["seed"], 10, 0, 999))
    build_model(config["model"], config["preprocess"], data.feature_names, {}, config["seed"])
    validate_space(config["optimizer"]["space"])
    if config["explain"]["enabled"]:
        import shap  # noqa: F401
    if config["plots"]["enabled"]:
        import matplotlib  # noqa: F401
    if config["outputs"]["excel"]:
        import openpyxl  # noqa: F401
    return data, {"positive_rows": len(data.positive), "unlabeled_rows": len(data.unlabeled),
                  "prediction_rows": len(data.prediction), "features": data.feature_names,
                  "configured_ratio": ratio, "configured_draw_size": count,
                  "rounds_if_ratio_unchanged": rounds,
                  "expected_coverage_if_ratio_unchanged": expected_coverage(len(data.unlabeled), count, rounds),
                  "note": "The final number of rounds is recomputed AFTER selection of the sampling ratio.",
                  "diagnostics": data.diagnostics}


def run(config_or_path):
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    data, report = validate(config)
    directory = resolve_path(config, config["output_dir"])
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}. Choose a new output_dir.")
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pu_bagging.{directory}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.addHandler(stream)
    write_json(directory / "config_resolved.json", config)
    write_json(directory / "data_summary.json", report)
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "shap", "xgboost", "optuna", "rasterio", "geopandas", "matplotlib"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {"status": "running", "python": platform.python_version(), "packages": versions,
                "seed": config["seed"], "seed_stages": {"search": 10, "evaluation": 20, "ensemble": 30}}
    write_json(directory / "run_metadata.json", metadata)
    try:
        def objective(candidate):
            params, ratio, weight = candidate_settings(config, candidate)
            return evaluate(data, config, params, ratio, config["optimizer"]["screening_rounds"],
                            stage=10, positive_weight=weight)["mean_auc"]
        logger.info("Data ready: %s positives, %s unlabeled, %s features", len(data.positive), len(data.unlabeled), len(data.feature_names))
        selected, history = optimize(objective, config["optimizer"], seed_for(config["seed"], 90),
                                     lambda row: logger.info("Search evaluation %s: mean CV AUC %.5f", row["evaluation"], row["mean_cv_auc"]))
        params, ratio, weight = candidate_settings(config, selected)
        count, rounds = planned_rounds(config, len(data.positive), len(data.unlabeled), ratio)
        selected_parameters = {"candidate": selected, "model": config["model"], "ratio": ratio,
                               "effective_model_params": {**config["model"]["params"], **params},
                               "positive_weight": weight, "unlabeled_per_draw": count, "rounds": rounds,
                               "rounds_source": "explicit" if config["sampling"]["rounds"] else "expected_coverage",
                               "expected_coverage": expected_coverage(len(data.unlabeled), count, rounds)}
        write_json(directory / "selected_parameters.json", selected_parameters)
        logger.info("Selected ratio %g; %s unlabeled/draw; %s rounds", ratio, count, rounds)
        evaluation = evaluate(data, config, params, ratio, rounds, stage=20, positive_weight=weight,
                              progress=lambda row: logger.info("CV draw %s/%s: AUC %.5f", row["draw"], rounds, row["mean_fold_auc"]))
        logger.info("Repeated internal CV AUC %.5f ± %.5f", evaluation["mean_auc"], evaluation["std_auc"])
        moments = RunningMoments(len(data.prediction))
        counts = np.zeros(len(data.unlabeled), dtype=int)
        members, sampling_records = [], []
        collector = ExplanationCollector(directory / "explanations", data.feature_names, config["explain"]) if config["explain"]["enabled"] else None
        if config["prediction"]["save_models"]:
            (directory / "models").mkdir()
        for draw in range(rounds):
            sample_seed = seed_for(config["seed"], 30, draw)
            model_draw = draw if config["model"]["seed_mode"] == "per_draw" else 0
            model_seed = seed_for(config["seed"], 100, model_draw)
            X, y, _, ix = training_draw(data, ratio, sample_seed)
            model = build_model(config["model"], config["preprocess"], data.feature_names, params, model_seed)
            fit_classifier(model, X, y, weight)
            moments.update(batched_probability(model, data.prediction, config["prediction"]["batch_size"]))
            counts[ix] += 1
            sampling_records.append({"draw": draw + 1, "sample_seed": sample_seed, "model_seed": model_seed,
                                     "n_unlabeled": len(ix), "observed_coverage": float(np.mean(counts > 0))})
            if collector:
                ids = np.r_[data.positive_ids, data.unlabeled_ids[ix]]
                collector.add(model, X, ids, draw + 1, seed_for(config["seed"], 40, draw))
            if config["prediction"]["save_models"]:
                filename = f"models/member_{draw + 1:05d}.joblib"
                joblib.dump(model, directory / filename, compress=3)
                members.append(filename)
            logger.info("Ensemble %s/%s; observed background coverage %.2f%%", draw + 1, rounds, 100 * np.mean(counts > 0))
        predictions = data.prediction_metadata.copy()
        predictions["probability_mean"], predictions["probability_std"] = moments.mean, moments.std
        correlation = pd.DataFrame(data.prediction, columns=data.feature_names).corr().rename_axis("feature").reset_index()
        tables = {"cv_draws": evaluation["draws"], "cv_folds": evaluation["folds"], "roc": evaluation["roc"],
                  "optimization": history, "predictions": predictions, "sampling": pd.DataFrame(sampling_records),
                  "unlabeled_counts": pd.DataFrame({"id": data.unlabeled_ids, "draw_count": counts}),
                  "feature_correlation": correlation,
                  "capture_in_sample": capture_curve(moments.mean, predictions.known_positive)}
        if collector:
            tables.update(collector.summarize())
        metrics = {"mean_cv_auc": evaluation["mean_auc"], "std_cv_auc": evaluation["std_auc"],
                   "draws": rounds, "positive_count": len(data.positive), "unlabeled_count": len(data.unlabeled),
                   "expected_coverage": selected_parameters["expected_coverage"],
                   "observed_coverage": float(np.mean(counts > 0)),
                   "evaluation_scope": "Repeated internal CV after parameter selection; not nested CV or independent external validation.",
                   "label_scope": "Known positives versus temporarily labeled unlabeled background.",
                   "probability_scope": "Classifier positive-class scores; no PU class-prior calibration is performed.",
                   "dispersion_scope": "Population SD across draws/members; not standard error or calibrated uncertainty.",
                   "shap_scope": "Training samples per fit, pooled with repeated occurrences; not held-out explanations."}
        tables["metrics"] = pd.DataFrame([metrics])
        export_tables(tables, directory, config["outputs"])
        write_json(directory / "plot_context.json", {"raster_shape": [data.raster_profile["height"], data.raster_profile["width"]]
                                                     if data.raster_profile else None})
        if config["outputs"]["geotiff"]:
            export_geotiffs(data, moments.mean, moments.std, directory)
        plot_results(tables, data, directory, config["plots"])
        write_json(directory / "metrics.json", metrics)
        if members:
            write_json(directory / "ensemble.json", {"features": data.feature_names, "members": members})
        metadata["status"] = "complete"
        write_json(directory / "run_metadata.json", metadata)
        logger.info("Completed: %s", directory)
        return {"output_dir": str(directory), "metrics": metrics, "selected_parameters": selected_parameters}
    except (Exception, KeyboardInterrupt) as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        write_json(directory / "run_metadata.json", metadata)
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.removeHandler(handler)
        logger.removeHandler(stream)
        handler.close()
        stream.close()
