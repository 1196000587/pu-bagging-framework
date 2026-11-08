"""YAML configuration. Paths are relative to the configuration file."""
from pathlib import Path
import copy
import importlib
import math
import yaml

DEFAULTS = {
    "seed": 42,
    "output_dir": "../outputs/run",
    "model": {"name": "random_forest", "params": {}, "n_jobs": 1, "seed_mode": "fixed"},
    "preprocess": {"scaling": "none", "feature_scales": {}},
    "training": {"positive_weight": 1.0},
    "sampling": {"ratio": 3.0, "coverage": 0.9, "rounds": None, "max_rounds": 10000},
    "cv": {"strategy": "stratified", "folds": 5},
    "optimizer": {"name": "none", "screening_rounds": 20, "space": {}, "options": {}},
    "explain": {"enabled": False, "method": "tree", "max_samples": None,
                "background_samples": 100, "bins": 20, "clip_percentile": 1.0,
                "interactions": False, "top_interactions": 5},
    "prediction": {"batch_size": 5000, "save_models": False},
    "outputs": {"csv": True, "excel": False, "geotiff": True},
    "plots": {"enabled": False, "dpi": 600, "formats": ["png"], "auc": True,
              "roc": True, "maps": True, "correlation": True, "capture": True,
              "optimization": True, "shap_importance": True, "shap_dependence": True,
              "shap_interactions": True},
}


def import_object(path):
    """Import trusted local extension code, e.g. my_models:MyClassifier."""
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError("Custom import paths must use module:object syntax.")
    return getattr(importlib.import_module(module), name)


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    unknown = set(supplied) - set(DEFAULTS) - {"data"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    config = copy.deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if key in config and isinstance(config[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping.")
            unknown = set(value) - set(config[key])
            if unknown:
                raise ValueError(f"Unknown {key} options: {sorted(unknown)}")
            config[key].update(value)
        else:
            config[key] = value
    if "data" not in config:
        raise ValueError("The data section is required.")
    if not isinstance(config["data"], dict):
        raise ValueError("data must be a mapping.")
    if config["data"].get("invalid_samples", "error") not in {"error", "drop"}:
        raise ValueError("data.invalid_samples must be error or drop.")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if config["cv"]["strategy"] not in {"stratified", "group", "spatial"}:
        raise ValueError("cv.strategy must be stratified, group, or spatial.")
    if config["cv"]["folds"] < 2:
        raise ValueError("cv.folds must be at least 2.")
    if config["preprocess"]["scaling"] not in {"none", "center", "standard"}:
        raise ValueError("preprocess.scaling must be none, center, or standard.")
    if config["model"]["seed_mode"] not in {"fixed", "per_draw"}:
        raise ValueError("model.seed_mode must be fixed or per_draw.")
    if not 0 < config["sampling"]["coverage"] < 1:
        raise ValueError("sampling.coverage must lie strictly between zero and one.")
    for section, key in [("optimizer", "screening_rounds"), ("sampling", "max_rounds"),
                         ("prediction", "batch_size"), ("explain", "bins")]:
        value = config[section][key]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{section}.{key} must be a positive integer.")
    rounds = config["sampling"]["rounds"]
    if rounds is not None and (not isinstance(rounds, int) or rounds < 1):
        raise ValueError("sampling.rounds must be null or a positive integer.")
    maximum = config["explain"]["max_samples"]
    if maximum is not None and (not isinstance(maximum, int) or maximum < 1):
        raise ValueError("explain.max_samples must be null or a positive integer.")
    if not 0 <= config["explain"]["clip_percentile"] < 50:
        raise ValueError("explain.clip_percentile must be in [0, 50).")
    for section, key in [("sampling", "ratio"), ("training", "positive_weight")]:
        value = config[section][key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{key} must be finite and positive.")
    if config["plots"]["dpi"] < 1:
        raise ValueError("plots.dpi must be positive.")
    formats = config["plots"]["formats"]
    if not isinstance(formats, list) or not formats or set(formats) - {"png", "pdf", "svg", "tif", "tiff"}:
        raise ValueError("plots.formats must be a list of png/pdf/svg/tif/tiff.")
    if config["explain"]["method"] not in {"tree", "permutation"}:
        raise ValueError("explain.method must be tree or permutation.")
    if config["explain"]["interactions"] and config["explain"]["method"] != "tree":
        raise ValueError("SHAP interaction values require method: tree.")
    for key in ["background_samples", "top_interactions"]:
        if not isinstance(config["explain"][key], int) or config["explain"][key] < 1:
            raise ValueError(f"explain.{key} must be a positive integer.")
    config["_base_dir"] = str(path.parent)
    config["_source"] = str(path)
    return config


def resolve_path(config, path):
    return (Path(config["_base_dir"]) / path).resolve()
