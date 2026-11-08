"""Optional SHAP, pooled quantile bins, and per-fit importance distributions."""
from pathlib import Path
import numpy as np
import pandas as pd
from .models import positive_probability


def quantile_statistics(x, values, bins=20, clip_percentile=1.0):
    x, values = np.asarray(x), np.asarray(values)
    good = np.isfinite(x) & np.isfinite(values)
    x, values = x[good], values[good]
    if not len(x):
        return pd.DataFrame()
    low, high = np.percentile(x, [clip_percentile, 100 - clip_percentile])
    keep = (x >= low) & (x <= high)
    x, values = x[keep], values[keep]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        edges = np.repeat(edges, 2)
    rows = []
    for i in range(len(edges) - 1):
        mask = (x >= edges[i]) & ((x <= edges[i+1]) if i == len(edges)-2 else (x < edges[i+1]))
        selected = values[mask]
        if len(selected):
            rows.append({"bin": i + 1, "left": edges[i], "right": edges[i+1],
                         "center": float((edges[i] + edges[i+1]) / 2), "n": len(selected),
                         "min": selected.min(), "q25": np.quantile(selected, .25),
                         "median": np.median(selected), "q75": np.quantile(selected, .75),
                         "max": selected.max(), "mean": selected.mean(), "std": selected.std(ddof=0)})
    return pd.DataFrame(rows)


def _positive_shap(values, positive_index, interaction=False):
    if isinstance(values, list):
        return np.asarray(values[positive_index])
    values = np.asarray(values)
    expected = 3 if interaction else 2
    if values.ndim == expected + 1:
        return values[..., positive_index]
    if values.ndim == expected:
        return values
    raise ValueError(f"Unsupported SHAP array shape {values.shape}.")


class ExplanationCollector:
    def __init__(self, directory, names, settings):
        self.directory, self.names, self.settings = Path(directory), names, settings
        self.directory.mkdir()
        self.importance, self.interactions, self.files, self.metadata = [], [], [], []

    def add(self, pipeline, X, ids, draw, seed):
        try:
            import shap
        except ImportError as exc:
            raise ImportError('SHAP requires pip install ".[explain]"') from exc
        rng = np.random.default_rng(seed)
        indices = np.arange(len(X))
        cap = self.settings["max_samples"]
        if cap is not None and len(indices) > cap:
            indices = np.sort(rng.choice(len(X), cap, replace=False))
        raw = X[indices]
        estimator = pipeline.named_steps["model"]
        positive_index = list(pipeline.classes_).index(1)
        interaction = None
        if self.settings["method"] == "tree":
            transformed = pipeline[:-1].transform(raw)
            explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
            values = _positive_shap(explainer.shap_values(transformed), positive_index)
            output_units = "tree native output (RF: probability; XGBoost: raw margin)"
            if self.settings["interactions"]:
                interaction = _positive_shap(explainer.shap_interaction_values(transformed), positive_index, True)
        elif self.settings["method"] == "permutation":
            if self.settings["interactions"]:
                raise ValueError("SHAP interactions require explain.method: tree.")
            background = X[rng.choice(len(X), min(len(X), self.settings["background_samples"]), replace=False)]
            explainer = shap.Explainer(lambda a: positive_probability(pipeline, a), background,
                                       algorithm="permutation", seed=seed)
            values = np.asarray(explainer(raw, max_evals=2 * X.shape[1] + 1).values)
            output_units = "positive-class probability"
        else:
            raise ValueError("explain.method must be tree or permutation.")
        if values.shape != raw.shape or not np.isfinite(values).all():
            raise ValueError("SHAP values must be finite and match the feature matrix.")
        record = {"draw": draw, **dict(zip(self.names, np.abs(values).mean(axis=0)))}
        self.importance.append(record)
        payload = {"X": raw, "shap": values, "ids": np.asarray(ids[indices], dtype=str)}
        if interaction is not None:
            if interaction.shape != (len(raw), len(self.names), len(self.names)):
                raise ValueError("Unexpected SHAP interaction dimensions.")
            if not np.isfinite(interaction).all():
                raise ValueError("Non-finite SHAP interaction values.")
            payload["interactions"] = interaction
            self.interactions.append(np.abs(interaction).mean(axis=0))
        path = self.directory / f"draw_{draw:05d}.npz"
        np.savez_compressed(path, **payload)
        self.files.append(path)
        self.metadata.append({"draw": draw, "explained_rows": len(raw), "training_rows": len(X),
                              "output_units": output_units, "method": self.settings["method"]})

    def _pooled(self, feature, interaction_with=None):
        xs, values = [], []
        for path in self.files:
            with np.load(path, allow_pickle=False) as item:
                xs.append(item["X"][:, feature])
                values.append(item["shap"][:, feature] if interaction_with is None
                              else item["interactions"][:, feature, interaction_with])
        return np.concatenate(xs), np.concatenate(values)

    def summarize(self):
        importance = pd.DataFrame(self.importance)
        scores = importance[self.names]
        summary = pd.DataFrame({"feature": self.names, "mean_abs_shap": scores.mean().to_numpy(),
                                "std_abs_shap": scores.std(ddof=0).to_numpy()})
        tables = {"shap_importance_runs": importance,
                  "shap_importance": summary.sort_values("mean_abs_shap", ascending=False),
                  "shap_metadata": pd.DataFrame(self.metadata)}
        bins = []
        for index, feature in enumerate(self.names):
            x, value = self._pooled(index)
            table = quantile_statistics(x, value, self.settings["bins"], self.settings["clip_percentile"])
            table.insert(0, "feature", feature)
            bins.append(table)
        tables["shap_dependence"] = pd.concat(bins, ignore_index=True)
        if self.interactions:
            matrix = np.mean(self.interactions, axis=0)
            np.fill_diagonal(matrix, 0)
            tables["shap_interaction_matrix"] = pd.DataFrame(matrix, columns=self.names).assign(feature=self.names)
            pairs = sorted([(matrix[i, j], i, j) for i in range(len(self.names))
                            for j in range(i + 1, len(self.names))], reverse=True)
            interaction_bins = []
            for rank, (score, i, j) in enumerate(pairs[:self.settings["top_interactions"]], 1):
                x, value = self._pooled(i, j)
                table = quantile_statistics(x, value, self.settings["bins"], self.settings["clip_percentile"])
                table.insert(0, "feature", self.names[i])
                table.insert(1, "partner", self.names[j])
                table["rank"], table["mean_abs_interaction"] = rank, score
                interaction_bins.append(table)
            if interaction_bins:
                tables["shap_interaction_dependence"] = pd.concat(interaction_bins, ignore_index=True)
        return tables
