"""Estimator adapters. Hyperparameters use native sklearn parameter names."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import import_object


class FeatureMultiplier(TransformerMixin, BaseEstimator):
    def __init__(self, factors=None):
        self.factors = factors

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * np.asarray(self.factors, dtype=float)


def build_model(model_config, preprocess, feature_names, params, seed):
    name = model_config["name"]
    registry = {"random_forest": RandomForestClassifier, "extra_trees": ExtraTreesClassifier,
                "logistic_regression": LogisticRegression}
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('Install the optional dependency: pip install ".[xgboost]"') from exc
        cls = XGBClassifier
    elif name in registry:
        cls = registry[name]
    else:
        cls = import_object(name)
    estimator = cls()
    if not is_classifier(estimator) or not hasattr(estimator, "predict_proba"):
        raise TypeError("Models must be sklearn-compatible classifiers with predict_proba.")
    settings = dict(model_config.get("params", {}))
    settings.update(params)
    supported = estimator.get_params(deep=True)
    # The framework controls randomness, so candidates cannot silently override it.
    if "random_state" in supported:
        settings["random_state"] = seed
    if "n_jobs" in supported and name != "logistic_regression":
        settings.setdefault("n_jobs", model_config.get("n_jobs", 1))
    if name == "logistic_regression":
        settings.setdefault("max_iter", 1000)
    estimator.set_params(**settings)
    scales = preprocess.get("feature_scales", {})
    unknown = set(scales) - set(feature_names)
    if unknown:
        raise ValueError(f"Unknown feature scales: {sorted(unknown)}")
    factors = np.asarray([scales.get(f, 1.0) for f in feature_names], dtype=float)
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("Feature scaling factors must be finite and positive.")
    scaler = preprocess["scaling"]
    steps = [("features", FeatureMultiplier(factors))]
    if scaler != "none":
        steps.append(("scale", StandardScaler(with_std=scaler == "standard")))
    steps.append(("model", estimator))
    return Pipeline(steps)


def positive_probability(model, X):
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("The fitted classifier must expose classes 0 and 1.")
    values = np.asarray(model.predict_proba(X), dtype=float)[:, classes.index(1)]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("predict_proba returned non-finite or out-of-range values.")
    return values


def fit_classifier(model, X, y, positive_weight=1.0):
    if not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("training.positive_weight must be finite and > 0.")
    if positive_weight == 1:
        return model.fit(X, y)
    weights = np.where(y == 1, positive_weight, 1.0)
    # Custom classifiers must accept sample_weight when positive_weight != 1.
    return model.fit(X, y, model__sample_weight=weights)
