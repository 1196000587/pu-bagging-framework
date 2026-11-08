"""Streaming ensemble moments and reusable saved-model prediction."""
from pathlib import Path
import json
import numpy as np
import joblib
from .models import positive_probability


class RunningMoments:
    def __init__(self, size):
        self.count = 0
        self.mean = np.zeros(size, dtype=float)
        self.m2 = np.zeros(size, dtype=float)

    def update(self, values):
        values = np.asarray(values)
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("Invalid prediction vector for ensemble aggregation.")
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)

    @property
    def std(self):
        if self.count == 0:
            raise ValueError("No ensemble members fitted.")
        return np.sqrt(np.maximum(self.m2 / self.count, 0))


def batched_probability(model, X, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    return np.concatenate([positive_probability(model, X[start:start + batch_size])
                           for start in range(0, len(X), batch_size)])


def predict_saved(run_directory, frame, batch_size=5000):
    """Predict a numeric feature DataFrame. Only load trusted local joblib files."""
    directory = Path(run_directory)
    manifest = json.loads((directory / "ensemble.json").read_text(encoding="utf-8"))
    X = frame[manifest["features"]].to_numpy(dtype=float)
    if not len(X) or not np.isfinite(X).all():
        raise ValueError("Prediction matrix must be nonempty and finite.")
    moments = RunningMoments(len(X))
    for relative in manifest["members"]:
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError("Model path leaves the run directory.")
        model = joblib.load(path)
        moments.update(batched_probability(model, X, batch_size))
    return moments.mean, moments.std
