"""Repeated internal CV. Labels compare known positives against sampled unlabeled rows."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from .models import build_model, positive_probability, fit_classifier
from .sampling import draw_indices, seed_for


def training_draw(data, ratio, seed):
    ix = draw_indices(len(data.positive), len(data.unlabeled), ratio, seed)
    X = np.vstack([data.positive, data.unlabeled[ix]])
    y = np.r_[np.ones(len(data.positive), dtype=int), np.zeros(len(ix), dtype=int)]
    groups = None
    if data.positive_groups is not None:
        groups = np.r_[data.positive_groups, data.unlabeled_groups[ix]]
    return X, y, groups, ix


def split_folds(X, y, groups, cv, seed):
    n = cv["folds"]
    if min(np.bincount(y, minlength=2)) < n:
        raise ValueError(f"Both classes need at least {n} samples for {n}-fold CV.")
    if cv["strategy"] == "stratified":
        splits = StratifiedKFold(n, shuffle=True, random_state=seed).split(X, y)
    else:
        if groups is None or len(np.unique(groups)) < n:
            raise ValueError("Insufficient spatial groups; change block size/folds or provide group data.")
        splits = GroupKFold(n).split(X, y, groups)
    result = list(splits)
    for train, valid in result:
        if len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
            raise ValueError("A CV fold has only one class; adjust spatial blocks, folds, or sampling ratio. No fold is silently skipped.")
    return result


def evaluate(data, config, params, ratio, rounds, stage=10, positive_weight=1.0, progress=None):
    folds, draws, curves = [], [], []
    grid = np.linspace(0, 1, 101)
    for draw in range(rounds):
        seed = seed_for(config["seed"], stage, draw)
        X, y, groups, _ = training_draw(data, ratio, seed)
        splits = split_folds(X, y, groups, config["cv"], seed_for(config["seed"], stage, 0, 999))
        oof = np.empty(len(y))
        fold_scores, fold_curves = [], []
        for fold, (train, valid) in enumerate(splits):
            model_draw = draw if config["model"]["seed_mode"] == "per_draw" else 0
            model = build_model(config["model"], config["preprocess"], data.feature_names,
                                params, seed_for(config["seed"], 100, model_draw, fold + 1))
            fit_classifier(model, X[train], y[train], positive_weight)
            probability = positive_probability(model, X[valid])
            oof[valid] = probability
            auc = float(roc_auc_score(y[valid], probability))
            fpr, tpr, _ = roc_curve(y[valid], probability)
            interp = np.interp(grid, fpr, tpr)
            interp[0], interp[-1] = 0, 1
            fold_scores.append(auc)
            fold_curves.append(interp)
            folds.append({"draw": draw + 1, "fold": fold + 1, "auc": auc,
                          "n_train": len(train), "n_valid": len(valid), "seed": seed})
        draws.append({"draw": draw + 1, "mean_fold_auc": float(np.mean(fold_scores)),
                      "oof_auc": float(roc_auc_score(y, oof)), "seed": seed})
        curves.append(np.mean(fold_curves, axis=0))
        if progress:
            progress(draws[-1])
    values = np.asarray([r["mean_fold_auc"] for r in draws])
    return {"mean_auc": float(values.mean()), "std_auc": float(values.std(ddof=0)),
            "draws": pd.DataFrame(draws), "folds": pd.DataFrame(folds),
            "roc": pd.DataFrame({"fpr": grid, "mean_tpr": np.mean(curves, axis=0),
                                 "std_tpr": np.std(curves, axis=0)})}
