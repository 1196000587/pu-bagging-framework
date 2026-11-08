"""Interchangeable maximizers over the same named parameter-space contract."""
from itertools import product
import json
import math
import numpy as np
import pandas as pd
from .config import import_object


def validate_space(space):
    if not isinstance(space, dict):
        raise ValueError("optimizer.space must be a mapping.")
    for key, spec in space.items():
        if key not in {"sampling.ratio", "training.positive_weight"} and not key.startswith("model."):
            raise ValueError(f"Unsupported search parameter: {key}")
        kind = spec.get("type")
        if kind == "categorical":
            if not isinstance(spec.get("choices"), list) or not spec["choices"]:
                raise ValueError(f"{key}: nonempty choices are required.")
        elif kind in {"int", "float"}:
            low, high = spec["low"], spec["high"]
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError(f"{key}: invalid bounds.")
            if kind == "int" and (int(low) != low or int(high) != high):
                raise ValueError(f"{key}: integer bounds required.")
            if spec.get("log") and low <= 0:
                raise ValueError(f"{key}: log bounds must be positive.")
        else:
            raise ValueError(f"{key}: type must be int, float, or categorical.")


def decode(vector, space):
    result = {}
    for value, (key, spec) in zip(vector, space.items()):
        value = float(np.clip(value, 0, 1))
        kind = spec["type"]
        if kind == "categorical":
            choices = spec["choices"]
            result[key] = choices[min(int(value * len(choices)), len(choices) - 1)]
        else:
            low, high = spec["low"], spec["high"]
            raw = math.exp(math.log(low) + value * (math.log(high) - math.log(low))) if spec.get("log") else low + value * (high - low)
            result[key] = int(np.clip(round(raw), low, high)) if kind == "int" else float(raw)
    return result


def candidate_settings(config, candidate):
    params, ratio, weight = {}, config["sampling"]["ratio"], config["training"]["positive_weight"]
    for key, value in candidate.items():
        if key == "sampling.ratio":
            ratio = float(value)
        elif key == "training.positive_weight":
            weight = float(value)
        elif key.startswith("model."):
            params[key[6:]] = value
        else:
            raise ValueError(f"Unknown candidate parameter: {key}")
    return params, float(ratio), float(weight)


def _random(objective, space, options, seed):
    rng = np.random.default_rng(seed)
    trials = int(options.get("n_trials", 30))
    if trials < 1:
        raise ValueError("n_trials must be positive.")
    best, score = None, -np.inf
    for _ in range(trials):
        candidate = decode(rng.random(len(space)), space)
        value = objective(candidate)
        if value > score:
            best, score = candidate, value
    return best


def _grid(objective, space, options, seed):
    choices = []
    for key, spec in space.items():
        if spec["type"] == "categorical":
            values = spec["choices"]
        elif "values" in spec:
            values = spec["values"]
        elif spec["type"] == "int":
            values = range(spec["low"], spec["high"] + 1)
        else:
            raise ValueError(f"Grid search needs explicit values for continuous parameter {key}.")
        if len(values) == 0:
            raise ValueError(f"Grid values for {key} must be nonempty.")
        if spec["type"] != "categorical" and any(v < spec["low"] or v > spec["high"] for v in values):
            raise ValueError(f"Grid values outside declared bounds for {key}.")
        choices.append(values)
    total = math.prod(map(len, choices))
    if total > options.get("max_candidates", 1000):
        raise ValueError(f"Grid has {total} candidates; reduce it or raise max_candidates explicitly.")
    best, score = None, -np.inf
    for values in product(*choices):
        candidate = dict(zip(space, values))
        value = objective(candidate)
        if value > score:
            best, score = candidate, value
    return best


def _genetic(objective, space, options, seed):
    """Tournament selection, simulated-binary crossover, polynomial mutation, elitism."""
    rng = np.random.default_rng(seed)
    size, generations = int(options.get("population_size", 20)), int(options.get("generations", 20))
    if size < 2 or generations < 1:
        raise ValueError("GA needs population_size >= 2 and generations >= 1.")
    crossover = float(options.get("crossover_probability", 0.9))
    mutation = float(options.get("mutation_probability", 0.1))
    eta_c, eta_m = float(options.get("eta_c", 15)), float(options.get("eta_m", 20))
    if not 0 <= crossover <= 1 or not 0 <= mutation <= 1 or min(eta_c, eta_m) <= 0:
        raise ValueError("Invalid GA probabilities or distribution indices.")
    pop = rng.random((size, len(space)))
    scores = np.array([objective(decode(row, space)) for row in pop])
    stall, previous = 0, float(scores.max())
    for _ in range(generations):
        children = []
        while len(children) < size:
            def select():
                a, b = rng.integers(size, size=2)
                return pop[a if scores[a] >= scores[b] else b].copy()
            first, second = select(), select()
            if rng.random() < crossover:
                u = rng.random(len(space))
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                first, second = (0.5 * ((1 + beta) * first + (1 - beta) * second),
                                 0.5 * ((1 - beta) * first + (1 + beta) * second))
            for child in (first, second):
                u = rng.random(len(space))
                delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1,
                                 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
                child += delta * (rng.random(len(space)) < mutation)
                children.append(np.clip(child, 0, 1))
        children = np.asarray(children[:size])  # supports odd population sizes
        child_scores = np.array([objective(decode(row, space)) for row in children])
        all_pop, all_scores = np.vstack([pop, children]), np.r_[scores, child_scores]
        order = np.argsort(-all_scores, kind="stable")[:size]
        pop, scores = all_pop[order], all_scores[order]
        if scores[0] - previous <= options.get("tolerance", 1e-7):
            stall += 1
        else:
            stall = 0
        previous = float(scores[0])
        if stall >= options.get("patience", 10):
            break
    return decode(pop[np.argmax(scores)], space)


def _optuna(objective, space, options, seed):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError('Optuna search requires pip install ".[optuna]"') from exc
    def trial_objective(trial):
        candidate = {}
        for key, spec in space.items():
            if spec["type"] == "categorical":
                # Index-based categories also support dict-valued class weights.
                index = trial.suggest_int(key, 0, len(spec["choices"]) - 1)
                candidate[key] = spec["choices"][index]
            else:
                method = trial.suggest_int if spec["type"] == "int" else trial.suggest_float
                candidate[key] = method(key, spec["low"], spec["high"], log=spec.get("log", False))
        trial.set_user_attr("candidate", candidate)
        return objective(candidate)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(trial_objective, n_trials=int(options.get("n_trials", 30)), n_jobs=1)
    return study.best_trial.user_attrs["candidate"]


def optimize(objective, settings, seed, progress=None):
    """Custom optimizer: f(objective, space, options, seed) -> candidate dict.

    Objective is maximized; errors fail loudly instead of hiding bad fits/folds.
    Identical parameter dictionaries are evaluated once using common draw seeds.
    """
    space, name = settings["space"], settings["name"]
    if name == "none":
        if space:
            raise ValueError("optimizer none cannot have a nonempty search space.")
        return {}, pd.DataFrame(columns=["evaluation", "mean_cv_auc", "params", "best_so_far"])
    validate_space(space)
    if not space:
        raise ValueError("A nonempty optimizer.space is required when optimizing.")
    history, cache = [], {}
    def cached(candidate):
        if set(candidate) != set(space):
            raise ValueError("Optimizer candidate keys do not match search-space keys.")
        for key, spec in space.items():
            value = candidate[key]
            if spec["type"] == "categorical":
                if value not in spec["choices"]:
                    raise ValueError(f"Invalid category for {key}.")
            elif not np.isfinite(value) or not spec["low"] <= value <= spec["high"] or (spec["type"] == "int" and int(value) != value):
                raise ValueError(f"Candidate value outside {key} domain.")
        key = json.dumps(candidate, sort_keys=True)
        if key not in cache:
            score = float(objective(candidate))
            if not np.isfinite(score):
                raise ValueError("Optimizer objective must return a finite score.")
            cache[key] = score
            history.append({"evaluation": len(history) + 1, "mean_cv_auc": score,
                            "params": key, "best_so_far": max(score, max(cache.values()))})
            if progress:
                progress(history[-1])
        return cache[key]
    registry = {"random": _random, "grid": _grid, "genetic": _genetic, "optuna": _optuna}
    runner = registry[name] if name in registry else import_object(name)
    best = runner(cached, space, settings["options"], seed)
    cached(best)
    if cached(best) < max(cache.values()):
        raise ValueError("Custom optimizer did not return its best evaluated candidate.")
    return best, pd.DataFrame(history)
