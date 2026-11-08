"""Example plugin optimizer; run from repository root via python -m pu_bagging."""
import numpy as np
from pu_bagging.optimizers import decode


def midpoint_then_random(objective, space, options, seed):
    """Maximize objective, return a candidate with EXACTLY the search-space keys."""
    rng = np.random.default_rng(seed)
    candidates = [decode(np.full(len(space), .5), space)]
    candidates += [decode(rng.random(len(space)), space) for _ in range(options.get("n_trials", 10))]
    return max(candidates, key=objective)
