"""Uniform sampling without replacement within a draw, with reuse across draws."""
import math
import numpy as np


def sample_count(n_positive: int, n_unlabeled: int, ratio: float) -> int:
    if n_positive < 1 or n_unlabeled < 1 or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Positive/unlabeled pools must be nonempty and ratio must be > 0.")
    return min(n_unlabeled, math.ceil(n_positive * ratio))


def expected_coverage(n_unlabeled: int, count: int, rounds: int) -> float:
    if not 1 <= count <= n_unlabeled or rounds < 1:
        raise ValueError("Require 1 <= count <= pool size and rounds >= 1.")
    if count == n_unlabeled:
        return 1.0
    return -math.expm1(rounds * math.log1p(-count / n_unlabeled))


def rounds_for_coverage(n_unlabeled: int, count: int, coverage: float = 0.9) -> int:
    """Smallest M with EXPECTED coverage >= target; not an observed guarantee."""
    if not 0 < coverage < 1 or not 1 <= count <= n_unlabeled:
        raise ValueError("Require 0 < coverage < 1 and 1 <= count <= pool size.")
    if count == n_unlabeled:
        return 1
    rounds = max(1, math.ceil(math.log1p(-coverage) / math.log1p(-count / n_unlabeled)))
    while rounds > 1 and expected_coverage(n_unlabeled, count, rounds - 1) >= coverage:
        rounds -= 1
    return rounds


def seed_for(seed: int, stage: int, draw: int = 0, fold: int = 0) -> int:
    return int(np.random.SeedSequence([seed, stage, draw, fold]).generate_state(1)[0])


def draw_indices(n_positive, n_unlabeled, ratio, seed):
    count = sample_count(n_positive, n_unlabeled, ratio)
    return np.random.default_rng(seed).choice(n_unlabeled, count, replace=False)
