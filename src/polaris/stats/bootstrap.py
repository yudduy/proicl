"""Bootstrap CI over per-query correctness (migrated from io/metrics.py)."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[bool], resamples: int = 1000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    samples = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
