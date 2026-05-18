"""2×2×2 factorial regression over {archive, sharpening, memory} (PROPOSAL §6.5).

Fit per-query correctness on the three binary factors and their interactions
on a logit scale. The three-way interaction term is the load-bearing
quantity for the "complementary mechanisms" claim. Bootstrap CIs are
clustered by problem_id. The current implementation is a small ridge-logit fit
for infrastructure smokes; full statistical reporting can layer on this schema.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


TERMS = (
    "intercept",
    "archive",
    "sharpening",
    "memory",
    "archive:sharpening",
    "archive:memory",
    "sharpening:memory",
    "archive:sharpening:memory",
)


def _design(row: dict[str, Any]) -> list[float]:
    a = 1.0 if row["archive"] else 0.0
    s = 1.0 if row["sharpening"] else 0.0
    m = 1.0 if row["memory"] else 0.0
    return [1.0, a, s, m, a * s, a * m, s * m, a * s * m]


def _fit_ridge_logit(rows: list[dict[str, Any]]) -> dict[str, float]:
    x = np.asarray([_design(row) for row in rows], dtype=float)
    y = np.asarray([1.0 if row["passed"] else 0.0 for row in rows], dtype=float)
    try:
        from scipy.optimize import minimize

        def objective(beta):
            z = x @ beta
            # stable logistic negative log likelihood + tiny ridge.
            nll = np.sum(np.logaddexp(0, z) - y * z)
            return float(nll + 1e-3 * np.sum(beta * beta))

        result = minimize(objective, np.zeros(x.shape[1]), method="BFGS")
        beta = result.x if result.success else np.linalg.pinv(x) @ _logit_smoothed(y)
    except Exception:
        beta = np.linalg.pinv(x) @ _logit_smoothed(y)
    return {term: float(value) for term, value in zip(TERMS, beta)}


def _logit_smoothed(y):
    eps = 1e-3
    clipped = np.clip(y, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _cluster_resample(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(str(row["problem_id"]), []).append(row)
    problem_ids = list(by_problem)
    sampled: list[dict[str, Any]] = []
    for _ in problem_ids:
        sampled.extend(by_problem[rng.choice(problem_ids)])
    return sampled


def fit_factorial_logit(
    rows: list[dict[str, Any]],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit y ~ archive * sharpening * memory on per-query correctness.

    Each row needs: problem_id, archive (bool), sharpening (bool), memory (bool),
    passed (bool). Returns coefficients (main + 2-way + 3-way), bootstrap CIs
    clustered by problem_id, and the logit-scale three-way interaction summary.

    """
    if not rows:
        raise ValueError("fit_factorial_logit requires at least one row")
    for row in rows:
        missing = {"problem_id", "archive", "sharpening", "memory", "passed"} - set(row)
        if missing:
            raise ValueError(f"factorial row missing keys: {sorted(missing)}")

    coefficients = _fit_ridge_logit(rows)
    rng = random.Random(seed)
    boot_values: list[float] = []
    for _ in range(max(0, bootstrap_resamples)):
        sampled = _cluster_resample(rows, rng)
        boot_values.append(_fit_ridge_logit(sampled)["archive:sharpening:memory"])
    if boot_values:
        lo, hi = np.percentile(np.asarray(boot_values), [2.5, 97.5])
        ci95 = [float(lo), float(hi)]
    else:
        v = coefficients["archive:sharpening:memory"]
        ci95 = [v, v]
    return {
        "model": "ridge_logit",
        "terms": list(TERMS),
        "coefficients": coefficients,
        "three_way": {
            "term": "archive:sharpening:memory",
            "estimate": coefficients["archive:sharpening:memory"],
            "ci95": ci95,
            "bootstrap_resamples": bootstrap_resamples,
            "cluster": "problem_id",
        },
    }
