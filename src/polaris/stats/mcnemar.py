"""Paired binary contrast (migrated from io/metrics.py).

Used for condition-vs-condition tests after `selected_correctness` reduces
each problem to a single per-condition pass/fail outcome.
"""

from __future__ import annotations

import math
from typing import Any


def mcnemar(
    selected: dict[str, dict[str, bool]],
    condition_a: str,
    condition_b: str,
) -> dict[str, Any]:
    common = sorted(set(selected[condition_a]) & set(selected[condition_b]))
    b = sum(
        selected[condition_a][pid] and not selected[condition_b][pid] for pid in common
    )
    c = sum(
        (not selected[condition_a][pid]) and selected[condition_b][pid]
        for pid in common
    )
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(
                binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
            )
        except Exception:
            tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
            p_value = min(1.0, 2 * tail)
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "n_discordant": n,
        "p_value": p_value,
    }
