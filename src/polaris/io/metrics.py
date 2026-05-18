from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PassSummary:
    condition: str
    n_problems: int
    pass_rate: float
    ci_low: float
    ci_high: float
    total_cost: float
    cost_normalized_accuracy: float


def selected_correctness(
    rows: list[dict[str, Any]],
    pass_key: str = "humaneval_plus_pass",
) -> dict[str, dict[str, bool]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["condition"]][row["problem_id"]].append(row)

    selected: dict[str, dict[str, bool]] = {}
    for condition, by_problem in grouped.items():
        selected[condition] = {}
        for problem_id, problem_rows in by_problem.items():
            selected[condition][problem_id] = any(bool(r.get(pass_key)) for r in problem_rows)
    return selected


def bootstrap_ci(values: list[bool], resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    samples = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_conditions(
    rows: list[dict[str, Any]],
    pass_key: str = "humaneval_plus_pass",
    resamples: int = 1000,
) -> list[PassSummary]:
    selected = selected_correctness(rows, pass_key=pass_key)
    cost_by_condition: dict[str, float] = defaultdict(float)
    for row in rows:
        cost_by_condition[row["condition"]] += float(row.get("estimated_dollar_cost", 0.0))

    summaries: list[PassSummary] = []
    for condition in sorted(selected):
        values = list(selected[condition].values())
        pass_rate = float(np.mean(values)) if values else float("nan")
        ci_low, ci_high = bootstrap_ci(values, resamples=resamples, seed=0)
        total_cost = cost_by_condition[condition]
        summaries.append(
            PassSummary(
                condition=condition,
                n_problems=len(values),
                pass_rate=pass_rate,
                ci_low=ci_low,
                ci_high=ci_high,
                total_cost=total_cost,
                cost_normalized_accuracy=pass_rate / total_cost if total_cost > 0 else float("inf"),
            )
        )
    return summaries


def mcnemar(
    selected: dict[str, dict[str, bool]],
    condition_a: str,
    condition_b: str,
) -> dict[str, Any]:
    common = sorted(set(selected[condition_a]) & set(selected[condition_b]))
    b = sum(selected[condition_a][pid] and not selected[condition_b][pid] for pid in common)
    c = sum((not selected[condition_a][pid]) and selected[condition_b][pid] for pid in common)
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
        except Exception:
            tail = sum(_comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
            p_value = min(1.0, 2 * tail)
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "n_discordant": n,
        "p_value": p_value,
    }


def _comb(n: int, k: int) -> int:
    import math

    return math.comb(n, k)


def archive_utilization(rows: list[dict[str, Any]], condition: str = "T4") -> dict[str, float]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            by_problem[row["problem_id"]].append(row)

    counts: dict[str, int] = defaultdict(int)
    for problem_rows in by_problem.values():
        passing = [r for r in problem_rows if r.get("humaneval_plus_pass")]
        chosen = passing[0] if passing else problem_rows[0]
        counts[str(chosen.get("prompt_id", "unknown"))] += 1

    total = sum(counts.values())
    return {key: count / total for key, count in sorted(counts.items())} if total else {}


def composition_ratio(summaries: list[PassSummary]) -> float | None:
    by_condition = {summary.condition: summary.pass_rate for summary in summaries}
    denominator = by_condition.get("T3", 0.0) - by_condition.get("T2", 0.0)
    numerator = by_condition.get("T4", 0.0) - by_condition.get("T3", 0.0)
    if denominator <= 0:
        return None
    return numerator / denominator
