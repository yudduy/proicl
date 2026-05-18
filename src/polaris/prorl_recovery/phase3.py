from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from polaris.prorl_recovery.protocol import (
    BRO_RL_MODEL_KEY,
    HIGH_BASE_LOGPROB_QUANTILE,
    PRORL_V2_MODEL_KEY,
)

PHASE3_SOLVER_KEYS = frozenset({PRORL_V2_MODEL_KEY, BRO_RL_MODEL_KEY})
PHASE3_BUCKETS = (
    "search_limited",
    "prompt_conditional",
    "memory_conditional",
    "weight_only",
)


def _problem_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("task_family") or row.get("track")), str(row["problem_id"]))


def _passed(row: dict[str, Any]) -> bool:
    verifier = row.get("verifier_result")
    verifier_passed = verifier.get("passed") if isinstance(verifier, dict) else False
    return bool(
        row.get("passed", row.get("verified", row.get("selected_passed", verifier_passed)))
    )


def derive_phase3_input_set(
    *,
    phase1_rows: Iterable[dict[str, Any]],
    rung7_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically select ProRL/BroRL-only items for Phase 3.

    Input criterion: ProRL v2 or BroRL solves; base rung-7 fails. The output is
    sorted and contains one row per `(task_family, problem_id, checkpoint)`.
    """

    rung7_passed = {_problem_key(row) for row in rung7_rows if _passed(row)}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in phase1_rows:
        checkpoint = str(row.get("checkpoint") or row.get("model_key"))
        if checkpoint not in PHASE3_SOLVER_KEYS or not _passed(row):
            continue
        task_family, problem_id = _problem_key(row)
        if (task_family, problem_id) in rung7_passed:
            continue
        key = (task_family, problem_id, checkpoint)
        selected[key] = {
            "task_family": task_family,
            "problem_id": problem_id,
            "checkpoint": checkpoint,
            "phase1_source_id": row.get("candidate_id") or row.get("row_id"),
            "base_rung7_failed": True,
        }
    return [selected[key] for key in sorted(selected)]


def _checkpoint(row: dict[str, Any]) -> str:
    return str(row.get("checkpoint") or row.get("model_key"))


def materialize_phase3_trajectories(
    *,
    phase3_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach exact successful ProRL/BroRL trajectories to Phase 3 items.

    `derive_phase3_input_set` identifies problem/checkpoint pairs. Phase 3.1
    needs the exact successful response and prompt from raw Phase 1 candidates
    so HF/RWS can score the trajectory under the frozen base.
    """

    needed = {
        (
            str(row["task_family"]),
            str(row["problem_id"]),
            str(row["checkpoint"]),
        )
        for row in phase3_rows
    }
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in candidate_rows:
        task_family, problem_id = _problem_key(row)
        key = (task_family, problem_id, _checkpoint(row))
        if key not in needed or key in selected or not _passed(row):
            continue
        candidate_id = str(
            row.get("candidate_id")
            or row.get("row_id")
            or f"{task_family}:{problem_id}:{key[2]}:{row.get('sample_index', 0)}"
        )
        selected[key] = {
            "row_id": candidate_id,
            "task_family": task_family,
            "problem_id": problem_id,
            "checkpoint": key[2],
            "prompt_text": str(row["prompt_text"]),
            "response_text": str(row.get("response_text") or row.get("generation", "")),
            "source_candidate_id": candidate_id,
            "sample_index": int(row.get("sample_index", 0)),
        }

    missing = [key for key in sorted(needed) if key not in selected]
    if missing:
        formatted = ", ".join("/".join(key) for key in missing[:10])
        extra = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise ValueError(f"missing successful Phase 1 trajectories: {formatted}{extra}")
    return [selected[key] for key in sorted(selected)]


def require_phase3_inputs(phase1_path: Path, rung7_path: Path) -> None:
    missing = [str(path) for path in (phase1_path, rung7_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required Phase 3 inputs: " + ", ".join(missing))


def write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def assign_high_base_logprob(
    *,
    base_generated_lp_means: dict[str, list[float]],
    trace_task_family: str,
    trace_lp_base_mean: float,
) -> bool:
    values = sorted(float(x) for x in base_generated_lp_means.get(trace_task_family, []))
    if not values:
        raise ValueError(f"missing base logprob reference values for {trace_task_family!r}")
    idx = min(len(values) - 1, int((len(values) - 1) * HIGH_BASE_LOGPROB_QUANTILE))
    threshold = values[idx]
    return float(trace_lp_base_mean) >= threshold


def assign_bucket(row: dict[str, Any]) -> str:
    """Apply the locked ordered bucket rules.

    Required booleans:
    - `high_base_logprob`
    - `prompt_variant_solves`
    - `memory_transplant_passes`
    """

    if bool(row.get("high_base_logprob")):
        return "search_limited"
    if bool(row.get("prompt_variant_solves")):
        return "prompt_conditional"
    if bool(row.get("memory_transplant_passes")):
        return "memory_conditional"
    return "weight_only"


def bucket_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in PHASE3_BUCKETS}
    for row in rows:
        counts[assign_bucket(row)] += 1
    return counts


def group_lp_means_by_family(rows: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family = str(row.get("task_family") or row.get("track"))
        grouped[family].append(float(row["lp_base_mean"]))
    return dict(grouped)
