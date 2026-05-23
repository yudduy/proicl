"""Materialize a CPU-only ProICL artifact-contract smoke.

This script does not generate model outputs and must not be used as scientific
evidence. It exists to prove that the ProICL run layout, condition-specific
trace files, memory ledgers, aggregate analysis, and artifact audit are wired
end to end before a paid GPU launch.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_TRACKS = (
    "reasoning_gym_family_relationships",
    "reasoning_gym_graph_color_n10",
    "reasoning_gym_boxnet",
)

DEFAULT_CONDITIONS = (
    "base_greedy",
    "bon_temp1",
    "mcmc_only",
    "mixed_alpha_mcmc",
    "fork_search",
    "gepa_only",
    "gepa_mcmc",
    "gepa_mcmc_repair",
    "gepa_mcmc_fork_repair",
    "gepa_mcmc_fork_repair_memory",
    "gepa_mcmc_memory",
    "prorl_v2_greedy",
)

ACCURACY_BY_CONDITION = {
    "base_greedy": 0.20,
    "bon_temp1": 0.25,
    "sps_only": 0.30,
    "mcmc_only": 0.30,
    "mixed_alpha_mcmc": 0.35,
    "fork_search": 0.32,
    "gepa_only": 0.40,
    "gepa_sps_fixed": 0.45,
    "gepa_mcmc": 0.45,
    "gepa_mcmc_repair": 0.48,
    "gepa_mcmc_fork_repair": 0.50,
    "gepa_mcmc_fork_repair_memory": 0.55,
    "gepa_mcmc_memory": 0.52,
    "prorl_v2_greedy": 0.60,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", type=Path, default=Path("runs/proicl_experiments"))
    parser.add_argument("--run-tag", default="artifact-contract")
    parser.add_argument("--tracks", nargs="+", default=list(DEFAULT_TRACKS))
    parser.add_argument("--archive-train-tracks", nargs="+", default=None)
    parser.add_argument("--archive-heldout-tracks", nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument(
        "--archive-scope",
        choices=["transductive_support", "within_family", "cross_family_curriculum"],
        default="within_family",
    )
    parser.add_argument("--split", type=int, nargs=2, default=(0, 2))
    parser.add_argument("--rollout-budget", type=int, default=2)
    parser.add_argument("--archive-size", type=int, default=2)
    parser.add_argument("--max-metric-calls", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mcmc-steps", type=int, default=2)
    parser.add_argument("--mcmc-block-num", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--memory-num-shards", type=int, default=1)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        row = {"event": event, **payload}
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_memory_sqlite(path: Path, *, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE memory_events (id TEXT PRIMARY KEY, track TEXT, condition TEXT, admitted INTEGER)"
        )
        conn.execute(
            "INSERT INTO memory_events VALUES (?, ?, ?, ?)",
            (
                f"{cell['track']}:{cell['proicl_condition']}:fixture",
                cell["track"],
                cell["proicl_condition"],
                1,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _write_cell_artifacts(cell: dict[str, Any]) -> None:
    out = Path(cell["artifact_dir"])
    out.mkdir(parents=True, exist_ok=True)
    condition = str(cell["proicl_condition"])
    accuracy = ACCURACY_BY_CONDITION[condition]
    n_problems = max(1, int(cell["split"][1]) - int(cell["split"][0]))
    common = {
        "artifact_contract_smoke": True,
        "track": cell["track"],
        "condition": condition,
        "shard_id": cell["shard_id"],
    }
    _write_json(out / "manifest.json", {"config": common, "cell": cell})
    _write_json(out / "archive.json", {"entries": [], **common})
    _write_jsonl(out / "candidates.jsonl", [{**common, "candidate_id": "fixture:0", "response": "fixture"}])
    _write_jsonl(out / "scores.jsonl", [{**common, "candidate_id": "fixture:0", "score": accuracy}])
    _write_jsonl(out / "selected.jsonl", [{**common, "candidate_id": "fixture:0", "selected": True}])
    _write_json(
        out / "metrics.json",
        {
            **common,
            "accuracy": accuracy,
            "n_problems": n_problems,
            "passed": int(round(accuracy * n_problems)),
        },
    )
    _write_json(out / "costs.json", {**common, "estimated_dollar_cost": 0.0})
    _write_json(out / "rollouts.json", {**common, "rollout_budget": cell["rollout_budget"]})
    _write_json(out / "preflight.json", {**common, "passed": True, "paid_run": False})
    _write_json(out / "environment.json", {**common, "backend": "fixture"})
    _write_json(out / "run_plan_cell.json", cell)
    (out / "audit.md").write_text("# Artifact Contract Smoke\n\npassed: true\n", encoding="utf-8")
    if cell.get("uses_gepa_archive") or cell.get("uses_memory"):
        _write_json(out / "archive_build_manifest.json", {**common, "archive_build_id": cell["archive_build_id"]})
    if cell.get("uses_repair"):
        _write_jsonl(out / "repair_traces.jsonl", [{**common, "repair_attempt": 1, "passed": True}])
    if cell.get("uses_fork_search"):
        _write_jsonl(out / "fork_traces.jsonl", [{**common, "fork_depth": 1, "fork_entropy": 1.23}])
    if cell.get("uses_memory"):
        _write_memory_sqlite(out / "memory.sqlite", cell=cell)
        _write_jsonl(out / "memory_events.jsonl", [{**common, "event": "admit", "admitted": True}])


def main() -> None:
    from polaris.proicl.analysis import write_proicl_decomposition_by_track
    from polaris.proicl.artifact_audit import write_proicl_artifact_audit
    from polaris.proicl.launcher import build_signal_cells, write_json
    from polaris.proicl.naming import (
        make_proicl_run_identity,
        standard_run_root,
        write_run_index,
    )

    args = _parse_args()
    identity = make_proicl_run_identity(
        run_stage="artifact_contract_smoke",
        tracks=args.tracks,
        archive_scope=args.archive_scope,
        backend="fixture",
        tag=args.run_tag,
    )
    run_root = standard_run_root(args.series_root, identity)
    full_root = run_root / "full"
    full_root.mkdir(parents=True, exist_ok=True)
    archive_train_tracks = tuple(args.archive_train_tracks or args.tracks)
    archive_heldout_tracks = tuple(args.archive_heldout_tracks or args.tracks)
    write_run_index(
        run_root / "run_index.json",
        identity=identity,
        tracks=args.tracks,
        conditions=args.conditions,
        split=tuple(args.split),
        rollout_budget=args.rollout_budget,
        archive_size=args.archive_size,
        max_metric_calls=args.max_metric_calls,
        max_new_tokens=args.max_new_tokens,
        mcmc_steps=args.mcmc_steps,
        mcmc_block_num=args.mcmc_block_num,
        num_shards=args.num_shards,
        memory_num_shards=args.memory_num_shards,
        reflection_provider="fixture",
        reflection_model_id="fixture",
        run_kind="local",
        cost_cap_dollars=0.0,
        notes="CPU-only artifact-contract smoke; not scientific evidence.",
    )
    cells = build_signal_cells(
        root=full_root,
        tracks=args.tracks,
        split=tuple(args.split),
        rollout_budget=args.rollout_budget,
        num_shards=args.num_shards,
        memory_num_shards=args.memory_num_shards,
        archive_scope=args.archive_scope,
        archive_train_tracks=archive_train_tracks,
        archive_heldout_tracks=archive_heldout_tracks,
        conditions=args.conditions,
    )
    plan_path = full_root / "proicl_signal_plan.json"
    write_json(plan_path, [cell.to_jsonable() for cell in cells])
    _append_event(full_root / "events.jsonl", "queue_start", cells=len(cells))
    for cell in cells:
        _write_cell_artifacts(cell.to_jsonable())
        _append_event(
            full_root / "events.jsonl",
            "cell_done",
            track=cell.track,
            condition=cell.proicl_condition,
            shard=cell.shard_id,
        )
    _append_event(full_root / "events.jsonl", "queue_done", failures=0)
    report = write_proicl_decomposition_by_track(
        root=full_root,
        tracks=tuple(args.tracks),
        out_dir=full_root / "analysis",
    )
    _write_json(full_root / "analysis" / "aggregate_stdout.json", report)
    audit = write_proicl_artifact_audit(plan_path, full_root / "analysis")
    _append_event(full_root / "events.jsonl", "aggregate_done", out=str(full_root / "analysis"))
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "plan_path": str(plan_path),
                "artifact_audit_passed": audit["passed"],
                "total_cells": audit["total_cells"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
