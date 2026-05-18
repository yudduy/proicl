"""CloudRift driver for the ProRL recoverable-fraction audit.

This script is intentionally artifact-first. It can render the exact run plan,
execute one indexed cell, and aggregate Phase 1 candidates into the canonical
parquet file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    archives = sub.add_parser("write-archives")
    archives.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "prorl_recovery_archives")

    plan = sub.add_parser("plan")
    plan.add_argument("--phase", choices=["phase0", "phase1", "phase2"], required=True)
    plan.add_argument("--root", default="/workspace/polaris/runs/prorl_recovery")
    plan.add_argument("--problem-count", type=int, default=100)
    plan.add_argument("--num-shards", type=int, default=4)
    plan.add_argument("--samples-per-problem", type=int, default=128)
    plan.add_argument("--tracks", nargs="+", default=None)
    plan.add_argument("--include-gpqa", action="store_true")
    plan.add_argument("--phase1-results", type=Path, default=None)
    plan.add_argument("--out", type=Path, default=None)

    run_cell = sub.add_parser("run-cell")
    run_cell.add_argument("--phase", choices=["phase0", "phase1", "phase2"], required=True)
    run_cell.add_argument("--cell-index", type=int, default=None)
    run_cell.add_argument("--root", default="/workspace/polaris/runs/prorl_recovery")
    run_cell.add_argument("--problem-count", type=int, default=100)
    run_cell.add_argument("--num-shards", type=int, default=4)
    run_cell.add_argument("--samples-per-problem", type=int, default=128)
    run_cell.add_argument("--tracks", nargs="+", default=None)
    run_cell.add_argument("--include-gpqa", action="store_true")
    run_cell.add_argument("--phase1-results", type=Path, default=None)
    run_cell.add_argument(
        "--run-kind",
        choices=["cloudrift", "farmshare", "local", "phase", "modal", "mithril", "flow", "bulk"],
        default="cloudrift",
    )
    run_cell.add_argument("--estimated-dollar-cost", type=float, default=None)
    run_cell.add_argument("--estimated-wall-clock-seconds", type=float, default=None)
    run_cell.add_argument("--cost-cap-dollars", type=float, default=None)
    run_cell.add_argument("--user-authorized-paid-run", action="store_true")

    aggregate = sub.add_parser("aggregate-phase1")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)

    aggregate0 = sub.add_parser("aggregate-phase0")
    aggregate0.add_argument("--root", type=Path, required=True)
    aggregate0.add_argument("--out", type=Path, required=True)
    aggregate0.add_argument("--expected-problems", type=int, default=500)

    plan_rws = sub.add_parser("plan-rws-exact")
    plan_rws.add_argument("--root", default="/workspace/polaris/runs/prorl_recovery")
    plan_rws.add_argument("--num-shards", type=int, default=5)
    plan_rws.add_argument("--num-seeds", type=int, default=8)
    plan_rws.add_argument("--out", type=Path, default=None)

    run_rws = sub.add_parser("run-rws-exact-cell")
    run_rws.add_argument("--root", default="/workspace/polaris/runs/prorl_recovery")
    run_rws.add_argument("--cell-index", type=int, default=None)
    run_rws.add_argument("--num-shards", type=int, default=5)
    run_rws.add_argument("--num-seeds", type=int, default=8)

    aggregate_rws = sub.add_parser("aggregate-rws-exact")
    aggregate_rws.add_argument("--root", type=Path, required=True)
    aggregate_rws.add_argument("--out", type=Path, required=True)
    aggregate_rws.add_argument("--expected-problems", type=int, default=500)
    aggregate_rws.add_argument("--expected-shards", type=int, default=5)
    aggregate_rws.add_argument("--expected-seeds", type=int, default=8)

    audit = sub.add_parser("audit-plan")
    audit.add_argument("--phase", choices=["phase0", "phase1", "phase2"], required=True)
    audit.add_argument("--root", default="/workspace/polaris/runs/prorl_recovery")
    audit.add_argument("--problem-count", type=int, default=100)
    audit.add_argument("--num-shards", type=int, default=4)
    audit.add_argument("--samples-per-problem", type=int, default=128)
    audit.add_argument("--tracks", nargs="+", default=None)
    audit.add_argument("--include-gpqa", action="store_true")
    audit.add_argument("--phase1-results", type=Path, default=None)
    audit.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()
    if hasattr(args, "root"):
        if isinstance(args.root, Path):
            args.root = Path(os.path.expandvars(str(args.root)))
        else:
            args.root = os.path.expandvars(args.root)
    return args


def _tracks(include_gpqa: bool, requested: list[str] | None) -> tuple[str, ...]:
    from polaris.prorl_recovery.orchestration import selected_tracks
    from polaris.runners.condition_runner import TRACK_CONFIGS

    if requested:
        unknown = sorted(set(requested) - set(TRACK_CONFIGS))
        if unknown:
            raise SystemExit(f"unknown tracks: {', '.join(unknown)}")
        if "gpqa_diamond" in requested and not include_gpqa:
            raise SystemExit("gpqa_diamond requires --include-gpqa")
        return tuple(requested)
    return selected_tracks(include_gpqa=include_gpqa)


def _read_phase1_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"phase1 results not found: {path}")
    if path.suffix == ".parquet":
        import pandas as pd

        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise SystemExit(f"unsupported phase1 results shape: {path}")


def _selected_problem_ids_by_track(args: argparse.Namespace) -> dict[str, tuple[str, ...]] | None:
    if args.phase != "phase2":
        return None
    if args.phase1_results is None:
        raise SystemExit("phase2 planning requires --phase1-results")
    from polaris.prorl_recovery.orchestration import derive_prorl_only_problem_ids

    selected = derive_prorl_only_problem_ids(_read_phase1_rows(args.phase1_results))
    requested_tracks = set(_tracks(args.include_gpqa, args.tracks))
    return {track: ids for track, ids in selected.items() if track in requested_tracks}


def _cells(args: argparse.Namespace):
    from polaris.prorl_recovery.orchestration import phase0_cells, phase1_cells, phase2_cells

    if args.phase == "phase0":
        return phase0_cells(root=args.root, num_shards=args.num_shards)
    if args.phase == "phase1":
        return phase1_cells(
            root=args.root,
            problem_count=args.problem_count,
            tracks=_tracks(args.include_gpqa, args.tracks),
            num_shards=args.num_shards,
            samples_per_problem=args.samples_per_problem,
        )
    return phase2_cells(
        root=args.root,
        problem_count=args.problem_count,
        tracks=_tracks(args.include_gpqa, args.tracks),
        num_shards=args.num_shards,
        selected_problem_ids_by_track=_selected_problem_ids_by_track(args),
    )


def _cmd_write_archives(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import write_archive

    for kind in (
        "direct",
        "reasoning_gym_direct",
        "reasoning_gym_seed_archive",
        "rws_math_direct",
        "seed_archive",
    ):
        write_archive(args.out_dir / f"{kind}.json", kind=kind)
    print(json.dumps({"out_dir": str(args.out_dir), "archives": 5}, sort_keys=True))


def _cmd_plan(args: argparse.Namespace) -> None:
    cells = _cells(args)
    payload = {"phase": args.phase, "cells": [cell.to_jsonable() for cell in cells]}
    if args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"phase": args.phase, "cells": len(cells), "out": str(args.out)}, sort_keys=True))


def _cmd_run_cell(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import cell_command

    cells = _cells(args)
    cell_index = args.cell_index
    if cell_index is None:
        cell_index = int(os.environ.get("POLARIS_ARRAY_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if cell_index < 0 or cell_index >= len(cells):
        raise SystemExit(f"cell index {cell_index} outside 0..{len(cells)-1}")
    cell = cells[cell_index]
    print(json.dumps({"selected_cell": cell.to_jsonable()}, sort_keys=True))
    repo_dir = os.environ.get("POLARIS_REPO_DIR", str(REPO_ROOT))
    subprocess.run(
        cell_command(
            cell,
            repo_dir=repo_dir,
            run_kind=args.run_kind,
            estimated_dollar_cost=args.estimated_dollar_cost,
            estimated_wall_clock_seconds=args.estimated_wall_clock_seconds,
            cost_cap_dollars=args.cost_cap_dollars,
            user_authorized_paid_run=args.user_authorized_paid_run,
        ),
        cwd=REPO_ROOT,
        check=True,
    )


def _cmd_aggregate_phase1(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import aggregate_phase1

    files = sorted(args.root.glob("phase1/**/candidates.jsonl"))
    if not files:
        raise SystemExit(f"no phase1 candidates found under {args.root}")
    summary = aggregate_phase1(files, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _cmd_aggregate_phase0(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import aggregate_phase0_gate

    files = sorted(args.root.glob("phase0/**/candidates.jsonl"))
    if not files:
        raise SystemExit(f"no phase0 candidates found under {args.root}")
    report = aggregate_phase0_gate(
        files,
        args.out,
        expected_problems=args.expected_problems,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def _cmd_plan_rws_exact(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import phase0_rws_exact_cells

    cells = phase0_rws_exact_cells(
        root=args.root,
        num_shards=args.num_shards,
        num_seeds=args.num_seeds,
    )
    payload = {
        "phase": "phase0",
        "kind": "rws_exact_upstream",
        "cells": [cell.to_jsonable() for cell in cells],
    }
    if args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"kind": "rws_exact_upstream", "cells": len(cells), "out": str(args.out)}, sort_keys=True))


def _cmd_run_rws_exact_cell(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import (
        phase0_rws_exact_cells,
        run_rws_exact_cell,
    )

    cells = phase0_rws_exact_cells(
        root=args.root,
        num_shards=args.num_shards,
        num_seeds=args.num_seeds,
    )
    cell_index = args.cell_index
    if cell_index is None:
        cell_index = int(os.environ.get("POLARIS_ARRAY_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if cell_index < 0 or cell_index >= len(cells):
        raise SystemExit(f"cell index {cell_index} outside 0..{len(cells)-1}")
    cell = cells[cell_index]
    print(json.dumps({"selected_cell": cell.to_jsonable()}, sort_keys=True))
    repo_dir = os.environ.get("POLARIS_REPO_DIR", str(REPO_ROOT))
    manifest = run_rws_exact_cell(cell, repo_dir=repo_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _cmd_aggregate_rws_exact(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import aggregate_rws_exact_phase0_gate

    report = aggregate_rws_exact_phase0_gate(
        args.root,
        args.out,
        expected_problems=args.expected_problems,
        expected_shards=args.expected_shards,
        expected_seeds=args.expected_seeds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def _cmd_audit_plan(args: argparse.Namespace) -> None:
    from polaris.prorl_recovery.orchestration import audit_recovery_cells

    report = audit_recovery_cells(_cells(args))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def main() -> None:
    args = _parse_args()
    dispatch = {
        "write-archives": _cmd_write_archives,
        "plan": _cmd_plan,
        "run-cell": _cmd_run_cell,
        "aggregate-phase1": _cmd_aggregate_phase1,
        "aggregate-phase0": _cmd_aggregate_phase0,
        "plan-rws-exact": _cmd_plan_rws_exact,
        "run-rws-exact-cell": _cmd_run_rws_exact_cell,
        "aggregate-rws-exact": _cmd_aggregate_rws_exact,
        "audit-plan": _cmd_audit_plan,
    }
    dispatch[args.action](args)


if __name__ == "__main__":
    main()
