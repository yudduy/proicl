"""Render or execute resource-profiled ProRL recovery cells.

This wrapper does not choose science settings. It binds an existing ProRL cell
plan to a resource profile, enforces profile cost caps before execution, and
records the backend-agnostic artifact contract expected from every run.
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
    parser.add_argument("--profile-config", type=Path, default=None)
    sub = parser.add_subparsers(dest="action", required=True)

    profiles = sub.add_parser("profiles")
    profiles.add_argument("--out", type=Path, default=None)

    plan = sub.add_parser("plan")
    plan.add_argument("--profile", required=True)
    plan.add_argument("--phase", choices=["phase0", "phase1", "phase2"], required=True)
    plan.add_argument("--root", default="/workspace/polaris/runs/prorl_recovery")
    plan.add_argument("--problem-count", type=int, default=100)
    plan.add_argument("--num-shards", type=int, default=4)
    plan.add_argument("--samples-per-problem", type=int, default=128)
    plan.add_argument("--tracks", nargs="+", default=None)
    plan.add_argument("--include-gpqa", action="store_true")
    plan.add_argument("--phase1-results", type=Path, default=None)
    plan.add_argument("--out", type=Path, required=True)

    run_cell = sub.add_parser("run-cell")
    run_cell.add_argument("--profile", required=True)
    run_cell.add_argument("--phase", choices=["phase0", "phase1", "phase2"], required=True)
    run_cell.add_argument("--cell-index", type=int, default=None)
    run_cell.add_argument("--root", default="/workspace/polaris/runs/prorl_recovery")
    run_cell.add_argument("--problem-count", type=int, default=100)
    run_cell.add_argument("--num-shards", type=int, default=4)
    run_cell.add_argument("--samples-per-problem", type=int, default=128)
    run_cell.add_argument("--tracks", nargs="+", default=None)
    run_cell.add_argument("--include-gpqa", action="store_true")
    run_cell.add_argument("--phase1-results", type=Path, default=None)
    run_cell.add_argument("--estimated-dollar-cost", type=float, required=True)
    run_cell.add_argument("--cost-cap-dollars", type=float, required=True)
    run_cell.add_argument("--estimated-wall-clock-seconds", type=float, default=None)
    run_cell.add_argument("--user-authorized-paid-run", action="store_true")
    run_cell.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _profile_path(args: argparse.Namespace) -> Path | None:
    return args.profile_config


def _load_profiles(args: argparse.Namespace):
    from polaris.infra.resources import load_resource_profiles

    path = _profile_path(args)
    return load_resource_profiles() if path is None else load_resource_profiles(path)


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
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".parquet":
        import pandas as pd

        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload


def _cells(args: argparse.Namespace):
    from polaris.prorl_recovery.orchestration import (
        derive_prorl_only_problem_ids,
        phase0_cells,
        phase1_cells,
        phase2_cells,
    )

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
    if args.phase1_results is None:
        raise SystemExit("phase2 planning requires --phase1-results")
    selected = derive_prorl_only_problem_ids(_read_phase1_rows(args.phase1_results))
    return phase2_cells(
        root=args.root,
        problem_count=args.problem_count,
        tracks=_tracks(args.include_gpqa, args.tracks),
        num_shards=args.num_shards,
        selected_problem_ids_by_track=selected,
    )


def _bundle(args: argparse.Namespace, profile) -> dict:
    from polaris.infra.resources import artifact_contract

    cells = _cells(args)
    return {
        "resource_profile": profile.to_jsonable(),
        "phase": args.phase,
        "cells": [cell.to_jsonable() for cell in cells],
        "artifact_contract": list(artifact_contract()),
        "launch_policy": {
            "tensor_parallelism": "disabled",
            "worker_shape": f"{profile.gpu_count} independent one-GPU worker(s)",
            "run_kind": profile.run_kind,
            "requires_fresh_user_go": True,
            "cost_cap_source": "resource_profile.initial_spend_cap_dollars plus per-launch cap",
        },
    }


def _cmd_profiles(args: argparse.Namespace) -> None:
    profiles = _load_profiles(args)
    payload = {"profiles": [profile.to_jsonable() for profile in profiles.values()]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_plan(args: argparse.Namespace) -> None:
    profiles = _load_profiles(args)
    profile = profiles[args.profile]
    payload = _bundle(args, profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"profile": profile.key, "phase": args.phase, "cells": len(payload["cells"]), "out": str(args.out)}, sort_keys=True))


def _cmd_run_cell(args: argparse.Namespace) -> None:
    from polaris.infra.resources import validate_profile_cost
    from polaris.prorl_recovery.orchestration import cell_command

    profiles = _load_profiles(args)
    profile = profiles[args.profile]
    validate_profile_cost(profile, estimated_dollar_cost=args.estimated_dollar_cost)
    if args.cost_cap_dollars > profile.initial_spend_cap_dollars:
        raise SystemExit(
            "per-launch cost cap exceeds profile initial_spend_cap_dollars "
            f"({args.cost_cap_dollars} > {profile.initial_spend_cap_dollars})"
        )
    cells = _cells(args)
    cell_index = args.cell_index
    if cell_index is None:
        cell_index = int(os.environ.get("POLARIS_ARRAY_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if cell_index < 0 or cell_index >= len(cells):
        raise SystemExit(f"cell index {cell_index} outside 0..{len(cells)-1}")
    cmd = cell_command(
        cells[cell_index],
        repo_dir=os.environ.get("POLARIS_REPO_DIR", str(REPO_ROOT)),
        run_kind=profile.run_kind,
        estimated_dollar_cost=args.estimated_dollar_cost,
        estimated_wall_clock_seconds=args.estimated_wall_clock_seconds,
        cost_cap_dollars=args.cost_cap_dollars,
        user_authorized_paid_run=args.user_authorized_paid_run,
    )
    if not args.execute:
        print(json.dumps({"profile": profile.to_jsonable(), "command": cmd}, indent=2, sort_keys=True))
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = _parse_args()
    dispatch = {
        "profiles": _cmd_profiles,
        "plan": _cmd_plan,
        "run-cell": _cmd_run_cell,
    }
    dispatch[args.action](args)


if __name__ == "__main__":
    main()
