"""Write a production run graph for all registered POLARIS tracks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/production/plan.json"))
    parser.add_argument(
        "--stage",
        action="append",
        choices=["smoke", "small_real_slice", "final"],
        help="Repeatable. Defaults to small_real_slice and final.",
    )
    parser.add_argument(
        "--track",
        action="append",
        choices=["math500", "humaneval_plus", "gpqa_diamond"],
        help="Repeatable. Defaults to all tracks.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from polaris.production_plan import build_production_plan, write_production_plan

    plan = build_production_plan(
        stages=tuple(args.stage or ["small_real_slice", "final"]),
        tracks=tuple(args.track or ["math500", "humaneval_plus", "gpqa_diamond"]),
        seed=args.seed,
    )
    write_production_plan(args.out, plan)
    print(json.dumps({"path": str(args.out), "cells": len(plan["cells"])}, indent=2))


if __name__ == "__main__":
    main()
