#!/usr/bin/env python
"""Estimate CloudRift run cost from explicit UI or catalog rates."""

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
    parser.add_argument("--gpu", required=True, choices=["rtx4090", "v100_sxm3", "mi350x"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seconds", type=float)
    group.add_argument("--hours", type=float)
    parser.add_argument("--hourly-rate", type=float, default=None)
    parser.add_argument("--on-demand", action="store_true")
    return parser.parse_args()


def main() -> None:
    from polaris.infra.cloudrift import (
        CLOUDRIFT_GPU_CATALOG,
        cloudrift_environment,
        estimate_cloudrift_cost,
        recommended_gpu_order,
    )

    args = _parse_args()
    seconds = args.seconds if args.seconds is not None else args.hours * 3600.0
    estimate = estimate_cloudrift_cost(
        args.gpu,
        wall_clock_seconds=seconds,
        hourly_rate=args.hourly_rate,
        use_spot=not args.on_demand,
    )
    payload = estimate.to_jsonable()
    payload["gpu"] = CLOUDRIFT_GPU_CATALOG[args.gpu].__dict__
    payload["environment"] = cloudrift_environment()
    payload["recommended_gpu_order"] = list(recommended_gpu_order())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
