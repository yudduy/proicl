"""Validate a saved SPS-vs-MCMC calibration summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = args.path
    if path.is_dir():
        path = path / "sps_calibration_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    diff = float(payload.get("accuracy_abs_diff", 1.0))
    if payload.get("passed") is not True or diff > args.tolerance:
        raise SystemExit(
            f"SPS calibration failed at {path}: passed={payload.get('passed')} "
            f"accuracy_abs_diff={diff} tolerance={args.tolerance}"
        )
    print(
        json.dumps(
            {
                "passed": True,
                "path": str(path),
                "accuracy_abs_diff": diff,
                "tolerance": args.tolerance,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
