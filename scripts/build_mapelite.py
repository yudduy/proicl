"""Build the frozen MAP-Elites prompt archive on external math dev rows.

Usage:
    python scripts/build_mapelite.py \
        --iterations 0 --dev-slice 0 500 --seed 17 \
        --out runs/2026-05-12-polaris-math500-v1/archive.json

`--iterations 0` evaluates each seeded prompt on the external optimizer-dev
split and emits the seed grid as the frozen archive. Do not tune archive
prompts on MATH500 rows used for final reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument(
        "--dev-slice",
        type=int,
        nargs=2,
        default=[0, 500],
        metavar=("START", "END"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-eval", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.core.descriptor import classify_trace
    from polaris.core.mapelite import run_mapelite
    from polaris.evals.datasets.math_optimizer_dev import load_math_optimizer_dev_slice
    from polaris.evals.verifiers.math import score_math
    from polaris.infra.serving.hf import RWSGenerator

    dev_set = load_math_optimizer_dev_slice(args.dev_slice[0], args.dev_slice[1])
    sampler = RWSGenerator(seed=args.seed)

    grid = run_mapelite(
        seeds=MATH500_ARCHIVE_V1.entries,
        dev_set=dev_set,
        sampler=sampler,
        scorer=score_math,
        descriptor_fn=classify_trace,
        n_iterations=args.iterations,
        samples_per_eval=args.samples_per_eval,
    )
    archive = grid.freeze()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": archive.to_jsonable(),
        "cell_fitness": grid.cell_fitness(),
        "dev_slice": list(args.dev_slice),
        "dev_source": "math_optimizer_dev",
        "seed": args.seed,
        "samples_per_eval": args.samples_per_eval,
        "iterations": args.iterations,
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {args.out}: {len(archive.entries)} cells filled")


if __name__ == "__main__":
    main()
