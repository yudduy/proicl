"""Materialize official dataset lock metadata without committing raw rows.

Raw rows are cached under ignored `runs/dataset_cache.tmp/`. The tracked
`data/locks/datasets.lock.json` contains only source metadata, row counts, and
SHA-256 hashes of row ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["math500", "humaneval_plus", "gpqa_diamond"],
        choices=["math500", "humaneval_plus", "gpqa_diamond"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "small_real_slice", "final"],
        choices=["dev", "small_real_slice", "final"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "locks" / "datasets.lock.json",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Write pending_auth/pending_cache locks instead of failing on unavailable gated data.",
    )
    return parser.parse_args()


def _pending_lock(
    *,
    track: str,
    split_id: str,
    split: str,
    notes: str,
    source_repo: str | None = None,
    config: str | None = None,
    loader_version: str | None = None,
    cache_path: str | None = None,
    status: str | None = None,
):
    from polaris.io.dataset_locks import build_dataset_lock

    source_repo = source_repo or {
        "math500": "HuggingFaceH4/MATH-500",
        "humaneval_plus": "evalplus/evalplus",
        "gpqa_diamond": "Idavidrein/gpqa",
    }[track]
    config = config or {
        "math500": "math500",
        "humaneval_plus": "humaneval_plus",
        "gpqa_diamond": "gpqa_diamond",
    }[track]
    loader_version = loader_version or {
        "math500": "polaris.evals.datasets.math500:v1",
        "humaneval_plus": "polaris.vendored.evalplus:data.humaneval",
        "gpqa_diamond": "polaris.evals.datasets.gpqa_diamond:v1",
    }[track]
    status = status or ("pending_auth" if track == "gpqa_diamond" else "pending_cache")
    return build_dataset_lock(
        track=track,
        source_repo=source_repo,
        config=config,
        split=split,
        split_id=split_id,
        rows=[],
        row_id=lambda row, idx: str(idx),
        loader_version=loader_version,
        cache_path=cache_path or f"runs/dataset_cache.tmp/{track}/{split_id}",
        status=status,
        notes=notes,
    )


def _split_for_track(track: str, split_id: str) -> tuple[str, tuple[int, int]]:
    if track == "math500":
        from polaris.evals.datasets.math500 import (
            MATH500_DEV_SLICE,
            MATH500_FINAL_SLICE,
            MATH500_SMALL_REAL_SLICE,
        )

        return {
            "dev": ("test", MATH500_DEV_SLICE),
            "small_real_slice": ("test", MATH500_SMALL_REAL_SLICE),
            "final": ("test", MATH500_FINAL_SLICE),
        }[split_id]
    if track == "humaneval_plus":
        from polaris.evals.datasets.humaneval_plus import (
            HUMANEVAL_PLUS_DEV_SLICE,
            HUMANEVAL_PLUS_FINAL_SLICE,
            HUMANEVAL_PLUS_SMALL_REAL_SLICE,
        )

        return {
            "dev": ("test", HUMANEVAL_PLUS_DEV_SLICE),
            "small_real_slice": ("test", HUMANEVAL_PLUS_SMALL_REAL_SLICE),
            "final": ("test", HUMANEVAL_PLUS_FINAL_SLICE),
        }[split_id]
    if track == "gpqa_diamond":
        from polaris.evals.datasets.gpqa_diamond import (
            GPQA_DIAMOND_DEV_SLICE,
            GPQA_DIAMOND_FINAL_SLICE,
            GPQA_DIAMOND_SMALL_REAL_SLICE,
        )

        return {
            "dev": ("train", GPQA_DIAMOND_DEV_SLICE),
            "small_real_slice": ("train", GPQA_DIAMOND_SMALL_REAL_SLICE),
            "final": ("train", GPQA_DIAMOND_FINAL_SLICE),
        }[split_id]
    raise ValueError(f"unknown track: {track!r}")


def _lock_for_track(track: str, split_id: str, *, allow_pending: bool):
    from polaris.io.dataset_locks import build_dataset_lock

    split_name, bounds = _split_for_track(track, split_id)
    split_text = f"{split_name}[{bounds[0]}:{bounds[1]}]"
    if track == "math500" and split_id == "dev":
        from polaris.evals.datasets.math_optimizer_dev import (
            MATH_OPTIMIZER_DEV_SLICE,
            load_math_optimizer_dev_slice,
        )

        split_text = f"train[{MATH_OPTIMIZER_DEV_SLICE[0]}:{MATH_OPTIMIZER_DEV_SLICE[1]}]"
        try:
            rows = load_math_optimizer_dev_slice(*MATH_OPTIMIZER_DEV_SLICE)
        except Exception as exc:
            if not allow_pending:
                raise
            return _pending_lock(
                track=track,
                split_id=split_id,
                split=split_text,
                source_repo="DigitalLearningGmbH/MATH-lighteval",
                config="math_optimizer_dev",
                loader_version="polaris.evals.datasets.math_optimizer_dev:v1",
                cache_path="runs/dataset_cache.tmp/math_optimizer_dev/dev",
                status="pending_cache",
                notes=f"MATH optimizer-dev pool could not be cached: {exc}",
            )
        return build_dataset_lock(
            track=track,
            source_repo="DigitalLearningGmbH/MATH-lighteval",
            config="math_optimizer_dev",
            split=split_text,
            split_id=split_id,
            rows=rows,
            row_id=lambda row, idx: row.problem_id,
            loader_version="polaris.evals.datasets.math_optimizer_dev:v1",
            cache_path="runs/dataset_cache.tmp/math_optimizer_dev/dev",
        )
    if track == "math500":
        from polaris.evals.datasets.math500 import load_math500_slice

        rows = load_math500_slice(*bounds)
        return build_dataset_lock(
            track=track,
            source_repo="HuggingFaceH4/MATH-500",
            config="math500",
            split=split_text,
            split_id=split_id,
            rows=rows,
            row_id=lambda row, idx: row.problem_id,
            loader_version="polaris.evals.datasets.math500:v1",
            cache_path=f"runs/dataset_cache.tmp/math500/{split_id}",
        )
    if track == "humaneval_plus" and split_id == "dev":
        from polaris.evals.datasets.code_optimizer_dev import (
            CODE_OPTIMIZER_DEV_SLICE,
            load_code_optimizer_dev_slice,
        )

        split_text = (
            f"mbpp_plus[{CODE_OPTIMIZER_DEV_SLICE[0]}:{CODE_OPTIMIZER_DEV_SLICE[1]}]"
        )
        try:
            rows = load_code_optimizer_dev_slice(*CODE_OPTIMIZER_DEV_SLICE)
        except Exception as exc:
            if not allow_pending:
                raise
            return _pending_lock(
                track=track,
                split_id=split_id,
                split=split_text,
                source_repo="evalplus/evalplus",
                config="mbpp_plus_optimizer_dev",
                loader_version="polaris.evals.datasets.code_optimizer_dev:v1",
                cache_path="runs/dataset_cache.tmp/mbpp_plus_optimizer_dev/dev",
                status="pending_cache",
                notes=f"MBPP+ optimizer-dev pool could not be cached: {exc}",
            )
        return build_dataset_lock(
            track=track,
            source_repo="evalplus/evalplus",
            config="mbpp_plus_optimizer_dev",
            split=split_text,
            split_id=split_id,
            rows=rows,
            row_id=lambda row, idx: row.problem_id,
            loader_version="polaris.evals.datasets.code_optimizer_dev:v1",
            cache_path="runs/dataset_cache.tmp/mbpp_plus_optimizer_dev/dev",
        )
    if track == "humaneval_plus":
        from polaris.evals.datasets.humaneval_plus import load_humaneval_plus_slice

        rows = load_humaneval_plus_slice(*bounds)
        return build_dataset_lock(
            track=track,
            source_repo="evalplus/evalplus",
            config="humaneval_plus",
            split=split_text,
            split_id=split_id,
            rows=rows,
            row_id=lambda row, idx: row.problem_id,
            loader_version="polaris.vendored.evalplus:data.humaneval",
            cache_path=f"runs/dataset_cache.tmp/humaneval_plus/{split_id}",
        )
    if track == "gpqa_diamond" and split_id == "dev":
        from polaris.evals.datasets.gpqa_non_diamond import (
            GPQA_NON_DIAMOND_DEV_SLICE,
            load_gpqa_non_diamond_slice,
        )

        split_text = (
            f"gpqa_main_minus_diamond[{GPQA_NON_DIAMOND_DEV_SLICE[0]}:"
            f"{GPQA_NON_DIAMOND_DEV_SLICE[1]}]"
        )
        try:
            rows = load_gpqa_non_diamond_slice(*GPQA_NON_DIAMOND_DEV_SLICE)
        except Exception as exc:
            if not allow_pending:
                raise
            return _pending_lock(
                track=track,
                split_id=split_id,
                split=split_text,
                source_repo="Idavidrein/gpqa",
                config="gpqa_main_minus_diamond",
                loader_version="polaris.evals.datasets.gpqa_non_diamond:v1",
                cache_path="runs/dataset_cache.tmp/gpqa_non_diamond/dev",
                status="pending_auth",
                notes=(
                    "GPQA non-Diamond optimizer-dev pool is gated; set "
                    f"GPQA_NON_DIAMOND_PATH or accepted Hugging Face auth. Loader error: {exc}"
                ),
            )
        return build_dataset_lock(
            track=track,
            source_repo="Idavidrein/gpqa",
            config="gpqa_main_minus_diamond",
            split=split_text,
            split_id=split_id,
            rows=rows,
            row_id=lambda row, idx: row.problem_id,
            loader_version="polaris.evals.datasets.gpqa_non_diamond:v1",
            cache_path="runs/dataset_cache.tmp/gpqa_non_diamond/dev",
        )
    if track == "gpqa_diamond":
        from polaris.evals.datasets.gpqa_diamond import load_gpqa_diamond_slice

        has_explicit_access = bool(
            os.environ.get("GPQA_DIAMOND_PATH")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )
        if not has_explicit_access:
            if not allow_pending:
                raise RuntimeError(
                    "GPQA is gated; set GPQA_DIAMOND_PATH or accepted Hugging Face auth"
                )
            return _pending_lock(
                track=track,
                split_id=split_id,
                split=split_text,
                notes=(
                    "GPQA is gated; set GPQA_DIAMOND_PATH or accepted Hugging Face "
                    "auth before final runs."
                ),
            )
        try:
            rows = load_gpqa_diamond_slice(*bounds)
        except Exception as exc:
            if not allow_pending:
                raise
            return _pending_lock(
                track=track,
                split_id=split_id,
                split=split_text,
                notes=(
                    "GPQA is gated; set GPQA_DIAMOND_PATH or accepted Hugging Face "
                    f"auth before final runs. Loader error: {exc}"
                ),
            )
        return build_dataset_lock(
            track=track,
            source_repo="Idavidrein/gpqa",
            config="gpqa_diamond",
            split=split_text,
            split_id=split_id,
            rows=rows,
            row_id=lambda row, idx: row.problem_id,
            loader_version="polaris.evals.datasets.gpqa_diamond:v1",
            cache_path=f"runs/dataset_cache.tmp/gpqa_diamond/{split_id}",
        )
    raise ValueError(f"unknown track: {track!r}")


def main() -> None:
    args = _parse_args()
    from polaris.io.dataset_locks import write_dataset_locks

    locks = [
        _lock_for_track(track, split_id, allow_pending=args.allow_pending)
        for track in args.tracks
        for split_id in args.splits
    ]
    payload = write_dataset_locks(args.out, locks)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
