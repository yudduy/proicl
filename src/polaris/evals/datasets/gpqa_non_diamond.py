"""GPQA optimizer-development pool excluding GPQA-Diamond rows.

GPQA-Diamond is final/offline evaluation only. If science QA prompt or memory
development is needed, use non-Diamond GPQA rows and keep Diamond locked away
from optimization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from polaris.evals.datasets.gpqa_diamond import Problem, _problem_from_row, _read_local_rows

GPQA_NON_DIAMOND_DEV_SLICE: tuple[int, int] = (0, 250)


def load_gpqa_non_diamond_slice(start: int, end: int) -> list[Problem]:
    if start < 0 or end < start:
        raise ValueError(f"invalid GPQA non-Diamond slice: {(start, end)!r}")

    path = os.environ.get("GPQA_NON_DIAMOND_PATH")
    if path:
        rows = _read_local_rows(Path(path))
    else:
        try:
            from datasets import load_dataset

            main_rows = list(load_dataset("Idavidrein/gpqa", "gpqa_main", split="train"))
            diamond_rows = list(
                load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            )
        except Exception as exc:  # pragma: no cover - depends on auth/cache
            raise RuntimeError(
                "GPQA_NON_DIAMOND_PATH is unset and Hugging Face GPQA non-Diamond "
                "load failed"
            ) from exc
        diamond_ids = {_row_id(row, idx) for idx, row in enumerate(diamond_rows)}
        rows = [
            row
            for idx, row in enumerate(main_rows)
            if _row_id(row, idx) not in diamond_ids
        ]

    return [
        _problem_from_row(i, row, source="gpqa_non_diamond_optimizer_dev")
        for i, row in enumerate(rows[start:end], start)
    ]


def _row_id(row: dict, idx: int) -> str:
    return str(
        row.get("Record ID")
        or row.get("problem_id")
        or row.get("id")
        or json.dumps(row, sort_keys=True)
        or idx
    )
