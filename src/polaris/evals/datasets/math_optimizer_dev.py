"""External math optimizer-development pool.

This loader is intentionally separate from MATH500. Sampling-only papers report
full MATH500 directly; POLARIS learns prompts and memory, so development rows
must come from a non-MATH500 pool.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from polaris.evals.datasets.math500 import Problem
from polaris.vendored.rws.grader_utils.parse_utils import parse_answer

MATH_OPTIMIZER_DEV_SLICE: tuple[int, int] = (0, 500)


def load_math_optimizer_dev_slice(start: int, end: int) -> list[Problem]:
    if start < 0 or end < start:
        raise ValueError(f"invalid math optimizer-dev slice: {(start, end)!r}")

    path = os.environ.get("MATH_OPTIMIZER_DEV_PATH")
    if path:
        rows = _read_local_rows(Path(path))
    else:
        try:
            from datasets import load_dataset

            rows = list(
                load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
            )
        except Exception as exc:  # pragma: no cover - depends on network/cache
            raise RuntimeError(
                "MATH_OPTIMIZER_DEV_PATH is unset and MATH train-pool load failed"
            ) from exc

    return [_problem_from_row(i, row) for i, row in enumerate(rows[start:end], start)]


def _read_local_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload if isinstance(payload, list) else payload["rows"])
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported math optimizer-dev format: {path}")


def _problem_from_row(idx: int, row: dict) -> Problem:
    prompt = str(row.get("problem") or row.get("prompt") or row.get("question") or "")
    solution = str(row.get("solution") or row.get("answer") or "")
    answer = str(row.get("answer") or (parse_answer(solution) or solution)).strip()
    return Problem(
        problem_id=str(row.get("problem_id") or row.get("id") or f"math-train-{idx}"),
        prompt=prompt,
        answer=answer,
        source="math_optimizer_dev",
    )
