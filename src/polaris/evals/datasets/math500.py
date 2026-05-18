from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Final MATH reporting follows the benchmark convention used by RWS/SPS:
# evaluate on the full MATH500 held-out set. Optimizer development must use
# `math_optimizer_dev.py`, not rows from this benchmark.
MATH500_TEST_SLICE: tuple[int, int] = (0, 500)
MATH500_DEV_SLICE: tuple[int, int] = (0, 0)  # deprecated; do not tune on MATH500
MATH500_SMALL_REAL_SLICE: tuple[int, int] = (0, 5)
MATH500_FINAL_SLICE: tuple[int, int] = (0, 500)

_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "vendored" / "rws" / "data" / "MATH500.json"
)


@dataclass(frozen=True)
class Problem:
    problem_id: str
    prompt: str
    answer: str
    source: str


def load_math500_all() -> list[Problem]:
    """Load all 500 MATH500 problems in file order."""
    with _DATA_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Problem(
            problem_id=row["id"],
            prompt=row["prompt"],
            answer=row["answer"],
            source=row.get("source", "math"),
        )
        for row in raw
    ]


def load_math500_slice(start: int, end: int) -> list[Problem]:
    return load_math500_all()[start:end]
