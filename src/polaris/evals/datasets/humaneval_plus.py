"""HumanEval+ loader (PROPOSAL §5.1 — Sustained-regime negative-control track).

Data source: vendored `evalplus` HumanEval+ JSON. Returns the same `Problem`
shape used by `math500.py` so condition_runner is benchmark-agnostic.

Final HumanEval+ reporting follows the benchmark convention used by code
evaluation papers: evaluate on all 164 HumanEval tasks with EvalPlus tests.
Optimizer development must use `code_optimizer_dev.py`, not HumanEval+ rows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

HUMANEVAL_PLUS_TEST_SLICE: tuple[int, int] = (0, 164)
HUMANEVAL_PLUS_DEV_SLICE: tuple[int, int] = (0, 0)  # deprecated; do not tune on HumanEval+
HUMANEVAL_PLUS_SMALL_REAL_SLICE: tuple[int, int] = (0, 5)
HUMANEVAL_PLUS_FINAL_SLICE: tuple[int, int] = (0, 164)


@dataclass(frozen=True)
class Problem:
    problem_id: str
    prompt: str
    answer: str  # for code track, this holds the canonical solution + tests
    source: str


def load_humaneval_plus_slice(start: int, end: int) -> list[Problem]:
    if start < 0 or end < start:
        raise ValueError(f"invalid HumanEval+ slice: {(start, end)!r}")

    override = os.environ.get("HUMANEVAL_OVERRIDE_PATH")
    if override:
        raw = [
            json.loads(line)
            for line in Path(override).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        problems = {row.get("task_id", str(i)): row for i, row in enumerate(raw)}
    else:
        from polaris.vendored.evalplus.data.humaneval import get_human_eval_plus

        problems = get_human_eval_plus()

    rows: list[Problem] = []
    for task_id in sorted(problems)[start:end]:
        payload = dict(problems[task_id])
        payload.setdefault("task_id", task_id)
        rows.append(
            Problem(
                problem_id=task_id,
                prompt=payload.get("prompt", ""),
                answer=json.dumps(payload, sort_keys=True),
                source="humaneval_plus",
            )
        )
    return rows
