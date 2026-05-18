"""External code optimizer-development pool.

HumanEval+ remains the held-out final benchmark. POLARIS prompt/archive/memory
development uses MBPP+ style tasks instead of HumanEval+ rows.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from polaris.evals.datasets.humaneval_plus import Problem

CODE_OPTIMIZER_DEV_SLICE: tuple[int, int] = (0, 378)


def load_code_optimizer_dev_slice(start: int, end: int) -> list[Problem]:
    if start < 0 or end < start:
        raise ValueError(f"invalid code optimizer-dev slice: {(start, end)!r}")

    path = os.environ.get("CODE_OPTIMIZER_DEV_PATH")
    if path:
        raw = _read_local_rows(Path(path))
        rows = {str(row.get("task_id", f"code-dev-{i}")): row for i, row in enumerate(raw)}
    else:
        from polaris.vendored.evalplus.data.mbpp import get_mbpp_plus

        rows = get_mbpp_plus()

    selected = [dict(rows[task_id]) for task_id in sorted(rows)[start:end]]
    problems: list[Problem] = []
    for idx, payload in enumerate(selected, start):
        task_id = str(payload.get("task_id", f"code-dev-{idx}"))
        payload.setdefault("task_id", task_id)
        problems.append(
            Problem(
                problem_id=task_id,
                prompt=str(payload.get("prompt", "")),
                answer=json.dumps(payload, sort_keys=True, default=str),
                source="mbpp_plus_optimizer_dev",
            )
        )
    return problems


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
    raise ValueError(f"unsupported code optimizer-dev format: {path}")
