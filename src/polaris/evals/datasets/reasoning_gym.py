from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Problem:
    problem_id: str
    prompt: str
    answer: str
    source: str


def _fixture_path(task: str) -> Path | None:
    task_env = f"REASONING_GYM_{task.upper()}_PATH"
    raw = os.environ.get(task_env) or os.environ.get("REASONING_GYM_FIXTURE_PATH")
    return Path(raw) if raw else None


def _load_fixture(path: Path, task: str) -> list[Problem]:
    rows: list[Problem] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("task", task) != task:
                continue
            prompt = str(raw.get("prompt", raw.get("question", "")))
            rows.append(
                Problem(
                    problem_id=str(raw.get("problem_id", f"{task}-{idx}")),
                    prompt=prompt,
                    answer=json.dumps(
                        {
                            "answer": raw.get("answer", ""),
                            "entry": raw.get("entry")
                            or {
                                "question": prompt,
                                "answer": raw.get("answer", ""),
                                "metadata": raw.get("metadata", {}),
                            },
                            "task": task,
                        },
                        sort_keys=True,
                    ),
                    source=str(raw.get("source", "fixture")),
                )
            )
    return rows


def _entry_to_problem(task: str, idx: int, entry: dict[str, Any]) -> Problem:
    return Problem(
        problem_id=f"{task}-{idx}",
        prompt=str(entry.get("question", "")),
        answer=json.dumps(
            {"answer": entry.get("answer", ""), "entry": entry, "task": task},
            sort_keys=True,
        ),
        source="reasoning_gym",
    )


def load_reasoning_gym_slice(task: str, start: int, end: int) -> list[Problem]:
    if start < 0 or end < start:
        raise ValueError("slice must satisfy 0 <= start <= end")
    fixture = _fixture_path(task)
    if fixture is not None:
        return _load_fixture(fixture, task)[start:end]

    try:
        import reasoning_gym
    except ImportError as exc:
        raise RuntimeError(
            "reasoning-gym is not installed; set REASONING_GYM_FIXTURE_PATH for CI"
        ) from exc

    dataset = reasoning_gym.create_dataset(task, size=end, seed=0)
    return [_entry_to_problem(task, idx, dataset[idx]) for idx in range(start, end)]
