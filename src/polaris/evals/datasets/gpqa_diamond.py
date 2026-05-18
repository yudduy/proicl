"""GPQA-Diamond loader (PROPOSAL §5.1 — offline-only track).

Answer keys are offline evaluators; inference-time selection uses a
non-oracle selector (`select_gpqa_non_oracle` in evals/verifiers/gpqa.py)
per PROPOSAL §4.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

GPQA_DIAMOND_TEST_SLICE: tuple[int, int] = (0, 198)
GPQA_DIAMOND_DEV_SLICE: tuple[int, int] = (0, 0)  # deprecated; do not tune on Diamond
GPQA_DIAMOND_SMALL_REAL_SLICE: tuple[int, int] = (0, 5)
GPQA_DIAMOND_FINAL_SLICE: tuple[int, int] = (0, 198)


@dataclass(frozen=True)
class Problem:
    problem_id: str
    prompt: str
    answer: str  # multiple-choice letter; oracle-only
    source: str


def load_gpqa_diamond_slice(start: int, end: int) -> list[Problem]:
    if start < 0 or end < start:
        raise ValueError(f"invalid GPQA-Diamond slice: {(start, end)!r}")

    path = os.environ.get("GPQA_DIAMOND_PATH")
    if path:
        rows = _read_local_rows(Path(path))
    else:
        try:
            from datasets import load_dataset

            rows = list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
        except Exception as exc:  # pragma: no cover - depends on network/cache
            raise RuntimeError(
                "GPQA_DIAMOND_PATH is unset and Hugging Face GPQA load failed"
            ) from exc

    return [
        _problem_from_row(i, row, source="gpqa_diamond")
        for i, row in enumerate(rows[start:end], start)
    ]


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
    raise ValueError(f"unsupported GPQA local dataset format: {path}")


def _problem_from_row(idx: int, row: dict, *, source: str = "gpqa_diamond") -> Problem:
    if "prompt" in row and "answer" in row:
        return Problem(
            problem_id=str(row.get("problem_id", f"gpqa-{idx}")),
            prompt=str(row["prompt"]),
            answer=str(row["answer"]).strip(),
            source=source,
        )
    if "Question" in row and "Correct Answer" in row:
        choices = [
            ("A", row["Correct Answer"]),
            ("B", row.get("Incorrect Answer 1", "")),
            ("C", row.get("Incorrect Answer 2", "")),
            ("D", row.get("Incorrect Answer 3", "")),
        ]
        prompt = str(row["Question"]) + "\n" + "\n".join(
            f"{letter}. {answer}" for letter, answer in choices
        )
        return Problem(
            problem_id=str(row.get("Record ID", row.get("id", f"gpqa-{idx}"))),
            prompt=prompt,
            answer="A",
            source=source,
        )
    if "input" in row and "target" in row:
        return Problem(
            problem_id=str(row.get("id", f"gpqa-{idx}")),
            prompt=str(row["input"]),
            answer=str(row["target"]).strip(),
            source=source,
        )
    raise ValueError(f"cannot parse GPQA row keys: {sorted(row)}")
