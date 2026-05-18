from __future__ import annotations

import json

from polaris.evals.datasets.code_optimizer_dev import load_code_optimizer_dev_slice
from polaris.evals.datasets.math_optimizer_dev import load_math_optimizer_dev_slice


def test_math_optimizer_dev_uses_external_pool(monkeypatch, tmp_path):
    path = tmp_path / "math_dev.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "train-row",
                "problem": "Compute 7+8.",
                "solution": "We get \\boxed{15}.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MATH_OPTIMIZER_DEV_PATH", str(path))

    rows = load_math_optimizer_dev_slice(0, 5)

    assert len(rows) == 1
    assert rows[0].problem_id == "train-row"
    assert rows[0].answer == "15"
    assert rows[0].source == "math_optimizer_dev"


def test_code_optimizer_dev_uses_mbpp_style_external_pool(monkeypatch, tmp_path):
    path = tmp_path / "code_dev.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "Mbpp/1",
                "prompt": "Write a function.",
                "canonical_solution": "def f(): return 1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_OPTIMIZER_DEV_PATH", str(path))

    rows = load_code_optimizer_dev_slice(0, 5)

    assert len(rows) == 1
    assert rows[0].problem_id == "Mbpp/1"
    assert rows[0].source == "mbpp_plus_optimizer_dev"
