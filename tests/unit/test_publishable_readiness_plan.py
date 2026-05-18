from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.evals.datasets.math500 import Problem
from polaris.io.artifact_audit import ArtifactAuditError, audit_run_artifacts
from polaris.io.dataset_locks import write_dataset_locks
from polaris.production_plan import build_production_plan
from polaris.registry import CONDITION_REGISTRY
from polaris.runners.condition_runner import run_condition as run_track_condition


class _Gen:
    response_contains_prompt = False
    prompt_token_count = 4
    generation_token_count = 8
    wall_clock_seconds = 0.0
    estimated_dollar_cost = 0.0
    acceptance_ratio = 0.5
    token_ids = [1, 2]
    logprobs_norm = [-1.0, -1.0]
    logprobs_unnorm = [-4.0, -4.0]

    def __init__(self, generation: str = "\nUse the tactic and return \\boxed{42}.") -> None:
        self.generation = generation


class _MemorySampler:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_greedy(self, prompt_text: str, max_new_tokens: int):
        self.prompts.append(prompt_text)
        return _Gen()

    def generate_low_temp(self, prompt_text: str, *, temperature: float, max_new_tokens: int):
        self.prompts.append(prompt_text)
        return _Gen()

    def generate_power(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int,
        mcmc_steps: int | None = None,
        block_num: int | None = None,
    ):
        self.prompts.append(prompt_text)
        return _Gen()


def _write_lock(path: Path, *, split: str = "test[0:1]") -> Path:
    from polaris.io.dataset_locks import build_dataset_lock

    lock = build_dataset_lock(
        track="math500",
        source_repo="fixture",
        config="math500",
        split=split,
        rows=[Problem(problem_id="p1", prompt="q", answer="42", source="math")],
        row_id=lambda row, idx: row.problem_id,
        loader_version="test",
    )
    write_dataset_locks(path, [lock])
    return path


def test_full_polaris_condition_registry_flags():
    spec = CONDITION_REGISTRY["polaris_full_verified_memory"]
    assert spec.uses_archive is True
    assert spec.uses_power_sampling is True
    assert spec.uses_memory is True
    assert spec.uses_gepa is True
    assert spec.production_baseline is False


def test_production_plan_contains_publishable_metadata():
    plan = build_production_plan(
        stages=("final",),
        tracks=("math500",),
        conditions=("polaris_full_verified_memory",),
    )
    cell = plan["cells"][0]
    assert cell["split_id"] == "final"
    assert cell["archive_build_id"]
    assert cell["memory_build_id"]
    assert cell["dataset_lock_id"]
    assert "model_revision" in cell
    assert cell["memory_mode"] == "distilled_strategies"


def test_lock_datasets_writes_split_level_locks(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "locks.json"
    math_dev = tmp_path / "math_dev.jsonl"
    math_dev.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "math-train-1",
                        "problem": "What is 2+2?",
                        "solution": "The answer is \\boxed{4}.",
                    }
                ),
                json.dumps(
                    {
                        "id": "math-train-2",
                        "problem": "What is 3+3?",
                        "solution": "The answer is \\boxed{6}.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    code_dev = tmp_path / "code_dev.jsonl"
    code_dev.write_text(
        json.dumps(
            {
                "task_id": "Mbpp/fixture",
                "prompt": "Write add_one.",
                "canonical_solution": "def add_one(x): return x + 1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["MATH_OPTIMIZER_DEV_PATH"] = str(math_dev)
    env["CODE_OPTIMIZER_DEV_PATH"] = str(code_dev)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/lock_datasets.py",
            "--tracks",
            "math500",
            "humaneval_plus",
            "--splits",
            "dev",
            "small_real_slice",
            "final",
            "--out",
            str(out),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    splits = {(lock["track"], lock["split_id"]): lock for lock in payload["locks"]}
    assert splits[("math500", "dev")]["source_repo"] == "DigitalLearningGmbH/MATH-lighteval"
    assert splits[("math500", "dev")]["config"] == "math_optimizer_dev"
    assert splits[("math500", "dev")]["split"] == "train[0:500]"
    assert splits[("math500", "dev")]["row_count"] == 2
    assert splits[("math500", "final")]["split"] == "test[0:500]"
    assert splits[("humaneval_plus", "dev")]["config"] == "mbpp_plus_optimizer_dev"
    assert splits[("humaneval_plus", "dev")]["split"] == "mbpp_plus[0:378]"
    assert splits[("humaneval_plus", "dev")]["row_count"] == 1
    assert splits[("humaneval_plus", "final")]["split"] == "test[0:164]"


def test_lock_datasets_records_gpqa_auth_blocker(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "locks.json"
    env = dict(os.environ)
    env.pop("GPQA_DIAMOND_PATH", None)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGINGFACE_HUB_TOKEN", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/lock_datasets.py",
            "--tracks",
            "gpqa_diamond",
            "--splits",
            "final",
            "--out",
            str(out),
            "--allow-pending",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lock = json.loads(out.read_text(encoding="utf-8"))["locks"][0]
    assert lock["status"] == "pending_auth"
    assert "gated" in lock["notes"].lower() or "auth" in lock["notes"].lower()


def test_full_polaris_condition_writes_memory_artifacts(tmp_path):
    sampler = _MemorySampler()
    out = tmp_path / "run"
    metrics = run_track_condition(
        track="math500",
        split=(0, 1),
        model_key="qwen2.5-math-7b",
        out_dir=out,
        condition="polaris_full_verified_memory",
        archive=MATH500_ARCHIVE_V1,
        cell_fitness={},
        sampler=sampler,
        problems=[Problem(problem_id="p1", prompt="What is 6*7?", answer="42", source="math")],
        seed=17,
        archive_hash="hash",
        polaris_source_hash="dev",
        vendored_commits={},
        preregistration_anchor="TODO.md#test",
        run_stage="small_real_slice",
        preflight_report={"passed": True, "protocol_sync_passed": True},
        memory_store_path=tmp_path / "memory.sqlite",
        memory_mode="distilled_strategies",
        admit_memory=True,
        archive_build_id="archive-build-test",
    )
    assert metrics["condition"] == "polaris_full_verified_memory"
    assert (out / "memory.sqlite").exists()
    assert (out / "memory_events.jsonl").exists()
    assert (out / "archive_build_manifest.json").exists()
    rows = [
        json.loads(line)
        for line in (out / "memory_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["event_type"] for row in rows} >= {"admission"}


def test_final_audit_rejects_memory_condition_without_ledger(tmp_path):
    out = tmp_path / "run"
    run_track_condition(
        track="math500",
        split=(0, 1),
        model_key="qwen2.5-math-7b",
        out_dir=out,
        condition="polaris_full_verified_memory",
        archive=MATH500_ARCHIVE_V1,
        cell_fitness={},
        sampler=_MemorySampler(),
        problems=[Problem(problem_id="p1", prompt="What is 6*7?", answer="42", source="math")],
        seed=17,
        archive_hash="hash",
        polaris_source_hash="dev",
        vendored_commits={},
        preregistration_anchor="TODO.md#test",
        run_stage="final",
        preflight_report={"passed": True, "protocol_sync_passed": True},
        run_plan_cell={
            "track": "math500",
            "model_key": "qwen2.5-math-7b",
            "split": [0, 1],
            "condition": "polaris_full_verified_memory",
            "run_stage": "final",
            "archive_build_id": "archive-build-test",
        },
        dataset_source_kind="real_or_cached",
    )
    (out / "memory_events.jsonl").unlink(missing_ok=True)
    lock_path = _write_lock(tmp_path / "locks.json")
    with pytest.raises(ArtifactAuditError, match="memory"):
        audit_run_artifacts(out, stage="final", dataset_lock_path=lock_path)


def test_build_polaris_archive_dry_run_emits_publishable_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "archive_build"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_polaris_archive.py",
            "--track",
            "math500",
            "--mode",
            "freeze",
            "--dev-split",
            "0",
            "1",
            "--out",
            str(out),
            "--dry-run",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in (
        "archive.json",
        "memory.sqlite",
        "descriptor_audit.json",
        "archive_build_manifest.json",
        "rollouts.json",
    ):
        assert (out / name).exists()
