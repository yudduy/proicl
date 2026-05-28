from __future__ import annotations

import json
from pathlib import Path

import pytest

from polaris.baselines import ACEPlaybookState, DynamicCheatsheetState
from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.core.memory import MemoryEntry
from polaris.core.persistent_memory import PersistentMemoryLedger
from polaris.evals.datasets.math500 import Problem
from polaris.evals.datasets.gpqa_diamond import Problem as GPQAProblem
from polaris.infra.cost import assert_cost_under_cap, project_cost_from_observed
from polaris.io.artifact_audit import ArtifactAuditError, audit_run_artifacts
from polaris.io.dataset_locks import (
    build_dataset_lock,
    scan_for_gpqa_leakage,
    write_dataset_locks,
)
from polaris.production_plan import build_production_plan
from polaris.registry import (
    CONDITION_REGISTRY,
    validate_model_for_track,
    validate_registry,
)
from polaris.runners.math500 import run_condition
from polaris.runners.condition_runner import run_condition as run_track_condition


class _Gen:
    generation = "... \\boxed{42}"
    response_contains_prompt = False
    prompt_token_count = 3
    generation_token_count = 4
    wall_clock_seconds = 0.01
    estimated_dollar_cost = 0.0
    acceptance_ratio = None
    token_ids = [1]
    logprobs_norm = [-1.0]
    logprobs_unnorm = [-2.0]


class _Sampler:
    def generate_greedy(self, prompt_text, max_new_tokens):
        return _Gen()

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        return _Gen()

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        return _Gen()


class _ChoiceGen:
    response_contains_prompt = False
    prompt_token_count = 3
    generation_token_count = 2
    wall_clock_seconds = 0.0
    estimated_dollar_cost = 0.0
    acceptance_ratio = None
    token_ids = [1]
    logprobs_norm = [-1.0]
    logprobs_unnorm = [-2.0]

    def __init__(self, generation: str) -> None:
        self.generation = generation


class _ChoiceSampler:
    def __init__(self) -> None:
        self.calls = 0

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        self.calls += 1
        return _ChoiceGen("Answer: A" if self.calls <= 5 else "Answer: B")

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        return self.generate_low_temp(
            prompt_text, temperature=temperature, max_new_tokens=max_new_tokens
        )

    def generate_greedy(self, prompt_text, max_new_tokens):
        return self.generate_low_temp(prompt_text, temperature=1.0, max_new_tokens=max_new_tokens)


def _write_clean_lock(tmp_path: Path) -> Path:
    rows = [Problem(problem_id="p1", prompt="q", answer="42", source="math")]
    lock = build_dataset_lock(
        track="math500",
        source_repo="fixture",
        config="math500",
        split="test[0:1]",
        rows=rows,
        row_id=lambda row, idx: row.problem_id,
        loader_version="test",
    )
    path = tmp_path / "datasets.lock.json"
    write_dataset_locks(path, [lock])
    payload = json.loads(path.read_text())
    assert payload["locks"][0]["row_id_hashes"] != ["p1"]
    return path


def _run_fixture_bundle(tmp_path: Path, *, run_stage: str = "small_real_slice") -> Path:
    out = tmp_path / "bundle"
    run_condition(
        out_dir=out,
        condition="greedy",
        archive=MATH500_ARCHIVE_V1,
        cell_fitness={},
        sampler=_Sampler(),
        problems=[Problem(problem_id="p1", prompt="What is 6*7?", answer="42", source="math")],
        seed=17,
        archive_hash="hash",
        polaris_source_hash="dev",
        vendored_commits={},
        preregistration_anchor="TODO.md#test",
        split=(0, 1),
        model_key="qwen2.5-math-7b",
        run_stage=run_stage,
        preflight_report={"passed": True, "protocol_sync_passed": True},
        run_plan_cell={
            "track": "math500",
            "model_key": "qwen2.5-math-7b",
            "split": [0, 1],
            "condition": "greedy",
            "run_stage": run_stage,
        },
        dataset_source_kind="real_or_cached",
    )
    return out


def test_registries_validate_and_reject_track_model_mismatch():
    validate_registry()
    assert "dynamic_cheatsheet" in CONDITION_REGISTRY
    with pytest.raises(ValueError, match="not registered"):
        validate_model_for_track("deepseek-math-7b", "humaneval_plus")


def test_production_plan_has_all_tracks_and_final_cells():
    plan = build_production_plan(stages=("small_real_slice", "final"))
    tracks = {cell["track"] for cell in plan["cells"]}
    stages = {cell["run_stage"] for cell in plan["cells"]}
    assert tracks == {"math500", "humaneval_plus", "gpqa_diamond"}
    assert stages == {"small_real_slice", "final"}
    assert all("model_key" in cell for cell in plan["cells"])


def test_no_gpqa_answer_content_in_tracked_metadata():
    root = Path(__file__).resolve().parents[2]
    offenders = [
        path
        for path in scan_for_gpqa_leakage(root)
        if "upstream" not in path.parts and "runs" not in path.parts
    ]
    assert offenders == []


def test_artifact_audit_fails_missing_production_artifact(tmp_path):
    bundle = _run_fixture_bundle(tmp_path)
    (bundle / "rollouts.json").unlink()
    with pytest.raises(ArtifactAuditError, match="rollouts.json"):
        audit_run_artifacts(bundle, stage="small_real_slice")


def test_artifact_audit_final_requires_clean_lock_and_preflight(tmp_path):
    bundle = _run_fixture_bundle(tmp_path, run_stage="final")
    lock_path = _write_clean_lock(tmp_path)
    report = audit_run_artifacts(bundle, stage="final", dataset_lock_path=lock_path)
    assert report["passed"] is True


def test_persistent_memory_ledger_replays_and_prunes(tmp_path):
    ledger = PersistentMemoryLedger(tmp_path / "memory.sqlite")
    try:
        first = MemoryEntry(
            id="m1",
            archive_prompt_id="direct",
            descriptor="direct",
            strategy_text="Use cancellation.",
            token_count=2,
            source_query_id="p1",
        )
        second = MemoryEntry(
            id="m2",
            archive_prompt_id="direct",
            descriptor="direct",
            strategy_text="Guess.",
            token_count=1,
            source_query_id="p2",
            reliability_alpha=1.0,
            reliability_beta=4.0,
        )
        ledger.admit(first, track="math500", verifier_id="math")
        ledger.admit(second, track="math500", verifier_id="math")
        ledger.record_retrieval(
            query_id="q",
            eligible_ids=["m1", "m2"],
            retrieved_ids=["m1"],
            verifier_metadata={"verifier_id": "math"},
        )
        ledger.update_posterior(["m1"], verifier_outcome=1, query_id="q")
        assert (
            ledger.rollback_incomplete_queries(
                {"p1", "p2"},
                expected_query_ids={"q"},
            )
            == 2
        )
        first_reloaded = next(entry for entry in ledger.entries() if entry.id == "m1")
        assert first_reloaded.reliability_alpha == 1.0
        assert first_reloaded.reliability_beta == 1.0
        ledger.record_retrieval(
            query_id="q",
            eligible_ids=["m1", "m2"],
            retrieved_ids=["m1"],
            verifier_metadata={"verifier_id": "math"},
        )
        ledger.update_posterior(["m1"], verifier_outcome=1, query_id="q")
        assert (
            ledger.rollback_incomplete_queries(
                {"p1", "p2", "q"},
                expected_query_ids={"q"},
            )
            == 0
        )
        pruned = ledger.prune(max_entries_per_prompt=1)
        assert pruned == ["m2"]
        assert [entry.id for entry in ledger.entries()] == ["m1"]
        assert {event["event_type"] for event in ledger.events()} >= {
            "admission",
            "retrieval",
            "posterior_update",
            "prune",
        }
    finally:
        ledger.close()


def test_cost_projection_gate():
    projection = project_cost_from_observed(
        observed_queries=2,
        target_queries=10,
        observed_dollars=1.0,
        observed_wall_clock_seconds=20.0,
        cost_cap_dollars=4.0,
    )
    assert projection.projected_dollars == 5.0
    assert projection.under_cap is False
    with pytest.raises(ValueError, match="exceeds cap"):
        assert_cost_under_cap(projection)


def test_baseline_state_machines_are_source_extracted():
    dc = DynamicCheatsheetState("old")
    assert dc.update_from_response("no sheet") == "old"

    ace = ACEPlaybookState.empty()
    updated = ace.apply_feedback(
        bullet_tags=[],
        curator_operations=[
            {
                "type": "ADD",
                "section": "OTHERS",
                "content": "Prefer verifier-backed answers.",
            }
        ],
    )
    assert "Prefer verifier-backed answers" in updated


def test_gpqa_track_runner_uses_non_oracle_selection(tmp_path):
    out = tmp_path / "gpqa"
    metrics = run_track_condition(
        track="gpqa_diamond",
        split=(0, 1),
        model_key="qwen2.5-7b",
        out_dir=out,
        condition="full_archive_fixed",
        archive=MATH500_ARCHIVE_V1,
        cell_fitness={},
        sampler=_ChoiceSampler(),
        problems=[
            GPQAProblem(
                problem_id="g0",
                prompt="Which is correct?\nA. wrong\nB. right",
                answer="B",
                source="gpqa_diamond",
            )
        ],
        seed=17,
        archive_hash="hash",
        polaris_source_hash="dev",
        vendored_commits={},
        preregistration_anchor="TODO.md#test",
        run_stage="smoke",
    )
    selected = json.loads((out / "selected.jsonl").read_text().splitlines()[0])
    assert selected["selection"]["selector_id"].startswith("gpqa/")
    assert selected["selection"]["oracle_used"] is False
    assert selected["selection"]["selected_answer"] == "A"
    assert metrics["accuracy"] == 0.0


def test_published_ledger_condition_writes_artifacts_without_generation(tmp_path):
    out = tmp_path / "ledger"
    metrics = run_track_condition(
        track="math500",
        split=(0, 1),
        model_key="qwen2.5-math-7b",
        out_dir=out,
        condition="published_grpo_ledger",
        archive=MATH500_ARCHIVE_V1,
        cell_fitness={},
        sampler=None,
        problems=[Problem(problem_id="p1", prompt="q", answer="42", source="math")],
        seed=17,
        archive_hash="hash",
        polaris_source_hash="dev",
        vendored_commits={},
        preregistration_anchor="TODO.md#test",
        run_stage="smoke",
        model_id="Qwen/Qwen2.5-Math-7B",
    )
    assert metrics["requires_external_score"] is True
    assert (out / "rollouts.json").exists()
    assert (out / "selected.jsonl").exists()
