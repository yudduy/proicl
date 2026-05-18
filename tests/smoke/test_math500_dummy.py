"""CPU-only end-to-end smoke: 2 problems x 6 conditions with a dummy sampler.

Verifies all 7 proposal-mandated artifacts are emitted per condition and that
the runner orchestration is wired correctly without needing torch on Mac.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.evals.datasets.math500 import Problem
from polaris.io.trajectory_cache import TrajectoryCache
from polaris.runners.math500 import CONDITIONS, run_condition


@dataclass
class _DummyGen:
    generation: str
    response_contains_prompt: bool = False
    prompt_token_count: int = 4
    generation_token_count: int = 4
    wall_clock_seconds: float = 0.001
    estimated_dollar_cost: float = 0.0
    acceptance_ratio: float | None = 0.5
    token_ids: list[int] | None = None
    logprobs_norm: list[float] | None = None
    logprobs_unnorm: list[float] | None = None


class _DummySampler:
    """All three paths return canned answer-containing generations."""

    def __init__(self, canned_answer: str = "42"):
        self._answer = canned_answer
        self.power_calls = 0
        self.low_temp_calls = 0
        self.greedy_calls = 0

    def generate_greedy(self, prompt_text, max_new_tokens):
        self.greedy_calls += 1
        return _DummyGen(
            generation=f"... \\boxed{{{self._answer}}}", acceptance_ratio=None
        )

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        self.power_calls += 1
        return _DummyGen(generation=f"... \\boxed{{{self._answer}}}")

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        self.low_temp_calls += 1
        return _DummyGen(
            generation=f"... \\boxed{{{self._answer}}}", acceptance_ratio=None
        )


class _BatchDummySampler(_DummySampler):
    """Batch-capable sampler used to verify vLLM-style runner orchestration."""

    def __init__(self, canned_answer: str = "42"):
        super().__init__(canned_answer=canned_answer)
        self.power_batch_sizes: list[int] = []
        self.low_temp_batch_sizes: list[int] = []

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        raise AssertionError("scalar generate_power should not be used")

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        raise AssertionError("scalar generate_low_temp should not be used")

    def generate_power_batch(
        self,
        prompt_texts,
        *,
        temperature,
        max_new_tokens,
        mcmc_steps=None,
        block_num=None,
        seed_base=None,
        seed_offsets=None,
    ):
        self.power_batch_sizes.append(len(prompt_texts))
        return [
            _DummyGen(
                generation=f"... \\boxed{{{self._answer}}}",
                token_ids=[1, 2, 3, 4],
                logprobs_norm=[-1.0] * 4,
                logprobs_unnorm=[-2.0] * 4,
            )
            for _ in prompt_texts
        ]

    def generate_low_temp_batch(
        self,
        prompt_texts,
        *,
        temperature,
        max_new_tokens,
        seed_base=None,
        seed_offsets=None,
    ):
        self.low_temp_batch_sizes.append(len(prompt_texts))
        return [
            _DummyGen(
                generation=f"... \\boxed{{{self._answer}}}",
                acceptance_ratio=None,
                token_ids=[1, 2, 3, 4],
                logprobs_norm=[-1.0] * 4,
                logprobs_unnorm=[-2.0] * 4,
            )
            for _ in prompt_texts
        ]


_PROBLEMS = [
    Problem(problem_id="p1", prompt="What is 6 x 7?", answer="42", source="math"),
    Problem(problem_id="p2", prompt="What is 21 x 2?", answer="42", source="math"),
]

_CELL_FITNESS = {
    "direct_computation": 0.7,
    "algebraic_transformation": 0.5,
    "backward_verification": 0.6,
    "stepwise_decomposition": 0.4,
}

_KW = dict(
    archive=MATH500_ARCHIVE_V1,
    cell_fitness=_CELL_FITNESS,
    problems=_PROBLEMS,
    seed=17,
    archive_hash="testhash",
    polaris_source_hash="dev",
    vendored_commits={
        "rws": "720a8e9d",
        "evalplus": "26d6d00b",
        "gepa": "ce51b50c",
        "dc": "5cfe3c37",
    },
    preregistration_anchor="TODO.md#smoke-test",
)

_MANDATED_ARTIFACTS = (
    "manifest.json",
    "archive.json",
    "candidates.jsonl",
    "scores.jsonl",
    "selected.jsonl",
    "costs.json",
    "metrics.json",
    "rollouts.json",
    "preflight.json",
    "environment.json",
    "run_plan_cell.json",
    "audit.md",
)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_condition_emits_all_seven_mandated_artifacts(tmp_path, condition):
    out_dir = tmp_path / condition
    sampler = _DummySampler(canned_answer="42")
    metrics = run_condition(
        out_dir=out_dir, condition=condition, sampler=sampler, **_KW
    )
    for name in _MANDATED_ARTIFACTS:
        assert (out_dir / name).exists(), f"{condition} missing {name}"
    assert metrics["condition"] == condition
    assert metrics["n_problems"] == 2


def test_condition_emits_empty_jsonl_files_for_empty_shard(tmp_path):
    out_dir = tmp_path / "empty-shard"
    sampler = _DummySampler(canned_answer="42")
    metrics = run_condition(
        out_dir=out_dir,
        condition="bon_temp1",
        sampler=sampler,
        **{**_KW, "problems": []},
    )

    assert metrics["n_problems"] == 0
    assert (out_dir / "candidates.jsonl").read_text() == ""
    assert (out_dir / "scores.jsonl").read_text() == ""
    assert (out_dir / "selected.jsonl").read_text() == ""


def test_greedy_uses_one_sample_per_problem(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "greedy", condition="greedy", sampler=sampler, **_KW
    )
    assert sampler.greedy_calls == 2  # 2 problems x 1 sample
    assert sampler.power_calls == 0
    assert sampler.low_temp_calls == 0
    rows = (tmp_path / "greedy" / "candidates.jsonl").read_text().splitlines()
    assert len(rows) == 2


def test_full_archive_fixed_uses_eight_samples_per_problem(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "fc", condition="full_archive_fixed", sampler=sampler, **_KW
    )
    rows = (tmp_path / "fc" / "candidates.jsonl").read_text().splitlines()
    assert len(rows) == 2 * 8  # 2 problems x B=8


def test_full_archive_mixed_alternates_paths(tmp_path):
    """MIXED_ALPHA_4_1 cycles alpha=4 (power) and alpha=1 (low_temp) by parity."""
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "fm",
        condition="full_archive_mixed",
        sampler=sampler,
        **_KW,
    )
    # B=8 per problem, half each path, x 2 problems
    assert sampler.power_calls == 8
    assert sampler.low_temp_calls == 8


def test_full_archive_decaying_records_decaying_policy(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "fd",
        condition="full_archive_decaying",
        sampler=sampler,
        **_KW,
    )
    manifest = json.loads((tmp_path / "fd" / "manifest.json").read_text())
    assert manifest["alpha_policy_id"] == "decaying_alpha_4_to_1"
    rows = [
        json.loads(line)
        for line in (tmp_path / "fd" / "candidates.jsonl").read_text().splitlines()
    ]
    assert {row["alpha"] for row in rows} == {1.0, 2.0, 3.0, 4.0}


def test_bon_temp1_only_uses_low_temp_path(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "bon", condition="bon_temp1", sampler=sampler, **_KW
    )
    assert sampler.power_calls == 0
    assert sampler.low_temp_calls == 2 * 8


def test_single_prompt_power_uses_only_direct_entry(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "spp",
        condition="single_prompt_power",
        sampler=sampler,
        **_KW,
    )
    archive_data = json.loads((tmp_path / "spp" / "archive.json").read_text())
    assert len(archive_data) == 1
    assert archive_data[0]["id"] == "direct"


def test_single_best_prompt_picks_highest_cell_fitness(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "sbp",
        condition="single_best_prompt",
        sampler=sampler,
        **_KW,
    )
    archive_data = json.loads((tmp_path / "sbp" / "archive.json").read_text())
    # _CELL_FITNESS direct_computation = 0.7 is highest
    assert len(archive_data) == 1
    assert archive_data[0]["descriptor_hint"] == "direct_computation"


def test_accuracy_is_one_when_all_correct(tmp_path):
    sampler = _DummySampler(canned_answer="42")  # both problems answer is "42"
    metrics = run_condition(
        out_dir=tmp_path / "fc", condition="full_archive_fixed", sampler=sampler, **_KW
    )
    assert metrics["accuracy"] == 1.0


def test_accuracy_is_zero_when_all_wrong(tmp_path):
    sampler = _DummySampler(canned_answer="999")  # always wrong
    metrics = run_condition(
        out_dir=tmp_path / "fc", condition="full_archive_fixed", sampler=sampler, **_KW
    )
    assert metrics["accuracy"] == 0.0


def test_manifest_records_condition_and_alpha_policy(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    run_condition(
        out_dir=tmp_path / "fm",
        condition="full_archive_mixed",
        sampler=sampler,
        **_KW,
    )
    manifest = json.loads((tmp_path / "fm" / "manifest.json").read_text())
    assert manifest["condition"] == "full_archive_mixed"
    assert manifest["alpha_policy_id"] == "mixed_alpha_4_1"
    assert manifest["config"]["trajectory_cache"] is None


def test_run_condition_replays_non_greedy_from_trajectory_cache(tmp_path):
    cache = TrajectoryCache(tmp_path / "trajectories.sqlite")
    try:
        sampler = _DummySampler(canned_answer="42")
        run_condition(
            out_dir=tmp_path / "cold",
            condition="single_prompt_power",
            sampler=sampler,
            trajectory_cache=cache,
            **_KW,
        )
        assert sampler.power_calls == 16

        replay_sampler = _DummySampler(canned_answer="999")
        metrics = run_condition(
            out_dir=tmp_path / "replay",
            condition="single_prompt_power",
            sampler=replay_sampler,
            trajectory_cache=cache,
            **_KW,
        )
        assert replay_sampler.power_calls == 0
        assert replay_sampler.low_temp_calls == 0
        assert metrics["accuracy"] == 1.0
        manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text())
        assert manifest["config"]["trajectory_cache"] == str(cache.path)
    finally:
        cache.close()


def test_run_condition_uses_batch_generation_when_available(tmp_path):
    sampler = _BatchDummySampler(canned_answer="42")
    kw = dict(_KW, problems=_PROBLEMS[:1])
    run_condition(
        out_dir=tmp_path / "batched",
        condition="full_archive_fixed",
        sampler=sampler,
        **kw,
    )
    assert sampler.power_batch_sizes == [8]
    assert sampler.low_temp_batch_sizes == []
    rows = (tmp_path / "batched" / "candidates.jsonl").read_text().splitlines()
    assert len(rows) == 8
    first = json.loads(rows[0])
    assert first["token_ids"] == [1, 2, 3, 4]
    assert first["logprobs_norm"] == [-1.0, -1.0, -1.0, -1.0]


def test_run_condition_batched_cache_replay_avoids_generation(tmp_path):
    cache = TrajectoryCache(tmp_path / "trajectories.sqlite")
    try:
        kw = dict(_KW, problems=_PROBLEMS[:1])
        sampler = _BatchDummySampler(canned_answer="42")
        run_condition(
            out_dir=tmp_path / "cold-batched",
            condition="single_prompt_power",
            sampler=sampler,
            trajectory_cache=cache,
            **kw,
        )
        assert sampler.power_batch_sizes == [8]

        replay_sampler = _BatchDummySampler(canned_answer="999")
        metrics = run_condition(
            out_dir=tmp_path / "replay-batched",
            condition="single_prompt_power",
            sampler=replay_sampler,
            trajectory_cache=cache,
            **kw,
        )
        assert replay_sampler.power_batch_sizes == []
        assert replay_sampler.low_temp_batch_sizes == []
        assert metrics["accuracy"] == 1.0
        replay_rows = (tmp_path / "replay-batched" / "candidates.jsonl").read_text().splitlines()
        assert json.loads(replay_rows[0])["token_ids"] == [1, 2, 3, 4]
    finally:
        cache.close()


def test_runner_rejects_unknown_condition(tmp_path):
    sampler = _DummySampler()
    with pytest.raises(ValueError, match="unknown condition"):
        run_condition(
            out_dir=tmp_path / "bad", condition="nope", sampler=sampler, **_KW
        )


def test_runner_rejects_empty_preregistration_anchor(tmp_path):
    sampler = _DummySampler(canned_answer="42")
    bad_kw = dict(_KW, preregistration_anchor="")
    with pytest.raises(ValueError, match="preregistration"):
        run_condition(
            out_dir=tmp_path / "x",
            condition="full_archive_fixed",
            sampler=sampler,
            **bad_kw,
        )
