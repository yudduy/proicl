from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from polaris.core.archive import FrozenArchive, PromptEntry
from polaris.core.descriptor import classify_trace
from polaris.core.inference import polaris_inference
from polaris.core.mapelite import run_mapelite
from polaris.core.memory import MemoryEntry, MemoryStore
from polaris.core.mixed_alpha import DECAYING_ALPHA_4_TO_1, FIXED_ALPHA_4
from polaris.evals.datasets.gpqa_diamond import load_gpqa_diamond_slice
from polaris.evals.datasets.humaneval_plus import load_humaneval_plus_slice
from polaris.evals.verifiers.code import score_code
from polaris.evals.verifiers.gpqa import (
    score_gpqa_oracle,
    select_gpqa_non_oracle,
)
from polaris.stats.factorial import fit_factorial_logit


class _Gen:
    generation = "\nUse the cancellation tactic, then return \\boxed{42}."
    response_contains_prompt = False
    prompt_token_count = 4
    generation_token_count = 8
    wall_clock_seconds = 0.0
    estimated_dollar_cost = 0.0
    acceptance_ratio = 0.5
    token_ids = [1, 2]
    logprobs_norm = [-1.0, -1.0]
    logprobs_unnorm = [-4.0, -4.0]


class _Sampler:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        self.prompts.append(prompt_text)
        return _Gen()

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        self.prompts.append(prompt_text)
        return _Gen()


def _archive(*, memory: bool = False) -> FrozenArchive:
    return FrozenArchive(
        entries=(
            PromptEntry(
                id="direct",
                prefix="Solve. ",
                suffix=" Answer in a box.",
                descriptor_hint="direct_computation",
            ),
        ),
        max_retrieved_memory_entries=1 if memory else 0,
        max_retrieved_memory_tokens=128 if memory else 0,
    )


def test_humaneval_plus_loader_reads_override_jsonl(tmp_path, monkeypatch):
    dataset = tmp_path / "humaneval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "entry_point": "add",
                "test_code": "assert add(1, 2) == 3",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HUMANEVAL_OVERRIDE_PATH", str(dataset))

    rows = load_humaneval_plus_slice(0, 1)

    assert rows[0].problem_id == "HumanEval/0"
    assert "def add" in rows[0].prompt
    assert json.loads(rows[0].answer)["entry_point"] == "add"


def test_code_verifier_executes_synthetic_humaneval_tests():
    reference = json.dumps(
        {
            "entry_point": "add",
            "test_code": "assert add(2, 5) == 7\nassert add(-1, 1) == 0",
        }
    )

    passed = score_code("def add(a, b):\n    return a + b\n", reference)
    failed = score_code("def add(a, b):\n    return a - b\n", reference)

    assert passed["passed"] is True
    assert passed["score"] == 1.0
    assert failed["passed"] is False
    assert failed["score"] == 0.0


def test_gpqa_loader_reads_local_jsonl_and_selector_is_non_oracle(tmp_path, monkeypatch):
    dataset = tmp_path / "gpqa.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "problem_id": "gpqa-0",
                "prompt": "Which option is correct?\nA. wrong\nB. right",
                "answer": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GPQA_DIAMOND_PATH", str(dataset))

    problems = load_gpqa_diamond_slice(0, 1)
    assert problems[0].problem_id == "gpqa-0"
    assert score_gpqa_oracle("The answer is (B).", problems[0].answer)["passed"] is True

    selected = select_gpqa_non_oracle(
        [
            {"candidate_id": "wrong-1", "generation": "Answer: A", "lp_norm_sum": -1.0},
            {"candidate_id": "wrong-2", "generation": "Answer: A", "lp_norm_sum": -2.0},
            {"candidate_id": "right", "generation": "Answer: B", "lp_norm_sum": -0.1},
        ]
    )
    assert selected["candidate_id"] == "wrong-1"
    assert selected["selected_answer"] == "A"


def test_mapelite_iterations_use_injected_reflection_candidate():
    class Problem:
        prompt = "q"
        answer = "42"
        problem_id = "p"

    class Sampler:
        def generate_power(
            self,
            prompt_text,
            *,
            temperature,
            max_new_tokens,
            mcmc_steps=None,
            block_num=None,
        ):
            class Gen:
                generation = "42" if "mutated" in prompt_text else "wrong"
                response_contains_prompt = False

            return Gen()

    def proposer(prompt, iteration, grid):
        return PromptEntry(
            id=f"{prompt.id}_mutated_{iteration}",
            prefix="mutated ",
            suffix="",
            descriptor_hint="direct_computation",
        )

    grid = run_mapelite(
        seeds=(PromptEntry("seed", "seed ", "", "direct_computation"),),
        dev_set=[Problem()],
        sampler=Sampler(),
        scorer=lambda response, answer: {"score": 1.0 if answer in response else 0.0},
        descriptor_fn=classify_trace,
        n_iterations=1,
        reflection_lm=proposer,
    )

    assert grid.freeze().entries[0].id == "seed_mutated_0"


def test_memory_enabled_inference_retrieves_admits_and_updates_reliability():
    store = MemoryStore(
        entries=[
            MemoryEntry(
                id="m1",
                archive_prompt_id="direct",
                descriptor="direct_computation",
                strategy_text="Cancel common factors before computing.",
                token_count=6,
                source_query_id="seed",
            )
        ]
    )
    sampler = _Sampler()

    best, candidates = polaris_inference(
        question="What is 6*7?",
        reference="42",
        archive=_archive(memory=True),
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=1,
        max_new_tokens=16,
        scorer=lambda response, answer: {
            "score": 1.0 if answer in response else 0.0,
            "passed": answer in response,
        },
        memory_store=store,
        cache_problem_id="p0",
        admit_memory=True,
    )

    assert best.retrieved_memory_ids == ["m1"]
    assert "Cancel common factors" in sampler.prompts[0]
    assert candidates[0].admitted_memory_id is not None
    assert store.entries[0].reliability_alpha == 2.0


def test_decaying_alpha_schedule_is_registered():
    assert [DECAYING_ALPHA_4_TO_1.alpha("p", i) for i in range(5)] == [
        4.0,
        3.0,
        2.0,
        1.0,
        4.0,
    ]


def test_factorial_fit_returns_three_way_interaction_shape():
    rows = []
    for archive in (False, True):
        for sharpening in (False, True):
            for memory in (False, True):
                rows.append(
                    {
                        "problem_id": f"p-{archive}-{sharpening}-{memory}",
                        "archive": archive,
                        "sharpening": sharpening,
                        "memory": memory,
                        "passed": archive and sharpening and memory,
                    }
                )

    result = fit_factorial_logit(rows, bootstrap_resamples=10, seed=0)

    assert "archive:sharpening:memory" in result["coefficients"]
    assert "ci95" in result["three_way"]
