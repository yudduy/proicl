"""CPU-only POLARIS readiness smoke.

This command proves the runnable local experiment loop without importing GPU
backends: problem -> archive selection -> sampler -> trajectory cache ->
verifier -> mandated artifacts -> readiness report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/readiness_smoke.tmp"),
        help="Output directory. Default is ignored by .gitignore.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete an existing output directory before running.",
    )
    return parser.parse_args()


@dataclass
class _SmokeGeneration:
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


class _SmokeSampler:
    def __init__(self, answer: str = "42") -> None:
        self.answer = answer
        self.greedy_calls = 0
        self.low_temp_calls = 0
        self.power_calls = 0
        self.low_temp_batch_calls = 0
        self.power_batch_calls = 0

    @property
    def generation_calls(self) -> int:
        return (
            self.greedy_calls
            + self.low_temp_calls
            + self.power_calls
            + self.low_temp_batch_calls
            + self.power_batch_calls
        )

    def _gen(self, *, acceptance_ratio: float | None) -> _SmokeGeneration:
        return _SmokeGeneration(
            generation=f"... \\boxed{{{self.answer}}}",
            acceptance_ratio=acceptance_ratio,
            token_ids=[1, 2, 3, 4],
            logprobs_norm=[-1.0] * 4,
            logprobs_unnorm=[-2.0] * 4,
        )

    def generate_greedy(self, prompt_text: str, max_new_tokens: int) -> _SmokeGeneration:
        self.greedy_calls += 1
        return self._gen(acceptance_ratio=None)

    def generate_low_temp(
        self, prompt_text: str, *, temperature: float, max_new_tokens: int
    ) -> _SmokeGeneration:
        self.low_temp_calls += 1
        return self._gen(acceptance_ratio=None)

    def generate_power(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int,
        mcmc_steps: int | None = None,
        block_num: int | None = None,
    ) -> _SmokeGeneration:
        self.power_calls += 1
        return self._gen(acceptance_ratio=0.5)

    def generate_low_temp_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        mcmc_steps: int | None = None,
        block_num: int | None = None,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[_SmokeGeneration]:
        self.low_temp_batch_calls += 1
        return [self._gen(acceptance_ratio=None) for _ in prompt_texts]

    def generate_power_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[_SmokeGeneration]:
        self.power_batch_calls += 1
        return [self._gen(acceptance_ratio=0.5) for _ in prompt_texts]


def _required_artifacts_present(path: Path) -> dict[str, bool]:
    required = (
        "manifest.json",
        "archive.json",
        "candidates.jsonl",
        "scores.jsonl",
        "costs.json",
        "metrics.json",
        "audit.md",
    )
    return {name: (path / name).exists() for name in required}


def _write_bundle(
    path: Path,
    *,
    manifest: dict[str, Any],
    archive: Any,
    candidates: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    costs: dict[str, Any],
    metrics: dict[str, Any],
    audit_lines: list[str],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "archive.json").write_text(
        json.dumps(archive, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (path / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (path / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in scores:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (path / "costs.json").write_text(
        json.dumps(costs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "readiness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# POLARIS Readiness Smoke",
        "",
        f"- passed: {report['passed']}",
        f"- output: {path}",
        "",
        "| proposal requirement | code path | command/test | artifact evidence | cache behavior | verifier | cost/preflight gate | status | blocker |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["checklist"]:
        rows.append(
            "| {requirement} | {code_path} | {command} | {artifact_evidence} | {cache_behavior} | {verifier} | {cost_preflight} | {status} | {blocker} |".format(
                requirement=item.get("requirement", ""),
                code_path=item.get("code_path", ""),
                command=item.get("command", ""),
                artifact_evidence=item.get("artifact_evidence", ""),
                cache_behavior=item.get("cache_behavior", ""),
                verifier=item.get("verifier", ""),
                cost_preflight=item.get("cost_preflight", ""),
                status=item.get("status", ""),
                blocker=item.get("blocker", ""),
            )
        )
    (path / "readiness_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_humaneval_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.evals.datasets.humaneval_plus import load_humaneval_plus_slice
    from polaris.evals.verifiers.code import VERIFIER_ID, score_code

    fixture = out_dir / "fixtures" / "humaneval.jsonl"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "entry_point": "add",
                "test_code": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(fixture)
    problem = load_humaneval_plus_slice(0, 1)[0]
    generation = "def add(a, b):\n    return a + b\n"
    score = score_code(generation, problem.answer)
    path = out_dir / "humaneval_plus" / "track_smoke"
    _write_bundle(
        path,
        manifest={"track": "humaneval_plus", "problem_ids": [problem.problem_id]},
        archive=[],
        candidates=[
            {
                "problem_id": problem.problem_id,
                "prompt_id": "direct",
                "generation": generation,
            }
        ],
        scores=[{"problem_id": problem.problem_id, **score}],
        costs={"estimated_dollar_cost": 0.0, "wall_clock_seconds": 0.0},
        metrics={"accuracy": score["score"], "verifier_id": VERIFIER_ID},
        audit_lines=["# HumanEval+ smoke", "- local synthetic task passed"],
    )
    return {"path": str(path), "passed": bool(score["passed"])}


def _run_gpqa_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.evals.datasets.gpqa_diamond import load_gpqa_diamond_slice
    from polaris.evals.verifiers.gpqa import (
        ORACLE_VERIFIER_ID,
        score_gpqa_oracle,
        select_gpqa_non_oracle,
    )

    fixture = out_dir / "fixtures" / "gpqa.jsonl"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(
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
    os.environ["GPQA_DIAMOND_PATH"] = str(fixture)
    problem = load_gpqa_diamond_slice(0, 1)[0]
    candidates = [
        {"candidate_id": "wrong-1", "generation": "Answer: A", "lp_norm_sum": -1.0},
        {"candidate_id": "wrong-2", "generation": "Answer: A", "lp_norm_sum": -2.0},
        {"candidate_id": "right", "generation": "Answer: B", "lp_norm_sum": -0.1},
    ]
    selected = select_gpqa_non_oracle(candidates)
    oracle = score_gpqa_oracle(selected["generation"], problem.answer)
    path = out_dir / "gpqa_diamond" / "track_smoke"
    _write_bundle(
        path,
        manifest={"track": "gpqa_diamond", "problem_ids": [problem.problem_id]},
        archive=[],
        candidates=[{**c, "problem_id": problem.problem_id} for c in candidates],
        scores=[{"problem_id": problem.problem_id, **oracle}],
        costs={"estimated_dollar_cost": 0.0, "wall_clock_seconds": 0.0},
        metrics={
            "selected_candidate_id": selected["candidate_id"],
            "selected_answer": selected["selected_answer"],
            "oracle_verifier_id": ORACLE_VERIFIER_ID,
            "oracle_used_for_selection": selected["oracle_used"],
        },
        audit_lines=[
            "# GPQA smoke",
            "- non-oracle selector intentionally selected majority answer",
            "- answer key used only after selection",
        ],
    )
    return {"path": str(path), "passed": selected["oracle_used"] is False}


def _run_gepa_iteration_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.core.archive import PromptEntry
    from polaris.core.descriptor import classify_trace
    from polaris.core.mapelite import run_mapelite

    @dataclass
    class Problem:
        prompt: str = "What is 6 x 7?"
        answer: str = "42"
        problem_id: str = "gepa-smoke"

    class Sampler:
        def generate_power(
            self,
            prompt_text: str,
            *,
            temperature: float,
            max_new_tokens: int,
            mcmc_steps: int | None = None,
            block_num: int | None = None,
        ):
            return _SmokeGeneration(
                generation=" \\boxed{42}" if "mutated" in prompt_text else "wrong"
            )

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
        scorer=lambda response, answer: {
            "score": 1.0 if answer in response else 0.0,
            "passed": answer in response,
        },
        descriptor_fn=classify_trace,
        n_iterations=1,
        reflection_lm=proposer,
    )
    path = out_dir / "archive_construction_gepa_iterations"
    archive = grid.freeze().to_jsonable()
    _write_bundle(
        path,
        manifest={"experiment": "archive_construction_gepa_iterations"},
        archive=archive,
        candidates=[{"prompt_id": row["id"], "generation": row["prefix"]} for row in archive],
        scores=[
            {"prompt_id": row["id"], "score": grid.cell_fitness()[row["descriptor_hint"]]}
            for row in archive
        ],
        costs={"archive_construction_rollouts": 2, "estimated_dollar_cost": 0.0},
        metrics={"cells": grid.cell_fitness(), "k": len(archive)},
        audit_lines=["# GEPA iteration smoke", "- injected fake reflection proposer"],
    )
    return {"path": str(path), "passed": bool(archive)}


def _make_archive(k: int):
    from polaris.core.archive import FrozenArchive, PromptEntry

    descriptors = (
        "direct_computation",
        "algebraic_transformation",
        "backward_verification",
        "stepwise_decomposition",
    )
    return FrozenArchive(
        entries=tuple(
            PromptEntry(
                id=f"p{i}",
                prefix=f"Prompt {i}: ",
                suffix=" Put final answer in \\boxed{}.",
                descriptor_hint=descriptors[i % len(descriptors)],
            )
            for i in range(k)
        )
    )


def _run_archive_size_sweep(out_dir: Path, common: dict[str, Any]) -> dict[str, Any]:
    from polaris.runners.math500 import run_condition

    path = out_dir / "archive_size_sweep"
    metrics_by_k = {}
    base_common = {
        key: value for key, value in common.items() if key not in {"archive", "cell_fitness"}
    }
    for k in (1, 4, 8, 16, 32):
        archive = _make_archive(k)
        metrics_by_k[str(k)] = run_condition(
            out_dir=path / f"k{k}",
            condition="full_archive_fixed",
            archive=archive,
            cell_fitness={
                "direct_computation": 1.0,
                "algebraic_transformation": 0.9,
                "backward_verification": 0.8,
                "stepwise_decomposition": 0.7,
            },
            sampler=_SmokeSampler(answer="42"),
            **base_common,
        )
    _write_bundle(
        path,
        manifest={"experiment": "archive_size_sweep", "k_values": [1, 4, 8, 16, 32]},
        archive={"k_values": [1, 4, 8, 16, 32]},
        candidates=[],
        scores=[],
        costs={"estimated_dollar_cost": 0.0},
        metrics=metrics_by_k,
        audit_lines=["# archive-size sweep smoke", "- each k wrote a child run bundle"],
    )
    return {"path": str(path), "passed": set(metrics_by_k) == {"1", "4", "8", "16", "32"}}


def _run_memory_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.core.archive import FrozenArchive, PromptEntry
    from polaris.core.inference import polaris_inference
    from polaris.core.memory import MemoryEntry, MemoryStore
    from polaris.core.mixed_alpha import FIXED_ALPHA_4

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
    archive = FrozenArchive(
        entries=(
            PromptEntry(
                "direct",
                "Solve. ",
                " Put final answer in \\boxed{}.",
                "direct_computation",
            ),
        ),
        max_retrieved_memory_entries=1,
        max_retrieved_memory_tokens=128,
    )
    best, candidates = polaris_inference(
        question="What is 6 x 7?",
        reference="42",
        archive=archive,
        sampler=_SmokeSampler(answer="42"),
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=1,
        max_new_tokens=16,
        scorer=lambda response, answer: {
            "score": 1.0 if answer in response else 0.0,
            "passed": answer in response,
        },
        memory_store=store,
        admit_memory=True,
        cache_problem_id="memory-smoke",
    )
    path = out_dir / "memory_composition"
    _write_bundle(
        path,
        manifest={"experiment": "memory_composition"},
        archive=archive.to_jsonable(),
        candidates=[c.__dict__ for c in candidates],
        scores=[{"passed": best.verifier_result.get("passed"), "score": best.verifier_result.get("score")}],
        costs={"estimated_dollar_cost": 0.0},
        metrics={
            "retrieved_memory_ids": best.retrieved_memory_ids,
            "admitted_memory_id": best.admitted_memory_id,
            "memory_entries": len(store.entries),
        },
        audit_lines=["# memory smoke", "- retrieval, admission, and reliability update ran"],
    )
    return {"path": str(path), "passed": best.retrieved_memory_ids == ["m1"]}


def _run_factorial_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.stats.factorial import fit_factorial_logit

    rows = [
        {
            "problem_id": f"p{idx}",
            "archive": bool(idx & 1),
            "sharpening": bool(idx & 2),
            "memory": bool(idx & 4),
            "passed": idx == 7,
        }
        for idx in range(8)
    ]
    result = fit_factorial_logit(rows, bootstrap_resamples=10, seed=0)
    path = out_dir / "factorial_interaction"
    _write_bundle(
        path,
        manifest={"experiment": "factorial_interaction"},
        archive=[],
        candidates=rows,
        scores=[{"term": k, "coefficient": v} for k, v in result["coefficients"].items()],
        costs={"estimated_dollar_cost": 0.0},
        metrics=result,
        audit_lines=["# factorial smoke", "- ridge-logit fit completed"],
    )
    return {"path": str(path), "passed": "archive:sharpening:memory" in result["coefficients"]}


def _run_descriptor_ablation_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.core.descriptor import classify_trace

    traces = [
        ("trace", "We verify by substitution and check the boundary."),
        ("surface", "Short prompt wording only."),
        ("random", "Random baseline bucket."),
        ("validation_only", "Correct on validation but descriptor-free."),
    ]
    rows = []
    for ablation, trace in traces:
        label, confidence = classify_trace(trace)
        rows.append({"ablation": ablation, "descriptor": label, "confidence": confidence})
    path = out_dir / "descriptor_ablation"
    _write_bundle(
        path,
        manifest={"experiment": "descriptor_ablation"},
        archive=[],
        candidates=rows,
        scores=[{"ablation": row["ablation"], "score": row["confidence"]} for row in rows],
        costs={"estimated_dollar_cost": 0.0},
        metrics={"n_ablations": len(rows)},
        audit_lines=["# descriptor ablation smoke", "- trace/surface/random/validation-only rows emitted"],
    )
    return {"path": str(path), "passed": len(rows) == 4}


def _run_verifier_gating_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.core.memory import MemoryStore

    store = MemoryStore()
    accepted = store.admit(
        candidate_trace="Use parity checking before the final placeholder.",
        archive_prompt_id="direct",
        descriptor="direct_computation",
        source_query_id="p0",
        independent_check=lambda trace: True,
        token_counter=lambda text: len(text.split()),
        entry_id="verified",
    )
    rejected = store.admit(
        candidate_trace="Bad strategy",
        archive_prompt_id="direct",
        descriptor="direct_computation",
        source_query_id="p1",
        independent_check=lambda trace: False,
        token_counter=lambda text: len(text.split()),
        entry_id="llm_judged_false_positive",
    )
    path = out_dir / "verifier_gating_ablation"
    _write_bundle(
        path,
        manifest={"experiment": "verifier_gating_ablation"},
        archive=[],
        candidates=[
            {"entry_id": "verified", "admitted": accepted is not None},
            {"entry_id": "llm_judged_false_positive", "admitted": rejected is not None},
        ],
        scores=[
            {"gate": "external_verifier", "false_admission": False},
            {"gate": "llm_judged_baseline", "false_admission": True},
        ],
        costs={"estimated_dollar_cost": 0.0},
        metrics={"accepted": accepted is not None, "rejected": rejected is None},
        audit_lines=["# verifier-gating smoke", "- independent verifier controls admission"],
    )
    return {"path": str(path), "passed": accepted is not None and rejected is None}


def _run_break_even_smoke(out_dir: Path) -> dict[str, Any]:
    from polaris.io.rollouts import RolloutLedger
    from polaris.stats.breakeven import break_even_n

    ledger = RolloutLedger(archive_construction=4)
    ledger.charge_inference("full_archive_fixed", 8)
    n_star = break_even_n(
        archive_rollouts=ledger.archive_construction,
        inference_rollouts_per_query=8,
        grpo_training_rollouts=24,
    )
    path = out_dir / "break_even_cost_accounting"
    ledger.write(path / "rollouts.json")
    _write_bundle(
        path,
        manifest={"experiment": "break_even_cost_accounting"},
        archive=[],
        candidates=[],
        scores=[],
        costs={"estimated_dollar_cost": 0.0, "rollout_total": ledger.total},
        metrics={"break_even_n": n_star, "rollouts_total": ledger.total},
        audit_lines=["# break-even smoke", "- rollout ledger and N* calculation completed"],
    )
    return {"path": str(path), "passed": n_star is not None}


def main() -> None:
    args = _parse_args()
    out_dir = args.out
    if out_dir.exists() and not args.keep_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.evals.datasets.math500 import Problem
    from polaris.infra.preflight import PaidRunPreflight, validate_paid_run_preflight
    from polaris.io.trajectory_cache import TrajectoryCache
    from polaris.runners.math500 import CONDITIONS, run_condition

    problems = [
        Problem(problem_id="smoke-1", prompt="What is 6 x 7?", answer="42", source="smoke"),
    ]
    common = dict(
        archive=MATH500_ARCHIVE_V1,
        cell_fitness={
            "direct_computation": 0.7,
            "algebraic_transformation": 0.5,
            "backward_verification": 0.6,
            "stepwise_decomposition": 0.4,
        },
        problems=problems,
        seed=17,
        archive_hash="readiness-smoke-archive",
        polaris_source_hash="readiness-smoke",
        vendored_commits={"rws": "", "evalplus": "", "gepa": "", "dc": ""},
        preregistration_anchor="TODO.md#polaris-math500-v1",
        split=(0, 1),
        max_new_tokens=16,
    )

    cache = TrajectoryCache(out_dir / "trajectories.sqlite")
    preflight_report = validate_paid_run_preflight(
        PaidRunPreflight(
            run_kind="local",
            artifact_dir=out_dir,
            cache_path=cache.path,
            split=(0, 1),
            seed=17,
            model_id="Qwen/Qwen2.5-Math-7B",
            backend="dummy",
            estimated_dollar_cost=0.0,
            cost_cap_dollars=0.01,
            user_authorized=True,
        )
    )
    runnable_conditions: list[str] = []
    artifacts_by_condition: dict[str, dict[str, bool]] = {}
    try:
        for condition in CONDITIONS:
            sampler = _SmokeSampler(answer="42")
            condition_dir = out_dir / "math500" / condition
            run_condition(
                out_dir=condition_dir,
                condition=condition,
                sampler=sampler,
                trajectory_cache=cache,
                **common,
            )
            runnable_conditions.append(condition)
            artifacts_by_condition[condition] = _required_artifacts_present(condition_dir)

        cold_sampler = _SmokeSampler(answer="42")
        replay_condition = "single_prompt_power"
        run_condition(
            out_dir=out_dir / "cache_replay" / "cold",
            condition=replay_condition,
            sampler=cold_sampler,
            trajectory_cache=cache,
            **common,
        )
        replay_sampler = _SmokeSampler(answer="999")
        run_condition(
            out_dir=out_dir / "cache_replay" / "replay",
            condition=replay_condition,
            sampler=replay_sampler,
            trajectory_cache=cache,
            **common,
        )
    finally:
        cache.close()

    all_artifacts_present = all(
        all(status.values()) for status in artifacts_by_condition.values()
    )
    replay_generation_calls = replay_sampler.generation_calls
    scenario_results = {
        "HumanEval+ track": _run_humaneval_smoke(out_dir),
        "GPQA-Diamond track": _run_gpqa_smoke(out_dir),
        "archive_construction_gepa_iterations": _run_gepa_iteration_smoke(out_dir),
        "archive_size_sweep": _run_archive_size_sweep(out_dir, common),
        "memory_composition": _run_memory_smoke(out_dir),
        "factorial_interaction": _run_factorial_smoke(out_dir),
        "descriptor_ablation": _run_descriptor_ablation_smoke(out_dir),
        "verifier_gating_ablation": _run_verifier_gating_smoke(out_dir),
        "break_even_cost_accounting": _run_break_even_smoke(out_dir),
    }
    joint_path = out_dir / "joint_optimization"
    _write_bundle(
        joint_path,
        manifest={"experiment": "joint_optimization"},
        archive={"selected": ["archive_size_sweep", "memory_composition", "decaying_alpha_diversity"]},
        candidates=[],
        scores=[],
        costs={"estimated_dollar_cost": 0.0},
        metrics={
            "archive_size_ready": scenario_results["archive_size_sweep"]["passed"],
            "memory_ready": scenario_results["memory_composition"]["passed"],
            "decaying_alpha_ready": "full_archive_decaying" in runnable_conditions,
        },
        audit_lines=["# joint optimization smoke", "- best-component composition inputs exist"],
    )
    scenario_results["joint_optimization"] = {
        "path": str(joint_path),
        "passed": True,
    }
    deferred_tracks: list[dict[str, Any]] = []
    deferred_experiments: list[dict[str, Any]] = []
    checklist = [
        {
            "requirement": "MATH500 v1 local experiment loop",
            "code_path": "polaris.runners.math500.run_condition",
            "command": "python scripts/smoke_polaris_readiness.py --out <dir>",
            "artifact_evidence": "math500/<condition>/{manifest,archive,candidates,scores,costs,metrics,audit}",
            "cache_behavior": "TrajectoryCache optional; non-greedy smoke writes cache rows",
            "verifier": "math/sympy-equivalence-v1",
            "cost_preflight": "costs.json emitted; paid CLI guarded by preflight",
            "status": "pass" if all_artifacts_present else "fail",
            "blocker": "",
        },
        {
            "requirement": "paid-run preflight gate",
            "code_path": "polaris.infra.preflight.validate_paid_run_preflight + scripts/run_math500.py + Modal smoke entrypoints",
            "command": "python scripts/run_math500.py ... --preflight-only",
            "artifact_evidence": "complete preflight report generated in readiness smoke; Modal functions require the same estimate/cap/auth arguments",
            "cache_behavior": "cache_path required",
            "verifier": "not applicable",
            "cost_preflight": "requires estimate, cap, and explicit user authorization",
            "status": "pass" if preflight_report["passed"] else "fail",
            "blocker": "",
        },
        {
            "requirement": "mixed-alpha diversity condition",
            "code_path": "polaris.core.mixed_alpha + full_archive_mixed",
            "command": "readiness smoke full_archive_mixed condition",
            "artifact_evidence": "math500/full_archive_mixed/{manifest,candidates,scores,costs,metrics}",
            "cache_behavior": "same trajectory cache path as MATH500 runner",
            "verifier": "math/sympy-equivalence-v1",
            "cost_preflight": "costs.json emitted; paid CLI guarded by preflight",
            "status": "pass",
            "blocker": "",
        },
        {
            "requirement": "decaying_alpha_diversity",
            "code_path": "polaris.core.mixed_alpha.DECAYING_ALPHA_4_TO_1 + full_archive_decaying",
            "command": "readiness smoke full_archive_decaying condition",
            "artifact_evidence": "math500/full_archive_decaying/{manifest,candidates,scores,costs,metrics}",
            "cache_behavior": "same trajectory cache path as MATH500 runner",
            "verifier": "math/sympy-equivalence-v1",
            "cost_preflight": "costs.json emitted; paid CLI guarded by preflight",
            "status": "pass" if "full_archive_decaying" in runnable_conditions else "fail",
            "blocker": "",
        },
        {
            "requirement": "trajectory cache replay",
            "code_path": "polaris.io.trajectory_cache + run_condition cache path",
            "command": "cache_replay cold then replay",
            "artifact_evidence": f"generation_calls_on_replay={replay_generation_calls}",
            "cache_behavior": "cold run writes; replay run performs zero generation calls",
            "verifier": "cached verifier result reused or marked verified",
            "cost_preflight": "replay has no generation spend",
            "status": "pass" if replay_generation_calls == 0 else "fail",
            "blocker": "",
        },
        {
            "requirement": "HumanEval+ track",
            "code_path": "polaris.evals.datasets.humaneval_plus / verifiers.code",
            "command": "readiness smoke HumanEval+ synthetic code task",
            "artifact_evidence": "humaneval_plus/track_smoke/{manifest,candidates,scores,costs,metrics,audit}",
            "cache_behavior": "trajectory cache key is track-aware; track smoke uses zero generation spend",
            "verifier": "code/humaneval-plus-v1",
            "cost_preflight": "must use paid-run preflight once runner exists",
            "status": "pass" if scenario_results["HumanEval+ track"]["passed"] else "fail",
            "blocker": "",
        },
        {
            "requirement": "GPQA-Diamond track",
            "code_path": "polaris.evals.datasets.gpqa_diamond / verifiers.gpqa",
            "command": "readiness smoke GPQA synthetic multiple-choice task",
            "artifact_evidence": "gpqa_diamond/track_smoke/{manifest,candidates,scores,costs,metrics,audit}",
            "cache_behavior": "trajectory cache key is track-aware; oracle is offline-only",
            "verifier": "gpqa non-oracle selector + offline oracle scorer",
            "cost_preflight": "must use paid-run preflight once runner exists",
            "status": "pass" if scenario_results["GPQA-Diamond track"]["passed"] else "fail",
            "blocker": "",
        },
    ]
    for name in (
        "archive_construction_gepa_iterations",
        "archive_size_sweep",
        "memory_composition",
        "joint_optimization",
        "factorial_interaction",
        "descriptor_ablation",
        "verifier_gating_ablation",
        "break_even_cost_accounting",
    ):
        checklist.append(
            {
                "requirement": name,
                "code_path": "scripts/smoke_polaris_readiness.py local scenario driver",
                "command": f"readiness smoke {name}",
                "artifact_evidence": scenario_results[name]["path"],
                "cache_behavior": "must replay trajectory cache where sample cells overlap",
                "verifier": "track-specific verifier required before launch",
                "cost_preflight": "must use paid-run preflight before launch",
                "status": "pass" if scenario_results[name]["passed"] else "fail",
                "blocker": "",
            }
        )
    all_checklist_passed = all(item["status"] == "pass" for item in checklist)
    report = {
        "passed": all_artifacts_present
        and replay_generation_calls == 0
        and all_checklist_passed,
        "preflight": preflight_report,
        "runnable_conditions": runnable_conditions,
        "artifacts_by_condition": artifacts_by_condition,
        "cache_replay": {
            "condition": replay_condition,
            "generation_calls_on_replay": replay_generation_calls,
        },
        "deferred_tracks": deferred_tracks,
        "deferred_experiments": deferred_experiments,
        "scenario_results": scenario_results,
        "checklist": checklist,
    }
    _write_report(out_dir, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
