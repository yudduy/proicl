"""Track-agnostic POLARIS condition runner.

This is the production entry point. It delegates shared candidate generation to
the MATH500-era runner while making track, model key, verifier, selector, and
dataset-source policy explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from polaris.baselines import PUBLISHED_COMPARATORS
from polaris.evals.datasets.code_optimizer_dev import load_code_optimizer_dev_slice
from polaris.evals.datasets.gpqa_diamond import (
    GPQA_DIAMOND_TEST_SLICE,
    load_gpqa_diamond_slice,
)
from polaris.evals.datasets.gpqa_non_diamond import load_gpqa_non_diamond_slice
from polaris.evals.datasets.humaneval_plus import (
    HUMANEVAL_PLUS_TEST_SLICE,
    load_humaneval_plus_slice,
)
from polaris.evals.datasets.math500 import MATH500_TEST_SLICE, load_math500_slice
from polaris.evals.datasets.math_optimizer_dev import load_math_optimizer_dev_slice
from polaris.evals.datasets.reasoning_gym import load_reasoning_gym_slice
from polaris.evals.verifiers.code import VERIFIER_ID as CODE_VERIFIER_ID
from polaris.evals.verifiers.code import score_code
from polaris.evals.verifiers.gpqa import (
    NON_ORACLE_SELECTOR_ID,
    ORACLE_VERIFIER_ID,
    score_gpqa_oracle,
    select_gpqa_non_oracle,
)
from polaris.evals.verifiers.math import VERIFIER_ID as MATH_VERIFIER_ID
from polaris.evals.verifiers.math import score_math
from polaris.evals.verifiers.reasoning_gym import VERIFIER_ID as RG_VERIFIER_ID
from polaris.evals.verifiers.reasoning_gym import score_reasoning_gym
from polaris.io.artifacts import write_json, write_jsonl
from polaris.io.manifest import write_run_manifest
from polaris.io.rollouts import RolloutLedger
from polaris.runners.math500 import BestSelector, run_condition as run_condition_bundle


@dataclass(frozen=True)
class TrackConfig:
    track_id: str
    benchmark: str
    dataset_loader: Callable[[int, int], list[Any]]
    optimizer_dev_loader: Callable[[int, int], list[Any]]
    verifier: Callable[[str, str], dict]
    verifier_id: str
    default_split: tuple[int, int]
    inference_time_verifier: bool
    selector_id: str


TRACK_CONFIGS: dict[str, TrackConfig] = {
    "math500": TrackConfig(
        track_id="math500",
        benchmark="MATH500",
        dataset_loader=load_math500_slice,
        optimizer_dev_loader=load_math_optimizer_dev_slice,
        verifier=score_math,
        verifier_id=MATH_VERIFIER_ID,
        default_split=MATH500_TEST_SLICE,
        inference_time_verifier=True,
        selector_id="argmax_verifier_score_iteration_tiebreak-v1",
    ),
    "humaneval_plus": TrackConfig(
        track_id="humaneval_plus",
        benchmark="HumanEval+",
        dataset_loader=load_humaneval_plus_slice,
        optimizer_dev_loader=load_code_optimizer_dev_slice,
        verifier=score_code,
        verifier_id=CODE_VERIFIER_ID,
        default_split=HUMANEVAL_PLUS_TEST_SLICE,
        inference_time_verifier=True,
        selector_id="argmax_verifier_score_iteration_tiebreak-v1",
    ),
    "gpqa_diamond": TrackConfig(
        track_id="gpqa_diamond",
        benchmark="GPQA-Diamond",
        dataset_loader=load_gpqa_diamond_slice,
        optimizer_dev_loader=load_gpqa_non_diamond_slice,
        verifier=score_gpqa_oracle,
        verifier_id=ORACLE_VERIFIER_ID,
        default_split=GPQA_DIAMOND_TEST_SLICE,
        inference_time_verifier=False,
        selector_id=NON_ORACLE_SELECTOR_ID,
    ),
    "reasoning_gym_boxnet": TrackConfig(
        track_id="reasoning_gym_boxnet",
        benchmark="Reasoning Gym boxnet",
        dataset_loader=lambda start, end: load_reasoning_gym_slice("boxnet", start, end),
        optimizer_dev_loader=lambda start, end: load_reasoning_gym_slice("boxnet", start, end),
        verifier=score_reasoning_gym,
        verifier_id=RG_VERIFIER_ID,
        default_split=(0, 100),
        inference_time_verifier=True,
        selector_id="reasoning_gym_score_answer-v1",
    ),
    "reasoning_gym_graph_color": TrackConfig(
        track_id="reasoning_gym_graph_color",
        benchmark="Reasoning Gym graph_color",
        dataset_loader=lambda start, end: load_reasoning_gym_slice("graph_color", start, end),
        optimizer_dev_loader=lambda start, end: load_reasoning_gym_slice("graph_color", start, end),
        verifier=score_reasoning_gym,
        verifier_id=RG_VERIFIER_ID,
        default_split=(0, 100),
        inference_time_verifier=True,
        selector_id="reasoning_gym_score_answer-v1",
    ),
    "reasoning_gym_family_relationships": TrackConfig(
        track_id="reasoning_gym_family_relationships",
        benchmark="Reasoning Gym family_relationships",
        dataset_loader=lambda start, end: load_reasoning_gym_slice(
            "family_relationships", start, end
        ),
        optimizer_dev_loader=lambda start, end: load_reasoning_gym_slice(
            "family_relationships", start, end
        ),
        verifier=score_reasoning_gym,
        verifier_id=RG_VERIFIER_ID,
        default_split=(0, 100),
        inference_time_verifier=True,
        selector_id="reasoning_gym_score_answer-v1",
    ),
}


def get_track_config(track: str) -> TrackConfig:
    try:
        return TRACK_CONFIGS[track]
    except KeyError as exc:
        raise ValueError(f"unknown track: {track!r}") from exc


def load_track_slice(track: str, start: int, end: int) -> list[Any]:
    config = get_track_config(track)
    return config.dataset_loader(start, end)


def load_track_optimizer_dev_slice(track: str, start: int, end: int) -> list[Any]:
    config = get_track_config(track)
    return config.optimizer_dev_loader(start, end)


def _gpqa_best_selector(candidates) -> tuple[Any, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        rows.append(
            {
                "__candidate_pos": idx,
                "generation": candidate.generation,
                "lp_norm_sum": 0.0,
            }
        )
    selected = select_gpqa_non_oracle(rows)
    pos = int(selected["__candidate_pos"])
    return candidates[pos], {
        "selector_id": selected["selector_id"],
        "selected_answer": selected["selected_answer"],
        "oracle_used": selected["oracle_used"],
    }


def selector_for_track(track: str) -> BestSelector | None:
    if track == "gpqa_diamond":
        return _gpqa_best_selector
    return None


def run_condition(
    *,
    track: str,
    split: tuple[int, int],
    model_key: str,
    dataset_source_kind: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run one registered track/condition over a concrete split."""
    config = get_track_config(track)
    problems = kwargs.pop("problems", None)
    if problems is None:
        problems = config.dataset_loader(split[0], split[1])
    condition = kwargs.get("condition")
    if condition in PUBLISHED_COMPARATORS:
        return _run_published_ledger_baseline(
            track=track,
            split=split,
            model_key=model_key,
            config=config,
            problems=problems,
            **kwargs,
        )
    return run_condition_bundle(
        track=track,
        benchmark=config.benchmark,
        split=split,
        model_key=model_key,
        verifier_id=config.verifier_id,
        selector_id=config.selector_id,
        best_selector=selector_for_track(track),
        scorer=config.verifier,
        problems=problems,
        dataset_source_kind=dataset_source_kind,
        **kwargs,
    )


def _run_published_ledger_baseline(
    *,
    track: str,
    split: tuple[int, int],
    model_key: str,
    config: TrackConfig,
    problems: list[Any],
    out_dir: Path,
    condition: str,
    archive_hash: str,
    polaris_source_hash: str,
    vendored_commits: dict[str, str],
    preregistration_anchor: str,
    seed: int,
    run_stage: str = "smoke",
    run_plan_cell: dict[str, Any] | None = None,
    preflight_report: dict[str, Any] | None = None,
    model_id: str = "",
    model_revision: str | None = None,
    model_revision_commit: str | None = None,
    model_artifact_etags: dict[str, str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    comparator = PUBLISHED_COMPARATORS[condition]
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = RolloutLedger(archive_construction=comparator.training_rollouts)
    ledger.charge_inference(
        condition,
        len(problems) * comparator.inference_rollouts_per_query,
    )
    metrics = {
        "track": track,
        "model_key": model_key,
        "model_revision": model_revision,
        "model_revision_commit": model_revision_commit,
        "model_artifact_etags": dict(model_artifact_etags or {}),
        "run_stage": run_stage,
        "condition": condition,
        "accuracy": None,
        "n_problems": len(problems),
        "n_candidates": 0,
        "requires_external_score": True,
        "comparator": comparator.to_jsonable(),
    }
    costs = {
        "wall_clock_seconds": 0.0,
        "estimated_dollar_cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "rollout_total": ledger.total,
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "costs.json", costs)
    write_json(out_dir / "archive.json", [])
    write_jsonl(out_dir / "candidates.jsonl", [])
    write_jsonl(out_dir / "scores.jsonl", [])
    write_jsonl(out_dir / "selected.jsonl", [])
    ledger.write(out_dir / "rollouts.json")
    write_json(out_dir / "environment.json", {"ledger_only": True})
    preflight_payload = dict(preflight_report or {})
    preflight_payload.setdefault("required", run_stage != "smoke")
    preflight_payload.setdefault("passed", run_stage == "smoke")
    preflight_payload.setdefault("protocol_sync_passed", False)
    write_json(out_dir / "preflight.json", preflight_payload)
    cell_payload = run_plan_cell or {
        "track": track,
        "model_key": model_key,
        "split": list(split),
        "condition": condition,
        "run_stage": run_stage,
        "budget": comparator.inference_rollouts_per_query,
        "artifact_dir": str(out_dir),
    }
    write_json(out_dir / "run_plan_cell.json", cell_payload)
    (out_dir / "audit.md").write_text(
        f"# audit\n\n- track: {track}\n- condition: {condition}\n"
        f"- ledger_only: true\n- rollout_total: {ledger.total}\n",
        encoding="utf-8",
    )
    write_run_manifest(
        path=out_dir / "manifest.json",
        model_id=model_id,
        model_revision=model_revision,
        model_revision_commit=model_revision_commit,
        benchmark=config.benchmark,
        split=split,
        seeds=[seed],
        condition=condition,
        archive_hash=archive_hash,
        alpha_policy_id="ledger_only",
        config={
            "track": track,
            "model_key": model_key,
            "run_stage": run_stage,
            "model_revision": model_revision,
            "model_revision_commit": model_revision_commit,
            "model_artifact_etags": dict(model_artifact_etags or {}),
            "dataset_source_kind": "external_ledger",
            "selector_id": "ledger_only",
            "production_artifacts_schema": 1,
        },
        polaris_source_hash=polaris_source_hash,
        vendored_commits=vendored_commits,
        verifier_id=config.verifier_id,
        preregistration_anchor=preregistration_anchor,
    )
    return metrics
