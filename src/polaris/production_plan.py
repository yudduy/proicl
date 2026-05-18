from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from polaris.registry import (
    CONDITION_REGISTRY,
    MODEL_REGISTRY,
    TRACK_REGISTRY,
    validate_condition_for_track,
    validate_model_for_track,
)


DEFAULT_SMALL_SLICE: dict[str, tuple[int, int]] = {
    "math500": (0, 5),
    "humaneval_plus": (0, 5),
    "gpqa_diamond": (0, 5),
}

DEFAULT_FINAL_SPLIT: dict[str, tuple[int, int]] = {
    "math500": (0, 500),
    "humaneval_plus": (0, 164),
    "gpqa_diamond": (0, 198),
}

DEFAULT_COST_CAP: dict[str, float] = {
    "smoke": 0.0,
    "small_real_slice": 25.0,
    "final": 750.0,
}


@dataclass(frozen=True)
class RunCell:
    track: str
    model_key: str
    split_id: str
    split: tuple[int, int]
    condition: str
    archive: str
    archive_build_id: str
    memory_build_id: str
    memory_mode: str
    alpha_policy: str
    seed: int
    budget: int
    cache_path: str
    artifact_dir: str
    dataset_lock_id: str
    model_revision: str
    cost_cap_dollars: float
    run_stage: str

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = list(self.split)
        return payload


def _budget_for_condition(condition: str, stage: str) -> int:
    if stage == "smoke":
        return 1
    if condition in ("greedy", "dynamic_cheatsheet", "ace"):
        return 1
    if condition == "full_archive_decaying":
        return 16
    return 8


def _alpha_policy_for_condition(condition: str) -> str:
    if condition == "bon_temp1" or condition == "bon_temp1_archive":
        return "bon_temp1"
    if condition in ("full_archive_mixed", "polaris_full_verified_memory"):
        return "mixed_alpha_4_1"
    if condition == "full_archive_decaying":
        return "decaying_alpha_4_to_1"
    return "fixed_alpha_4"


def _memory_mode_for_condition(condition: str) -> str:
    if condition in ("polaris_full_verified_memory", "proicl_gepa_mcmc_memory"):
        return "distilled_strategies"
    condition_spec = CONDITION_REGISTRY[condition]
    return "baseline_native" if condition_spec.uses_memory else "none"


def build_production_plan(
    *,
    stages: tuple[str, ...] = ("small_real_slice", "final"),
    tracks: tuple[str, ...] = ("math500", "humaneval_plus", "gpqa_diamond"),
    conditions: tuple[str, ...] | None = None,
    seed: int = 17,
    root: str = "runs/production",
) -> dict[str, Any]:
    selected_conditions = conditions or tuple(CONDITION_REGISTRY)
    cells: list[RunCell] = []
    for stage in stages:
        if stage not in DEFAULT_COST_CAP:
            raise ValueError(f"unknown run stage: {stage!r}")
        for track in tracks:
            track_spec = TRACK_REGISTRY[track]
            model_key = track_spec.primary_model
            validate_model_for_track(model_key, track)
            model_spec = MODEL_REGISTRY[model_key]
            split_id = "small_real_slice" if stage in ("smoke", "small_real_slice") else "final"
            split = DEFAULT_SMALL_SLICE[track] if split_id == "small_real_slice" else DEFAULT_FINAL_SPLIT[track]
            for condition in selected_conditions:
                condition_spec = CONDITION_REGISTRY[condition]
                if not condition_spec.supports_track(track):
                    continue
                validate_condition_for_track(condition, track)
                cells.append(
                    RunCell(
                        track=track,
                        model_key=model_key,
                        split_id=split_id,
                        split=split,
                        condition=condition,
                        archive=f"archives/{track}/{condition}.json",
                        archive_build_id=f"{track}-{condition}-{split_id}-archive",
                        memory_build_id=(
                            f"{track}-{condition}-{split_id}-memory"
                            if condition_spec.uses_memory
                            else "none"
                        ),
                        memory_mode=_memory_mode_for_condition(condition),
                        alpha_policy=_alpha_policy_for_condition(condition),
                        seed=seed,
                        budget=_budget_for_condition(condition, stage),
                        cache_path=f"{root}/trajectory_cache/{track}.sqlite",
                        artifact_dir=(
                            f"{root}/{stage}/{track}/{model_key}/"
                            f"{condition}/seed-{seed}"
                        ),
                        dataset_lock_id=f"{track}:{split_id}",
                        model_revision=model_spec.revision or "default",
                        cost_cap_dollars=DEFAULT_COST_CAP[stage],
                        run_stage=stage,
                    )
                )
    return {
        "schema_version": 1,
        "policy": {
            "no_paid_run_without_user_authorization": True,
            "no_final_run_without_dataset_lock": True,
            "no_final_run_without_clean_artifact_audit": True,
            "ambient_MODEL_ID_forbidden": True,
        },
        "models": {key: asdict(value) for key, value in MODEL_REGISTRY.items()},
        "tracks": {key: asdict(value) for key, value in TRACK_REGISTRY.items()},
        "conditions": {
            key: asdict(value) for key, value in CONDITION_REGISTRY.items()
        },
        "cells": [cell.to_jsonable() for cell in cells],
    }


def write_production_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
