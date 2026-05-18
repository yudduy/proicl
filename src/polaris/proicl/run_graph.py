from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from polaris.prorl_recovery.protocol import BASE_MODEL_KEY, PRORL_V2_MODEL_KEY


PROICL_PRIMARY_TRACKS: tuple[str, ...] = (
    "math500",
    "gpqa_diamond",
    "reasoning_gym_boxnet",
    "reasoning_gym_graph_color",
    "reasoning_gym_family_relationships",
)

PROICL_PRIMARY_CONDITIONS: tuple[str, ...] = (
    "base_greedy",
    "mcmc_only",
    "gepa_only",
    "gepa_mcmc",
    "gepa_mcmc_memory",
    "prorl_v2_greedy",
)


@dataclass(frozen=True)
class ProICLCell:
    proicl_condition: str
    runtime_condition: str
    condition: str
    track: str
    model_key: str
    split: tuple[int, int]
    shard_id: int
    num_shards: int
    rollout_budget: int
    archive_kind: str
    archive_path: str
    archive_build_id: str
    memory_build_id: str
    memory_store_path: str | None
    cache_path: str
    artifact_dir: str
    uses_power_sampling: bool
    uses_gepa_archive: bool
    uses_memory: bool
    slow_weight_reference: bool
    memory_mode: str = "off"
    alpha_policy: str = "fixed_alpha_4"
    seed: int = 17

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = list(self.split)
        return payload


def _logical_direct_archive(track: str) -> str:
    return "direct"


def _runtime_archive_path(root: str, archive_kind: str, track: str) -> str:
    if archive_kind == "proicl_cross_task_gepa":
        return f"{root}/archives/proicl_cross_task_gepa/archive.json"
    return f"{root}/archives/{track}/direct.json"


def _cell_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "proicl_condition": "base_greedy",
            "runtime_condition": "greedy",
            "model_key": BASE_MODEL_KEY,
            "archive_kind": "direct",
            "budget_role": "greedy",
            "uses_power_sampling": False,
            "uses_gepa_archive": False,
            "uses_memory": False,
            "slow_weight_reference": False,
            "alpha_policy": "greedy",
        },
        {
            "proicl_condition": "mcmc_only",
            "runtime_condition": "single_prompt_power",
            "model_key": BASE_MODEL_KEY,
            "archive_kind": "direct",
            "budget_role": "matched",
            "uses_power_sampling": True,
            "uses_gepa_archive": False,
            "uses_memory": False,
            "slow_weight_reference": False,
        },
        {
            "proicl_condition": "gepa_only",
            "runtime_condition": "gepa_only",
            "model_key": BASE_MODEL_KEY,
            "archive_kind": "proicl_cross_task_gepa",
            "budget_role": "matched",
            "uses_power_sampling": False,
            "uses_gepa_archive": True,
            "uses_memory": False,
            "slow_weight_reference": False,
            "alpha_policy": "bon_temp1",
        },
        {
            "proicl_condition": "gepa_mcmc",
            "runtime_condition": "proicl_gepa_mcmc",
            "model_key": BASE_MODEL_KEY,
            "archive_kind": "proicl_cross_task_gepa",
            "budget_role": "matched",
            "uses_power_sampling": True,
            "uses_gepa_archive": True,
            "uses_memory": False,
            "slow_weight_reference": False,
        },
        {
            "proicl_condition": "gepa_mcmc_memory",
            "runtime_condition": "proicl_gepa_mcmc_memory",
            "model_key": BASE_MODEL_KEY,
            "archive_kind": "proicl_cross_task_gepa",
            "budget_role": "matched",
            "uses_power_sampling": True,
            "uses_gepa_archive": True,
            "uses_memory": True,
            "slow_weight_reference": False,
            "memory_mode": "distilled_strategies",
        },
        {
            "proicl_condition": "prorl_v2_greedy",
            "runtime_condition": "greedy",
            "model_key": PRORL_V2_MODEL_KEY,
            "archive_kind": "direct",
            "budget_role": "greedy",
            "uses_power_sampling": False,
            "uses_gepa_archive": False,
            "uses_memory": False,
            "slow_weight_reference": True,
            "alpha_policy": "greedy",
        },
    )


def build_proicl_run_graph(
    *,
    root: str,
    tracks: Iterable[str] = PROICL_PRIMARY_TRACKS,
    problem_count: int = 100,
    rollout_budget: int = 128,
    num_shards: int = 4,
    seed: int = 17,
) -> dict[str, Any]:
    if problem_count <= 0:
        raise ValueError("problem_count must be positive")
    if rollout_budget <= 0:
        raise ValueError("rollout_budget must be positive")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    cells: list[ProICLCell] = []
    for track in tracks:
        for shard_id in range(num_shards):
            for spec in _cell_specs():
                archive_kind = str(spec["archive_kind"])
                logical_archive = (
                    _logical_direct_archive(track)
                    if archive_kind == "direct"
                    else archive_kind
                )
                budget = 1 if spec["budget_role"] == "greedy" else rollout_budget
                proicl_condition = str(spec["proicl_condition"])
                cells.append(
                    ProICLCell(
                        proicl_condition=proicl_condition,
                        runtime_condition=str(spec["runtime_condition"]),
                        condition=str(spec["runtime_condition"]),
                        track=track,
                        model_key=str(spec["model_key"]),
                        split=(0, problem_count),
                        shard_id=shard_id,
                        num_shards=num_shards,
                        rollout_budget=budget,
                        archive_kind=logical_archive,
                        archive_path=_runtime_archive_path(root, archive_kind, track),
                        archive_build_id=(
                            "proicl-cross-task-gepa"
                            if bool(spec["uses_gepa_archive"])
                            else "none"
                        ),
                        memory_build_id=(
                            f"proicl-memory-{track}"
                            if bool(spec["uses_memory"])
                            else "none"
                        ),
                        memory_store_path=(
                            f"{root}/memory/{track}-{proicl_condition}-"
                            f"shard-{shard_id}.sqlite"
                            if bool(spec["uses_memory"])
                            else None
                        ),
                        cache_path=(
                            f"{root}/trajectory_cache/{track}-"
                            f"{proicl_condition}-shard-{shard_id}.sqlite"
                        ),
                        artifact_dir=(
                            f"{root}/runs/{track}/{proicl_condition}/"
                            f"shard-{shard_id}"
                        ),
                        uses_power_sampling=bool(spec["uses_power_sampling"]),
                        uses_gepa_archive=bool(spec["uses_gepa_archive"]),
                        uses_memory=bool(spec["uses_memory"]),
                        slow_weight_reference=bool(spec["slow_weight_reference"]),
                        memory_mode=str(spec.get("memory_mode", "off")),
                        alpha_policy=str(spec.get("alpha_policy", "fixed_alpha_4")),
                        seed=seed,
                    )
                )

    return {
        "schema_version": 1,
        "experiment": "proicl_fast_weight_recovery_audit",
        "root": root,
        "tracks": list(tracks),
        "primary_conditions": list(PROICL_PRIMARY_CONDITIONS),
        "policy": {
            "base_model_frozen_for_fast_weight_cells": True,
            "no_prorl_or_brorl_traces_in_gepa_or_memory": True,
            "matched_rollout_budget_for_frozen_cells": rollout_budget,
            "prorl_v2_greedy_is_slow_weight_reference": True,
        },
        "cells": [cell.to_jsonable() for cell in cells],
    }


def write_proicl_run_graph(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
