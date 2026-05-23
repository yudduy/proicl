from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from polaris.prorl_recovery.protocol import BASE_MODEL_KEY, PRORL_V2_MODEL_KEY
from polaris.proicl.protocol import (
    ArchiveScope,
    ForkSearchMode,
    MemoryProtocol,
    PAPER_ALIGNED_SUSTAINED_TRACKS,
    ProICLConditionSpec,
    RepairMode,
    archive_scope_id,
    validate_archive_scope_membership,
)


PROICL_PRIMARY_TRACKS: tuple[str, ...] = PAPER_ALIGNED_SUSTAINED_TRACKS

PROICL_PRIMARY_CONDITIONS: tuple[str, ...] = (
    "base_greedy",
    "bon_temp1",
    "mcmc_only",
    "mixed_alpha_mcmc",
    "fork_search",
    "gepa_only",
    "gepa_mcmc",
    "gepa_mcmc_repair",
    "gepa_mcmc_fork_repair",
    "gepa_mcmc_fork_repair_memory",
    "gepa_mcmc_memory",
    "prorl_v2_greedy",
)

SPS_RECOVERY_CONDITIONS: tuple[str, ...] = (
    "base_greedy",
    "sps_only",
    "gepa_sps_fixed",
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
    archive_scope: str
    archive_scope_id: str
    archive_train_tracks: tuple[str, ...]
    archive_heldout_tracks: tuple[str, ...]
    memory_build_id: str
    memory_store_path: str | None
    cache_path: str
    artifact_dir: str
    uses_power_sampling: bool
    uses_gepa_archive: bool
    uses_memory: bool
    uses_repair: bool
    uses_fork_search: bool
    slow_weight_reference: bool
    memory_mode: str = "off"
    memory_protocol: str = "off"
    repair_mode: str = "off"
    fork_search_mode: str = "off"
    alpha_policy: str = "fixed_alpha_4"
    seed: int = 17

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = list(self.split)
        payload["archive_train_tracks"] = list(self.archive_train_tracks)
        payload["archive_heldout_tracks"] = list(self.archive_heldout_tracks)
        return payload


def _logical_direct_archive(track: str) -> str:
    return "direct"


def _runtime_archive_path(
    root: str,
    archive_kind: str,
    track: str,
    *,
    scope_id: str,
) -> str:
    if archive_kind == "gepa":
        return f"{root}/archives/{scope_id}/archive.json"
    return f"{root}/archives/{track}/direct.json"


def _condition_specs() -> tuple[ProICLConditionSpec, ...]:
    return (
        ProICLConditionSpec(
            key="base_greedy",
            runtime_condition="greedy",
            model_key=BASE_MODEL_KEY,
            archive_kind="direct",
            budget_role="greedy",
            alpha_policy="greedy",
        ),
        ProICLConditionSpec(
            key="bon_temp1",
            runtime_condition="bon_temp1",
            model_key=BASE_MODEL_KEY,
            archive_kind="direct",
            budget_role="matched",
            alpha_policy="bon_temp1",
        ),
        ProICLConditionSpec(
            key="sps_only",
            runtime_condition="single_prompt_power",
            model_key=BASE_MODEL_KEY,
            archive_kind="direct",
            budget_role="matched",
            alpha_policy="fixed_alpha_4",
            uses_power_sampling=True,
        ),
        ProICLConditionSpec(
            key="mcmc_only",
            runtime_condition="single_prompt_power",
            model_key=BASE_MODEL_KEY,
            archive_kind="direct",
            budget_role="matched",
            alpha_policy="fixed_alpha_4",
            uses_power_sampling=True,
        ),
        ProICLConditionSpec(
            key="mixed_alpha_mcmc",
            runtime_condition="mixed_alpha_mcmc",
            model_key=BASE_MODEL_KEY,
            archive_kind="direct",
            budget_role="matched",
            alpha_policy="mixed_alpha_4_1",
            uses_power_sampling=True,
        ),
        ProICLConditionSpec(
            key="fork_search",
            runtime_condition="fork_search",
            model_key=BASE_MODEL_KEY,
            archive_kind="direct",
            budget_role="matched",
            alpha_policy="bon_temp1",
            uses_fork_search=True,
            fork_search_mode=ForkSearchMode.ENTROPY_GATED,
        ),
        ProICLConditionSpec(
            key="gepa_only",
            runtime_condition="gepa_only",
            model_key=BASE_MODEL_KEY,
            archive_kind="gepa",
            budget_role="matched",
            alpha_policy="bon_temp1",
            uses_gepa_archive=True,
        ),
        ProICLConditionSpec(
            key="gepa_sps_fixed",
            runtime_condition="full_archive_fixed",
            model_key=BASE_MODEL_KEY,
            archive_kind="gepa",
            budget_role="matched",
            alpha_policy="fixed_alpha_4",
            uses_power_sampling=True,
            uses_gepa_archive=True,
        ),
        ProICLConditionSpec(
            key="gepa_mcmc",
            runtime_condition="proicl_gepa_mcmc",
            model_key=BASE_MODEL_KEY,
            archive_kind="gepa",
            budget_role="matched",
            alpha_policy="mixed_alpha_4_1",
            uses_power_sampling=True,
            uses_gepa_archive=True,
        ),
        ProICLConditionSpec(
            key="gepa_mcmc_repair",
            runtime_condition="proicl_gepa_mcmc_repair",
            model_key=BASE_MODEL_KEY,
            archive_kind="gepa",
            budget_role="matched",
            alpha_policy="mixed_alpha_4_1",
            uses_power_sampling=True,
            uses_gepa_archive=True,
            uses_repair=True,
            repair_mode=RepairMode.VERIFIER_GUIDED,
        ),
        ProICLConditionSpec(
            key="gepa_mcmc_fork_repair",
            runtime_condition="proicl_gepa_mcmc_fork_repair",
            model_key=BASE_MODEL_KEY,
            archive_kind="gepa",
            budget_role="matched",
            alpha_policy="mixed_alpha_4_1",
            uses_power_sampling=True,
            uses_gepa_archive=True,
            uses_repair=True,
            uses_fork_search=True,
            repair_mode=RepairMode.VERIFIER_GUIDED,
            fork_search_mode=ForkSearchMode.ENTROPY_GATED,
        ),
        ProICLConditionSpec(
            key="gepa_mcmc_fork_repair_memory",
            runtime_condition="proicl_gepa_mcmc_fork_repair_memory",
            model_key=BASE_MODEL_KEY,
            archive_kind="gepa",
            budget_role="matched",
            alpha_policy="mixed_alpha_4_1",
            uses_power_sampling=True,
            uses_gepa_archive=True,
            uses_repair=True,
            uses_fork_search=True,
            uses_memory=True,
            memory_mode="distilled_strategies",
            memory_protocol=MemoryProtocol.FROZEN_DEV,
            repair_mode=RepairMode.VERIFIER_GUIDED,
            fork_search_mode=ForkSearchMode.ENTROPY_GATED,
        ),
        ProICLConditionSpec(
            key="gepa_mcmc_memory",
            runtime_condition="proicl_gepa_mcmc_memory",
            model_key=BASE_MODEL_KEY,
            archive_kind="gepa",
            budget_role="matched",
            alpha_policy="mixed_alpha_4_1",
            uses_power_sampling=True,
            uses_gepa_archive=True,
            uses_memory=True,
            memory_mode="distilled_strategies",
            memory_protocol=MemoryProtocol.ONLINE,
            preliminary=True,
        ),
        ProICLConditionSpec(
            key="prorl_v2_greedy",
            runtime_condition="greedy",
            model_key=PRORL_V2_MODEL_KEY,
            archive_kind="direct",
            budget_role="greedy",
            alpha_policy="greedy",
            slow_weight_reference=True,
        ),
    )


def build_proicl_run_graph(
    *,
    root: str,
    tracks: Iterable[str] = PROICL_PRIMARY_TRACKS,
    problem_count: int = 100,
    rollout_budget: int = 128,
    num_shards: int = 4,
    seed: int = 17,
    archive_scope: ArchiveScope | str = ArchiveScope.WITHIN_FAMILY,
    archive_train_tracks: Iterable[str] | None = None,
    archive_heldout_tracks: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
) -> dict[str, Any]:
    if problem_count <= 0:
        raise ValueError("problem_count must be positive")
    if rollout_budget <= 0:
        raise ValueError("rollout_budget must be positive")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    track_tuple = tuple(tracks)
    condition_tuple = tuple(conditions or PROICL_PRIMARY_CONDITIONS)
    archive_scope = ArchiveScope(archive_scope)
    train_tracks = tuple(archive_train_tracks or track_tuple)
    heldout_tracks = tuple(archive_heldout_tracks or track_tuple)
    validate_archive_scope_membership(
        archive_scope=archive_scope,
        train_tracks=train_tracks,
        heldout_tracks=heldout_tracks,
    )
    scope_id = archive_scope_id(archive_scope, train_tracks)
    spec_by_key = {spec.key: spec for spec in _condition_specs()}
    unknown_conditions = sorted(set(condition_tuple) - set(spec_by_key))
    if unknown_conditions:
        raise ValueError("unknown ProICL conditions: " + ", ".join(unknown_conditions))

    cells: list[ProICLCell] = []
    for track in track_tuple:
        for shard_id in range(num_shards):
            for proicl_condition in condition_tuple:
                spec = spec_by_key[proicl_condition]
                archive_kind = spec.archive_kind
                logical_archive = (
                    _logical_direct_archive(track)
                    if archive_kind == "direct"
                    else archive_kind
                )
                budget = 1 if spec.budget_role == "greedy" else rollout_budget
                cells.append(
                    ProICLCell(
                        proicl_condition=proicl_condition,
                        runtime_condition=spec.runtime_condition,
                        condition=spec.runtime_condition,
                        track=track,
                        model_key=spec.model_key,
                        split=(0, problem_count),
                        shard_id=shard_id,
                        num_shards=num_shards,
                        rollout_budget=budget,
                        archive_kind=logical_archive,
                        archive_path=_runtime_archive_path(
                            root,
                            archive_kind,
                            track,
                            scope_id=scope_id,
                        ),
                        archive_build_id=(
                            scope_id
                            if spec.uses_gepa_archive
                            else "none"
                        ),
                        archive_scope=archive_scope.value,
                        archive_scope_id=scope_id,
                        archive_train_tracks=train_tracks if spec.uses_gepa_archive else (),
                        archive_heldout_tracks=heldout_tracks if spec.uses_gepa_archive else (),
                        memory_build_id=(
                            f"proicl-memory-{spec.memory_protocol.value}-{track}"
                            if spec.uses_memory
                            else "none"
                        ),
                        memory_store_path=(
                            f"{root}/memory/{track}-{proicl_condition}-"
                            f"shard-{shard_id}.sqlite"
                            if spec.uses_memory
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
                        uses_power_sampling=spec.uses_power_sampling,
                        uses_gepa_archive=spec.uses_gepa_archive,
                        uses_memory=spec.uses_memory,
                        uses_repair=spec.uses_repair,
                        uses_fork_search=spec.uses_fork_search,
                        slow_weight_reference=spec.slow_weight_reference,
                        memory_mode=spec.memory_mode,
                        memory_protocol=spec.memory_protocol.value,
                        repair_mode=spec.repair_mode.value,
                        fork_search_mode=spec.fork_search_mode.value,
                        alpha_policy=spec.alpha_policy,
                        seed=seed,
                    )
                )

    return {
        "schema_version": 1,
        "experiment": "proicl_fast_weight_recovery_audit",
        "root": root,
        "tracks": list(track_tuple),
        "primary_conditions": list(condition_tuple),
        "archive_scope": archive_scope.value,
        "archive_scope_id": scope_id,
        "archive_train_tracks": list(train_tracks),
        "archive_heldout_tracks": list(heldout_tracks),
        "policy": {
            "base_model_frozen_for_fast_weight_cells": True,
            "no_prorl_or_brorl_traces_in_gepa_or_memory": True,
            "matched_rollout_budget_for_frozen_cells": rollout_budget,
            "prorl_v2_greedy_is_slow_weight_reference": True,
            "full_proicl_condition": "gepa_mcmc_fork_repair_memory",
        },
        "cells": [cell.to_jsonable() for cell in cells],
    }


def write_proicl_run_graph(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
