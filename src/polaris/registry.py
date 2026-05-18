from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from polaris.config import MODEL_REGISTRY as RAW_MODEL_REGISTRY
from polaris.config import TRACK_REGISTRY as RAW_TRACK_REGISTRY


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    torch_dtype: str
    attn_impl: str
    default_tracks: tuple[str, ...]
    revision: str | None = None
    revision_commit: str | None = None
    artifact_etags: dict[str, str] | None = None


@dataclass(frozen=True)
class TrackSpec:
    key: str
    regime: str
    primary_model: str
    verifier_id: str
    dataset_module: str
    verifier_module: str
    inference_time_verifier: bool


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    compatible_tracks: tuple[str, ...]
    uses_archive: bool
    uses_power_sampling: bool
    uses_memory: bool
    uses_gepa: bool
    selector_policy: str
    production_baseline: bool
    source: str

    def supports_track(self, track: str) -> bool:
        return "*" in self.compatible_tracks or track in self.compatible_tracks


def _build_model_registry() -> dict[str, ModelSpec]:
    return {
        key: ModelSpec(
            key=key,
            hf_id=str(raw["hf_id"]),
            family=str(raw["family"]),
            torch_dtype=str(raw.get("torch_dtype", "bfloat16")),
            attn_impl=str(raw.get("attn_impl", "sdpa")),
            default_tracks=tuple(raw.get("default_tracks", ())),
            revision=raw.get("revision"),
            revision_commit=raw.get("revision_commit"),
            artifact_etags=dict(raw.get("artifact_etags", {})) or None,
        )
        for key, raw in RAW_MODEL_REGISTRY.items()
    }


def _build_track_registry() -> dict[str, TrackSpec]:
    return {
        key: TrackSpec(
            key=key,
            regime=str(raw["regime"]),
            primary_model=str(raw["primary_model"]),
            verifier_id=str(raw["verifier_id"]),
            dataset_module=str(raw["dataset_module"]),
            verifier_module=str(raw["verifier_module"]),
            inference_time_verifier=bool(raw["inference_time_verifier"]),
        )
        for key, raw in RAW_TRACK_REGISTRY.items()
    }


MODEL_REGISTRY: dict[str, ModelSpec] = _build_model_registry()
TRACK_REGISTRY: dict[str, TrackSpec] = _build_track_registry()


CONDITION_REGISTRY: dict[str, ConditionSpec] = {
    "greedy": ConditionSpec(
        key="greedy",
        label="Greedy",
        compatible_tracks=("*",),
        uses_archive=False,
        uses_power_sampling=False,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="single_candidate",
        production_baseline=True,
        source="local",
    ),
    "bon_temp1": ConditionSpec(
        key="bon_temp1",
        label="Best-of-N temperature 1",
        compatible_tracks=("*",),
        uses_archive=False,
        uses_power_sampling=False,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=True,
        source="local",
    ),
    "bon_temp1_archive": ConditionSpec(
        key="bon_temp1_archive",
        label="Archive Best-of-N temperature 1",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=False,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=True,
        source="local",
    ),
    "single_prompt_power": ConditionSpec(
        key="single_prompt_power",
        label="Single-prompt power sampling",
        compatible_tracks=("*",),
        uses_archive=False,
        uses_power_sampling=True,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=True,
        source="RWS/SPS",
    ),
    "single_best_prompt": ConditionSpec(
        key="single_best_prompt",
        label="Single best prompt",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=True,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=True,
        source="local",
    ),
    "full_archive_fixed": ConditionSpec(
        key="full_archive_fixed",
        label="Full archive fixed alpha",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=True,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=False,
        source="POLARIS",
    ),
    "full_archive_mixed": ConditionSpec(
        key="full_archive_mixed",
        label="Full archive mixed alpha",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=True,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=False,
        source="POLARIS",
    ),
    "full_archive_decaying": ConditionSpec(
        key="full_archive_decaying",
        label="Full archive decaying alpha",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=True,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=False,
        source="POLARIS",
    ),
    "polaris_full_verified_memory": ConditionSpec(
        key="polaris_full_verified_memory",
        label="POLARIS full verified memory",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=True,
        uses_memory=True,
        uses_gepa=True,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=False,
        source="POLARIS",
    ),
    "proicl_gepa_mcmc": ConditionSpec(
        key="proicl_gepa_mcmc",
        label="ProICL GEPA archive + fixed-alpha MCMC",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=True,
        uses_memory=False,
        uses_gepa=True,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=False,
        source="ProICL",
    ),
    "proicl_gepa_mcmc_memory": ConditionSpec(
        key="proicl_gepa_mcmc_memory",
        label="ProICL GEPA archive + fixed-alpha MCMC + verified memory",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=True,
        uses_memory=True,
        uses_gepa=True,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=False,
        source="ProICL",
    ),
    "dynamic_cheatsheet": ConditionSpec(
        key="dynamic_cheatsheet",
        label="Dynamic Cheatsheet",
        compatible_tracks=("*",),
        uses_archive=False,
        uses_power_sampling=False,
        uses_memory=True,
        uses_gepa=False,
        selector_policy="baseline_native",
        production_baseline=True,
        source="vendored/dc",
    ),
    "ace": ConditionSpec(
        key="ace",
        label="ACE playbook",
        compatible_tracks=("*",),
        uses_archive=False,
        uses_power_sampling=False,
        uses_memory=True,
        uses_gepa=False,
        selector_policy="baseline_native",
        production_baseline=True,
        source="vendored/ace",
    ),
    "gepa_only": ConditionSpec(
        key="gepa_only",
        label="GEPA-only prompt archive",
        compatible_tracks=("*",),
        uses_archive=True,
        uses_power_sampling=False,
        uses_memory=False,
        uses_gepa=True,
        selector_policy="argmax_verifier_or_non_oracle_track_selector",
        production_baseline=True,
        source="vendored/gepa",
    ),
    "published_p2o": ConditionSpec(
        key="published_p2o",
        label="Published P2O ledger",
        compatible_tracks=("*",),
        uses_archive=False,
        uses_power_sampling=False,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="ledger_only",
        production_baseline=True,
        source="protocol_ledger",
    ),
    "published_grpo_ledger": ConditionSpec(
        key="published_grpo_ledger",
        label="Published GRPO ledger",
        compatible_tracks=("*",),
        uses_archive=False,
        uses_power_sampling=False,
        uses_memory=False,
        uses_gepa=False,
        selector_policy="ledger_only",
        production_baseline=True,
        source="protocol_ledger",
    ),
}


def resolve_model(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unknown model key: {key!r}") from exc


def resolve_track(key: str) -> TrackSpec:
    try:
        return TRACK_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unknown track: {key!r}") from exc


def resolve_condition(key: str) -> ConditionSpec:
    try:
        return CONDITION_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unknown condition: {key!r}") from exc


def validate_model_for_track(model_key: str, track: str) -> None:
    model = resolve_model(model_key)
    track_spec = resolve_track(track)
    if track_spec.primary_model == model_key:
        return
    if track not in model.default_tracks:
        allowed = ", ".join(model.default_tracks) or "(none)"
        raise ValueError(
            f"model {model_key!r} is not registered for track {track!r}; "
            f"default_tracks={allowed}"
        )


def validate_condition_for_track(condition_key: str, track: str) -> None:
    condition = resolve_condition(condition_key)
    resolve_track(track)
    if not condition.supports_track(track):
        raise ValueError(f"condition {condition_key!r} is not compatible with {track!r}")


def validate_registry(
    *,
    model_keys: Iterable[str] | None = None,
    track_keys: Iterable[str] | None = None,
    condition_keys: Iterable[str] | None = None,
) -> None:
    for key in model_keys or MODEL_REGISTRY:
        resolve_model(key)
    for key in track_keys or TRACK_REGISTRY:
        track = resolve_track(key)
        resolve_model(track.primary_model)
    for key in condition_keys or CONDITION_REGISTRY:
        resolve_condition(key)
