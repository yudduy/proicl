from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from polaris.infra.preflight import RunKind
from polaris.io.artifact_audit import PRODUCTION_ARTIFACTS, SEVEN_ARTIFACTS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESOURCE_PROFILE_PATH = REPO_ROOT / "configs" / "prorl_live_resources.json"
REQUIRED_PRODUCTION_ARTIFACTS = SEVEN_ARTIFACTS + PRODUCTION_ARTIFACTS


@dataclass(frozen=True)
class ResourceProfile:
    key: str
    label: str
    run_kind: RunKind
    role: str
    gpu_model: str
    gpu_count: int
    max_concurrent_jobs: int
    allowed_phases: tuple[str, ...]
    free: bool
    fallback_only: bool
    initial_spend_cap_dollars: float
    hourly_rate_dollars: float | None = None
    max_bid_dollars_per_gpu_hour: float | None = None
    notes: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_phases"] = list(self.allowed_phases)
        return payload


def _profile_from_dict(payload: dict[str, Any]) -> ResourceProfile:
    return ResourceProfile(
        key=str(payload["key"]),
        label=str(payload["label"]),
        run_kind=payload["run_kind"],
        role=str(payload["role"]),
        gpu_model=str(payload["gpu_model"]),
        gpu_count=int(payload["gpu_count"]),
        max_concurrent_jobs=int(payload["max_concurrent_jobs"]),
        allowed_phases=tuple(str(item) for item in payload["allowed_phases"]),
        free=bool(payload["free"]),
        fallback_only=bool(payload.get("fallback_only", False)),
        initial_spend_cap_dollars=float(payload["initial_spend_cap_dollars"]),
        hourly_rate_dollars=(
            float(payload["hourly_rate_dollars"])
            if payload.get("hourly_rate_dollars") is not None
            else None
        ),
        max_bid_dollars_per_gpu_hour=(
            float(payload["max_bid_dollars_per_gpu_hour"])
            if payload.get("max_bid_dollars_per_gpu_hour") is not None
            else None
        ),
        notes=str(payload.get("notes", "")),
    )


def validate_resource_profile(profile: ResourceProfile) -> None:
    if not profile.key:
        raise ValueError("profile key is required")
    if profile.gpu_count <= 0:
        raise ValueError(f"{profile.key}: gpu_count must be positive")
    if profile.max_concurrent_jobs <= 0:
        raise ValueError(f"{profile.key}: max_concurrent_jobs must be positive")
    if not profile.allowed_phases:
        raise ValueError(f"{profile.key}: allowed_phases is required")
    if profile.initial_spend_cap_dollars < 0:
        raise ValueError(f"{profile.key}: initial_spend_cap_dollars must be non-negative")
    if profile.hourly_rate_dollars is not None and profile.hourly_rate_dollars < 0:
        raise ValueError(f"{profile.key}: hourly_rate_dollars must be non-negative")
    if (
        profile.max_bid_dollars_per_gpu_hour is not None
        and profile.max_bid_dollars_per_gpu_hour < 0
    ):
        raise ValueError(f"{profile.key}: max_bid_dollars_per_gpu_hour must be non-negative")
    if profile.free and profile.initial_spend_cap_dollars != 0.0:
        raise ValueError(f"{profile.key}: free profiles must have zero spend cap")
    if profile.free and profile.run_kind != "farmshare":
        raise ValueError(f"{profile.key}: only FarmShare is marked free")


def load_resource_profiles(
    path: Path = DEFAULT_RESOURCE_PROFILE_PATH,
) -> dict[str, ResourceProfile]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles: dict[str, ResourceProfile] = {}
    for raw in payload.get("profiles", []):
        profile = _profile_from_dict(raw)
        if profile.key in profiles:
            raise ValueError(f"duplicate resource profile: {profile.key}")
        validate_resource_profile(profile)
        profiles[profile.key] = profile
    if not profiles:
        raise ValueError(f"no resource profiles found in {path}")
    return profiles


def get_resource_profile(
    key: str,
    *,
    path: Path = DEFAULT_RESOURCE_PROFILE_PATH,
) -> ResourceProfile:
    profiles = load_resource_profiles(path)
    try:
        return profiles[key]
    except KeyError as exc:
        raise ValueError(f"unknown resource profile: {key}") from exc


def validate_profile_cost(
    profile: ResourceProfile,
    *,
    estimated_dollar_cost: float,
) -> dict[str, Any]:
    if estimated_dollar_cost < 0:
        raise ValueError("estimated_dollar_cost must be non-negative")
    if profile.free and estimated_dollar_cost != 0.0:
        raise ValueError("FarmShare profile must remain zero-cost")
    if estimated_dollar_cost > profile.initial_spend_cap_dollars:
        raise ValueError(
            "estimated_dollar_cost exceeds profile initial_spend_cap_dollars "
            f"({estimated_dollar_cost} > {profile.initial_spend_cap_dollars})"
        )
    return {
        "passed": True,
        "profile": profile.key,
        "estimated_dollar_cost": estimated_dollar_cost,
        "initial_spend_cap_dollars": profile.initial_spend_cap_dollars,
    }


def artifact_contract() -> tuple[str, ...]:
    return REQUIRED_PRODUCTION_ARTIFACTS
