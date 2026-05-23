from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SAFE_SLUG_RE = re.compile(r"[^a-z0-9]+")


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def slugify(value: str) -> str:
    slug = SAFE_SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "none"


def panel_slug(tracks: Iterable[str]) -> str:
    track_tuple = tuple(tracks)
    if not track_tuple:
        return "no-tracks"
    if len(track_tuple) == 1:
        return slugify(track_tuple[0].replace("reasoning_gym_", ""))
    if set(track_tuple) >= {
        "reasoning_gym_family_relationships",
        "reasoning_gym_boxnet",
    } and any("graph_color" in track for track in track_tuple):
        return f"sustained-rg-{len(track_tuple)}t"
    return f"custom-{len(track_tuple)}t"


@dataclass(frozen=True)
class ProICLRunIdentity:
    run_id: str
    experiment: str
    run_stage: str
    panel: str
    archive_scope: str
    backend: str
    timestamp: str
    tag: str | None = None


def make_proicl_run_identity(
    *,
    run_stage: str,
    tracks: Iterable[str],
    archive_scope: str,
    backend: str,
    timestamp: str | None = None,
    tag: str | None = None,
) -> ProICLRunIdentity:
    stamp = timestamp or utc_stamp()
    panel = panel_slug(tracks)
    parts = [
        "proicl",
        slugify(run_stage),
        panel,
        slugify(archive_scope),
        slugify(backend),
    ]
    if tag:
        parts.append(slugify(tag))
    parts.append(stamp)
    return ProICLRunIdentity(
        run_id="_".join(parts),
        experiment="proicl_sustained_recovery",
        run_stage=run_stage,
        panel=panel,
        archive_scope=archive_scope,
        backend=backend,
        timestamp=stamp,
        tag=tag,
    )


def standard_run_root(base_dir: Path, identity: ProICLRunIdentity) -> Path:
    return base_dir / identity.run_id


def write_run_index(
    path: Path,
    *,
    identity: ProICLRunIdentity,
    tracks: Iterable[str],
    conditions: Iterable[str],
    split: tuple[int, int],
    rollout_budget: int,
    archive_size: int,
    max_metric_calls: int,
    max_new_tokens: int,
    mcmc_steps: int | None,
    mcmc_block_num: int | None,
    num_shards: int,
    memory_num_shards: int,
    reflection_provider: str,
    reflection_model_id: str,
    run_kind: str,
    cost_cap_dollars: float | None,
    power_sampler: str = "mcmc",
    sps_top_k: int | None = None,
    sps_candidate_pool_size: int | None = None,
    sps_rollouts_per_candidate: int | None = None,
    sps_rollout_horizon: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    payload = {
        "identity": asdict(identity),
        "tracks": list(tracks),
        "conditions": list(conditions),
        "split": list(split),
        "rollout_budget": rollout_budget,
        "archive_size": archive_size,
        "max_metric_calls": max_metric_calls,
        "max_new_tokens": max_new_tokens,
        "power_sampler": power_sampler,
        "mcmc": {
            "steps": mcmc_steps,
            "block_num": mcmc_block_num,
        },
        "sps": {
            "top_k": sps_top_k,
            "candidate_pool_size": sps_candidate_pool_size,
            "rollouts_per_candidate": sps_rollouts_per_candidate,
            "rollout_horizon": sps_rollout_horizon,
        },
        "num_shards": num_shards,
        "memory_num_shards": memory_num_shards,
        "reflection": {
            "provider": reflection_provider,
            "model_id": reflection_model_id,
        },
        "run_kind": run_kind,
        "cost_cap_dollars": cost_cap_dollars,
        "notes": notes,
        "layout": {
            "smoke": "smoke/",
            "full": "full/",
            "logs": "logs/",
            "analysis": "full/analysis/",
            "archives": "full/archives/",
            "trajectory_cache": "full/trajectory_cache/",
            "remote_mirror": "remote_mirrors/",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
