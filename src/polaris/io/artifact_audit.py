from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polaris.config import MODEL_ID
from polaris.io.dataset_locks import assert_locked_dataset, read_dataset_locks
from polaris.registry import CONDITION_REGISTRY


SEVEN_ARTIFACTS = (
    "manifest.json",
    "archive.json",
    "candidates.jsonl",
    "scores.jsonl",
    "costs.json",
    "metrics.json",
    "audit.md",
)

PRODUCTION_ARTIFACTS = (
    "rollouts.json",
    "selected.jsonl",
    "preflight.json",
    "environment.json",
    "run_plan_cell.json",
)


class ArtifactAuditError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_run_artifacts(
    artifact_dir: Path,
    *,
    stage: str,
    dataset_lock_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"smoke", "small_real_slice", "final"}:
        raise ValueError(f"unknown audit stage: {stage!r}")

    missing = [name for name in SEVEN_ARTIFACTS if not (artifact_dir / name).exists()]
    if stage != "smoke":
        missing.extend(
            name for name in PRODUCTION_ARTIFACTS if not (artifact_dir / name).exists()
        )
    if missing:
        raise ArtifactAuditError(f"missing artifacts: {', '.join(sorted(missing))}")

    manifest = _read_json(artifact_dir / "manifest.json")
    metrics = _read_json(artifact_dir / "metrics.json")
    costs = _read_json(artifact_dir / "costs.json")
    rollouts = (
        _read_json(artifact_dir / "rollouts.json")
        if (artifact_dir / "rollouts.json").exists()
        else {}
    )
    preflight = (
        _read_json(artifact_dir / "preflight.json")
        if (artifact_dir / "preflight.json").exists()
        else {}
    )
    run_plan_cell = (
        _read_json(artifact_dir / "run_plan_cell.json")
        if (artifact_dir / "run_plan_cell.json").exists()
        else {}
    )
    candidates = _read_jsonl(artifact_dir / "candidates.jsonl")
    selected = _read_jsonl(artifact_dir / "selected.jsonl")

    failures: list[str] = []
    config = manifest.get("config", {})
    track = config.get("track") or metrics.get("track") or run_plan_cell.get("track")
    ledger_only = config.get("dataset_source_kind") == "external_ledger"

    if stage == "final":
        condition = str(manifest.get("condition") or metrics.get("condition") or "")
        condition_spec = CONDITION_REGISTRY.get(condition)
        if not candidates and not ledger_only:
            failures.append("final run has no candidate rows")
        if not selected and not ledger_only:
            failures.append("final run has no selected rows")
        if config.get("dataset_source_kind") == "fixture":
            failures.append("final run is marked as fixture data")
        if any(str(row.get("source", "")).lower() == "smoke" for row in candidates):
            failures.append("final run contains fixture/smoke candidate rows")
        if not rollouts or int(rollouts.get("total", 0)) <= 0:
            failures.append("rollouts.json missing positive total")
        if not preflight.get("passed"):
            failures.append("preflight did not pass")
        if not preflight.get("protocol_sync_passed"):
            failures.append("protocol sync evidence missing")
        if not config.get("model_key"):
            failures.append("manifest is missing explicit model_key")
        if manifest.get("model") == MODEL_ID and not config.get("model_key"):
            failures.append("manifest appears to inherit global MODEL_ID")
        if not run_plan_cell:
            failures.append("run_plan_cell.json is empty")
        if condition_spec is not None and condition_spec.uses_memory:
            has_memory_events = (artifact_dir / "memory_events.jsonl").exists()
            has_memory_sqlite = (artifact_dir / "memory.sqlite").exists()
            if not has_memory_events or not has_memory_sqlite:
                failures.append("memory condition missing memory ledger artifacts")
        if condition_spec is not None and condition_spec.uses_gepa:
            archive_build_id = (
                run_plan_cell.get("archive_build_id")
                or config.get("archive_build_id")
            )
            if not archive_build_id:
                failures.append("GEPA/archive condition missing archive_build_id")
            if not (artifact_dir / "archive_build_manifest.json").exists():
                failures.append("GEPA/archive condition missing archive build provenance")
        if ledger_only:
            pass
        elif dataset_lock_path is None:
            failures.append("dataset lock path is required for final audit")
        elif track:
            locks_payload = read_dataset_locks(dataset_lock_path)
            split = run_plan_cell.get("split") or manifest.get("split")
            split_text = None
            if isinstance(split, list) and len(split) == 2:
                prefix = "train" if track == "gpqa_diamond" else "test"
                split_text = f"{prefix}[{split[0]}:{split[1]}]"
            try:
                assert_locked_dataset(
                    locks_payload,
                    track=str(track),
                    split=split_text,
                    split_id=run_plan_cell.get("split_id"),
                )
            except ValueError as exc:
                failures.append(str(exc))
        else:
            failures.append("cannot resolve track for dataset lock audit")

    if costs.get("rollout_total") is not None and rollouts:
        if int(costs["rollout_total"]) != int(rollouts.get("total", -1)):
            failures.append("costs.rollout_total does not match rollouts.total")

    if failures:
        raise ArtifactAuditError("; ".join(failures))

    return {
        "passed": True,
        "stage": stage,
        "artifact_dir": str(artifact_dir),
        "track": track,
        "condition": manifest.get("condition"),
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "rollouts_total": rollouts.get("total"),
        "estimated_dollar_cost": costs.get("estimated_dollar_cost"),
    }
