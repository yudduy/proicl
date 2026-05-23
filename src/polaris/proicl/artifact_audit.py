from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


COMMON_CELL_ARTIFACTS: tuple[str, ...] = (
    "manifest.json",
    "archive.json",
    "candidates.jsonl",
    "scores.jsonl",
    "selected.jsonl",
    "metrics.json",
    "costs.json",
    "rollouts.json",
    "preflight.json",
    "environment.json",
    "run_plan_cell.json",
    "audit.md",
)


def required_cell_artifacts(cell: dict[str, Any]) -> tuple[str, ...]:
    extra: list[str] = []
    if cell.get("uses_gepa_archive") or cell.get("uses_memory"):
        extra.append("archive_build_manifest.json")
    if cell.get("uses_repair"):
        extra.append("repair_traces.jsonl")
    if cell.get("uses_fork_search"):
        extra.append("fork_traces.jsonl")
    if cell.get("uses_memory"):
        extra.extend(["memory.sqlite", "memory_events.jsonl"])
    return COMMON_CELL_ARTIFACTS + tuple(extra)


def load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"cells": payload}
    if not isinstance(payload, dict) or "cells" not in payload:
        raise ValueError(f"not a ProICL plan file: {path}")
    return payload


def audit_proicl_plan(plan_path: Path) -> dict[str, Any]:
    plan = load_plan(plan_path)
    cells = list(plan.get("cells", []))
    by_condition: Counter[str] = Counter()
    by_track: Counter[str] = Counter()
    complete_by_condition: Counter[str] = Counter()
    missing_rows: list[dict[str, Any]] = []
    for cell in cells:
        condition = str(cell.get("proicl_condition") or cell.get("condition"))
        track = str(cell.get("track"))
        by_condition[condition] += 1
        by_track[track] += 1
        artifact_dir = Path(str(cell.get("artifact_dir")))
        required = required_cell_artifacts(cell)
        missing = [
            name
            for name in required
            if not (artifact_dir / name).exists() or (artifact_dir / name).stat().st_size == 0
        ]
        if missing:
            missing_rows.append(
                {
                    "track": track,
                    "condition": condition,
                    "shard_id": cell.get("shard_id"),
                    "artifact_dir": str(artifact_dir),
                    "missing": missing,
                }
            )
        else:
            complete_by_condition[condition] += 1
    complete_cells = len(cells) - len(missing_rows)
    return {
        "plan_path": str(plan_path),
        "total_cells": len(cells),
        "complete_cells": complete_cells,
        "missing_cells": len(missing_rows),
        "passed": len(missing_rows) == 0,
        "by_condition": dict(sorted(by_condition.items())),
        "by_track": dict(sorted(by_track.items())),
        "complete_by_condition": dict(sorted(complete_by_condition.items())),
        "missing": missing_rows,
    }


def write_proicl_artifact_audit(plan_path: Path, out_dir: Path) -> dict[str, Any]:
    report = audit_proicl_plan(plan_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifact_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ProICL Artifact Audit",
        "",
        f"- plan: `{report['plan_path']}`",
        f"- total_cells: {report['total_cells']}",
        f"- complete_cells: {report['complete_cells']}",
        f"- missing_cells: {report['missing_cells']}",
        f"- passed: {str(report['passed']).lower()}",
        "",
        "## Conditions",
    ]
    for condition, total in report["by_condition"].items():
        complete = report["complete_by_condition"].get(condition, 0)
        lines.append(f"- `{condition}`: {complete}/{total}")
    if report["missing"]:
        lines.extend(["", "## Missing"])
        for row in report["missing"][:100]:
            lines.append(
                "- `{track}` `{condition}` shard `{shard_id}` missing: {missing}".format(
                    track=row["track"],
                    condition=row["condition"],
                    shard_id=row["shard_id"],
                    missing=", ".join(row["missing"]),
                )
            )
    (out_dir / "artifact_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
