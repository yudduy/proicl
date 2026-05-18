from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_readiness_smoke_command_writes_artifacts_and_cache_replays(tmp_path):
    out_dir = tmp_path / "readiness"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_polaris_readiness.py",
            "--out",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report_path = out_dir / "readiness_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["preflight"]["passed"] is True
    assert report["cache_replay"]["generation_calls_on_replay"] == 0

    required = {
        "manifest.json",
        "archive.json",
        "candidates.jsonl",
        "scores.jsonl",
        "costs.json",
        "metrics.json",
        "audit.md",
    }
    for condition in report["runnable_conditions"]:
        condition_dir = out_dir / "math500" / condition
        assert required <= {p.name for p in condition_dir.iterdir()}

    assert report["deferred_tracks"] == []
    assert report["deferred_experiments"] == []
    requirements = {item["requirement"]: item for item in report["checklist"]}
    for requirement in (
        "HumanEval+ track",
        "GPQA-Diamond track",
        "archive_size_sweep",
        "archive_construction_gepa_iterations",
        "memory_composition",
        "decaying_alpha_diversity",
        "joint_optimization",
        "factorial_interaction",
        "descriptor_ablation",
        "verifier_gating_ablation",
        "break_even_cost_accounting",
    ):
        assert requirements[requirement]["status"] == "pass"
