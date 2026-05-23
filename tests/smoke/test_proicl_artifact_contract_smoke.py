from __future__ import annotations

import json
import subprocess
import sys


def test_proicl_artifact_contract_smoke_writes_audited_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_proicl_artifact_contract.py",
            "--series-root",
            str(tmp_path / "runs"),
            "--tracks",
            "reasoning_gym_family_relationships",
            "reasoning_gym_boxnet",
            "--run-tag",
            "pytest",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["artifact_audit_passed"] is True
    assert payload["total_cells"] == 24

    run_root = tmp_path / "runs"
    run_dirs = list(run_root.glob("proicl_artifact-contract-smoke_*_fixture_pytest_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run_index.json").exists()
    assert (run_dirs[0] / "full" / "analysis" / "proicl_decomposition.json").exists()
    assert (run_dirs[0] / "full" / "analysis" / "artifact_audit.json").exists()
