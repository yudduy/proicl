from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from polaris.core.archive import MATH500_ARCHIVE_V1


def _archive(path: Path) -> Path:
    path.write_text(json.dumps(MATH500_ARCHIVE_V1.to_jsonable()), encoding="utf-8")
    return path


def test_generic_runner_preflight_only_passes_for_every_track(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    archive = _archive(tmp_path / "archive.json")
    for track, model_key in (
        ("math500", "qwen2.5-math-7b"),
        ("humaneval_plus", "qwen2.5-7b"),
        ("gpqa_diamond", "qwen2.5-7b"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_condition.py",
                "--track",
                track,
                "--model-key",
                model_key,
                "--condition",
                "greedy",
                "--archive",
                str(archive),
                "--split",
                "0",
                "1",
                "--out",
                str(tmp_path / track),
                "--polaris-source-hash",
                "dev",
                "--preregistration-anchor",
                "TODO.md#test",
                "--preflight-only",
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert '"passed": true' in result.stdout
