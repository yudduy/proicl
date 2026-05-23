from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_packager():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "package_sps_results.py"
    spec = importlib.util.spec_from_file_location("package_sps_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_package_sps_results_writes_small_bundle(tmp_path, monkeypatch):
    packager = _load_packager()
    run_root = tmp_path / "run"
    full = run_root / "full"
    shard = (
        full
        / "runs"
        / "reasoning_gym_graph_color_n12"
        / "gepa_sps_fixed"
        / "shard-0"
    )
    _write_json(
        shard / "metrics.json",
        {
            "accuracy": 1.0,
            "mean_selected_score": 1.0,
            "n_problems": 1,
            "n_candidates": 2,
            "B_per_problem": 2,
            "power_sampler": "sps",
            "alpha_policy_id": "fixed_alpha_4",
            "backend": "vllm",
            "vllm_scoring_mode": "forced_decode_v0",
        },
    )
    _write_jsonl(
        shard / "selected.jsonl",
        [
            {
                "problem_id": "reasoning_gym:graph_color:seed=0:index=23",
                "selected_passed": True,
                "selected_score": 1.0,
                "prompt_id": "gepa-0",
                "sample_index": 0,
                "alpha": 4.0,
            }
        ],
    )
    _write_jsonl(
        shard / "scores.jsonl",
        [{"problem_id": "reasoning_gym:graph_color:seed=0:index=23", "score": 1.0}],
    )
    (shard / "audit.md").write_text("# audit\n", encoding="utf-8")
    _write_json(run_root / "run_index.json", {"power_sampler": "sps"})
    (full / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["package_sps_results.py", "--run-root", str(run_root)],
    )
    packager.main()

    bundle = run_root / "sps_results_bundle"
    assert (bundle / "summary" / "metrics.csv").exists()
    assert (bundle / "summary" / "per_problem_selected.csv").exists()
    assert (bundle / "summary" / "per_problem_agreement.csv").exists()
    assert (bundle / "runs" / "reasoning_gym_graph_color_n12" / "gepa_sps_fixed" / "shard-0" / "selected.jsonl").exists()
    assert not (bundle / "runs" / "reasoning_gym_graph_color_n12" / "gepa_sps_fixed" / "shard-0" / "candidates.jsonl").exists()
    assert (run_root / "sps_results_bundle.tar.gz").exists()
