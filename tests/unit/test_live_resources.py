from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from polaris.gepa_reflection import (
    CapEnforcedReflectionLM,
    XAI_BASE_URL_DEFAULT,
    XAI_MODEL_DEFAULT,
    XAIReflectionConfig,
    estimate_xai_reflection_cost,
    load_env_file,
)
from polaris.infra.resources import (
    REQUIRED_PRODUCTION_ARTIFACTS,
    get_resource_profile,
    load_resource_profiles,
    validate_profile_cost,
)


def test_live_resource_profiles_lock_default_execution_order():
    profiles = load_resource_profiles()

    assert tuple(profiles) == (
        "farmshare_l40_free",
        "flow_a100_weekend",
        "modal_burst",
        "cloudrift_fallback",
    )

    farmshare = profiles["farmshare_l40_free"]
    flow = profiles["flow_a100_weekend"]
    modal = profiles["modal_burst"]
    cloudrift = profiles["cloudrift_fallback"]

    assert farmshare.run_kind == "farmshare"
    assert farmshare.free is True
    assert farmshare.gpu_count == 4
    assert farmshare.max_concurrent_jobs == 4
    assert flow.run_kind == "flow"
    assert flow.gpu_model == "a100-80gb.sxm"
    assert flow.gpu_count == 4
    assert flow.max_bid_dollars_per_gpu_hour == pytest.approx(0.025)
    assert flow.initial_spend_cap_dollars == pytest.approx(10.0)
    assert modal.allowed_phases == ("phase3", "debug")
    assert cloudrift.fallback_only is True


def test_resource_profile_cost_validation_fails_closed():
    flow = get_resource_profile("flow_a100_weekend")
    farmshare = get_resource_profile("farmshare_l40_free")

    assert validate_profile_cost(flow, estimated_dollar_cost=9.99)["passed"] is True
    with pytest.raises(ValueError, match="exceeds profile initial_spend_cap_dollars"):
        validate_profile_cost(flow, estimated_dollar_cost=10.01)
    with pytest.raises(ValueError, match="FarmShare profile must remain zero-cost"):
        validate_profile_cost(farmshare, estimated_dollar_cost=0.01)


def test_live_launcher_renders_backend_agnostic_contract(tmp_path):
    out = tmp_path / "bundle.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_prorl_recovery.py",
            "plan",
            "--profile",
            "flow_a100_weekend",
            "--phase",
            "phase1",
            "--tracks",
            "math500",
            "--problem-count",
            "20",
            "--samples-per-problem",
            "16",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    stdout = json.loads(result.stdout)
    assert stdout["cells"] == 16
    assert payload["resource_profile"]["key"] == "flow_a100_weekend"
    assert payload["resource_profile"]["run_kind"] == "flow"
    assert payload["artifact_contract"] == list(REQUIRED_PRODUCTION_ARTIFACTS)
    assert payload["launch_policy"]["tensor_parallelism"] == "disabled"
    assert payload["launch_policy"]["requires_fresh_user_go"] is True


def test_xai_reflection_config_redacts_secret_and_estimates_cost(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "XAI_API_KEY=secret-value",
                "XAI_BASE_URL=https://api.x.ai/v1",
                "XAI_REFLECTION_MODEL=grok-4.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = {}
    load_env_file(env_file, env=env)
    config = XAIReflectionConfig.from_env(env)

    assert config.api_key == "secret-value"
    assert config.base_url == XAI_BASE_URL_DEFAULT
    assert config.model == XAI_MODEL_DEFAULT
    assert config.to_manifest()["api_key"] == "<redacted>"
    assert "secret-value" not in json.dumps(config.to_manifest())

    estimate = estimate_xai_reflection_cost(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate["estimated_dollar_cost"] == pytest.approx(3.75)


def test_xai_reflection_lm_fails_closed_before_cost_cap():
    class FakePaidLM:
        total_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0

        def __call__(self, prompt):
            self.total_cost += 99.0
            return "should-not-run"

    config = XAIReflectionConfig(
        api_key="secret-value",
        initial_cost_cap_dollars=0.001,
        hard_cost_cap_dollars=0.001,
        max_tokens=1024,
    )
    capped = CapEnforcedReflectionLM(FakePaidLM(), config)

    with pytest.raises(ValueError, match="xAI reflection estimate exceeds"):
        capped("x" * 4000)


def test_xai_reflection_smoke_dry_run_never_prints_secret(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("XAI_API_KEY=secret-value\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/xai_reflection_smoke.py",
            "--env-file",
            str(env_file),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert "secret-value" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["provider"] == "xai"
    assert payload["config"]["api_key"] == "<redacted>"


def test_archive_builder_records_xai_reflection_without_secret(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("XAI_API_KEY=secret-value\n", encoding="utf-8")
    out = tmp_path / "archive"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_polaris_archive.py",
            "--track",
            "math500",
            "--mode",
            "freeze",
            "--dev-split",
            "75",
            "76",
            "--out",
            str(out),
            "--dry-run",
            "--reflection-provider",
            "xai",
            "--env-file",
            str(env_file),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert "secret-value" not in result.stdout
    manifest = json.loads((out / "archive_build_manifest.json").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest)
    assert "secret-value" not in serialized
    assert manifest["reflection"]["provider"] == "xai"
    assert manifest["reflection"]["config"]["api_key"] == "<redacted>"
    assert manifest["reflection"]["config"]["model"] == "grok-4.3"


def test_flow_a100x4_probe_yaml_is_bounded_and_secret_safe():
    root = Path(__file__).resolve().parents[2]
    config = (root / "configs/flow_prorl_a100x4_probe.yaml").read_text(encoding="utf-8")

    assert "instance_type: 4xa100" in config
    assert "region: us-central3-a" in config
    assert "allocation_mode: spot" in config
    assert "preemptible_ok: true" in config
    assert "max_price_per_hour: 0.10" in config
    assert "max_run_time_hours: 2" in config
    assert "terminate_on_exit: true" in config
    assert "upload_code: false" in config
    assert "POLARIS_FLOW_A100X4_PROBE_OK" in config
    assert "HF_HOME: /workspace/.cache/huggingface" in config
    assert "POLARIS_RUN_ROOT: /workspace/polaris/runs/prorl_recovery" in config
    assert re.search(r"hf_[A-Za-z0-9]{20,}", config) is None
    assert re.search(r"xai-[A-Za-z0-9]{20,}", config) is None


def test_flowignore_excludes_secrets_and_large_local_artifacts():
    root = Path(__file__).resolve().parents[2]
    ignored = set((root / ".flowignore").read_text(encoding="utf-8").splitlines())

    assert ".env" in ignored
    assert ".venv-eval/" in ignored
    assert "runs/" in ignored
    assert "upstream/" in ignored


def test_flow_probe_script_writes_integrity_artifacts_without_secret_values():
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/flow_prorl_a100x4_probe.sh").read_text(encoding="utf-8")

    for artifact in (
        "environment.json",
        "preflight_imports.json",
        "gpu_check.json",
        "manifest.json",
        "audit.md",
    ):
        assert artifact in script
    assert "expected_gpus\": 4" in script
    assert "nvidia-smi" in script
    assert "export OUT_DIR" in script
    assert re.search(r"hf_[A-Za-z0-9]{20,}", script) is None
    assert re.search(r"xai-[A-Za-z0-9]{20,}", script) is None
