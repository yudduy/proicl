from __future__ import annotations

import json

import pytest

from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.io.manifest import compute_archive_hash, write_run_manifest


def _kwargs(tmp_path, **overrides):
    base = dict(
        path=tmp_path / "manifest.json",
        model_id="Qwen/Qwen2.5-7B",
        benchmark="MATH500",
        split=(0, 75),
        seeds=[17, 71, 1729],
        condition="full_archive_fixed",
        archive_hash="deadbeef",
        alpha_policy_id="fixed_alpha_4",
        config={
            "mcmc_steps": 10,
            "mcmc_block_num": 16,
            "max_new_tokens": 3072,
            "alpha": 4,
        },
        polaris_source_hash="abc123",
        vendored_commits={
            "rws": "720a8e9d",
            "evalplus": "26d6d00b",
            "gepa": "ce51b50c",
            "dc": "5cfe3c37",
        },
        verifier_id="math/sympy-equivalence-v1",
        preregistration_anchor="TODO.md#polaris-math500-v1",
    )
    base.update(overrides)
    return base


def test_manifest_contains_all_required_fields(tmp_path):
    write_run_manifest(**_kwargs(tmp_path))
    data = json.loads((tmp_path / "manifest.json").read_text())
    for field in (
        "started_at",
        "host",
        "model",
        "benchmark",
        "split",
        "config",
        "seeds",
        "polaris_source_hash",
        "vendored_commits",
        "archive_hash",
        "alpha_policy_id",
        "verifier_id",
        "preregistration_anchor",
        "condition",
    ):
        assert field in data, f"manifest missing required field: {field}"


def test_manifest_preserves_caller_values(tmp_path):
    write_run_manifest(**_kwargs(tmp_path))
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["model"] == "Qwen/Qwen2.5-7B"
    assert data["benchmark"] == "MATH500"
    assert data["split"] == [0, 75]  # tuple → list in JSON
    assert data["seeds"] == [17, 71, 1729]
    assert data["condition"] == "full_archive_fixed"
    assert data["archive_hash"] == "deadbeef"
    assert data["alpha_policy_id"] == "fixed_alpha_4"
    assert data["verifier_id"] == "math/sympy-equivalence-v1"
    assert data["vendored_commits"]["rws"] == "720a8e9d"


def test_manifest_refuses_empty_preregistration_anchor(tmp_path):
    with pytest.raises(ValueError, match="preregistration"):
        write_run_manifest(**_kwargs(tmp_path, preregistration_anchor=""))


def test_manifest_started_at_is_iso_utc(tmp_path):
    write_run_manifest(**_kwargs(tmp_path))
    data = json.loads((tmp_path / "manifest.json").read_text())
    # ISO 8601 UTC, ends with Z or +00:00
    assert "T" in data["started_at"]
    assert data["started_at"].endswith("Z") or data["started_at"].endswith("+00:00")


def test_manifest_caller_can_override_started_at_and_host(tmp_path):
    write_run_manifest(
        **_kwargs(tmp_path, started_at="2026-05-12T00:00:00Z", host="test-host")
    )
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["started_at"] == "2026-05-12T00:00:00Z"
    assert data["host"] == "test-host"


def test_compute_archive_hash_is_deterministic():
    h1 = compute_archive_hash(MATH500_ARCHIVE_V1)
    h2 = compute_archive_hash(MATH500_ARCHIVE_V1)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_compute_archive_hash_differs_for_different_archives():
    from polaris.core.archive import FrozenArchive, PromptEntry

    a = FrozenArchive(
        entries=(PromptEntry(id="x", prefix="A", suffix="", descriptor_hint="d"),)
    )
    b = FrozenArchive(
        entries=(PromptEntry(id="x", prefix="B", suffix="", descriptor_hint="d"),)
    )
    assert compute_archive_hash(a) != compute_archive_hash(b)


def test_manifest_returns_written_dict(tmp_path):
    result = write_run_manifest(**_kwargs(tmp_path))
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert result == on_disk
