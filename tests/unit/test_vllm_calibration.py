from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from polaris.infra.serving import ScoreBatch
from polaris.infra.vllm_calibration import (
    CalibrationArtifactError,
    mh_log_r,
    mh_replay_row,
    score_parity_rows,
    summarize_mh_replay,
    summarize_score_parity,
    validate_vllm_calibration_artifact,
    write_calibration_artifacts,
)


_RUN_CONDITION_SPEC = importlib.util.spec_from_file_location(
    "run_condition_script", Path(__file__).resolve().parents[2] / "scripts" / "run_condition.py"
)
assert _RUN_CONDITION_SPEC is not None
_RUN_CONDITION = importlib.util.module_from_spec(_RUN_CONDITION_SPEC)
assert _RUN_CONDITION_SPEC.loader is not None
_RUN_CONDITION_SPEC.loader.exec_module(_RUN_CONDITION)


def _scores(
    *,
    norm_tokens: list[list[float]],
    unnorm_tokens: list[list[float]],
) -> ScoreBatch:
    return ScoreBatch(
        lp_norm=[float(sum(row)) for row in norm_tokens],
        lp_unnorm=[float(sum(row)) for row in unnorm_tokens],
        lp_norm_tokens=norm_tokens,
        lp_unnorm_tokens=unnorm_tokens,
    )


def test_score_parity_rows_gate_per_token_and_segment_diffs():
    hf = _scores(norm_tokens=[[-1.0, -2.0]], unnorm_tokens=[[-2.0, -4.0]])
    vllm = _scores(norm_tokens=[[-1.0004, -1.9997]], unnorm_tokens=[[-2.0002, -3.9999]])

    rows = score_parity_rows(
        hf=hf,
        vllm=vllm,
        prefix_lens=[5],
        target_segments=[[10, 11]],
        temperature=0.5,
    )
    summary = summarize_score_parity(rows)

    assert len(rows) == 2
    assert all(row.passed for row in rows)
    assert all(row.segment_passed for row in rows)
    assert rows[0].target_token_id == 10
    assert rows[0].prefix_len == 5
    assert summary["passed"] is True
    assert summary["max_abs_diff"] <= 1e-3
    assert summary["max_segment_abs_diff"] <= 2e-3


def test_mh_log_r_and_replay_match_non_ambiguous_decision():
    cur_hf = _scores(norm_tokens=[[-4.0]], unnorm_tokens=[[-8.0]])
    prop_hf = _scores(norm_tokens=[[-3.0]], unnorm_tokens=[[-5.0]])
    cur_vllm = _scores(norm_tokens=[[-4.0002]], unnorm_tokens=[[-8.0002]])
    prop_vllm = _scores(norm_tokens=[[-3.0002]], unnorm_tokens=[[-5.0002]])

    assert mh_log_r(
        cur_norm=-4.0,
        cur_unnorm=-8.0,
        prop_norm=-3.0,
        prop_unnorm=-5.0,
    ) == pytest.approx(2.0)
    row = mh_replay_row(
        row_id="mh-0",
        cur_hf=cur_hf,
        prop_hf=prop_hf,
        cur_vllm=cur_vllm,
        prop_vllm=prop_vllm,
        row_index=0,
        suffix_len=1,
        temperature=0.5,
        u=0.5,
    )
    summary = summarize_mh_replay([row])

    assert row.hf_accept is True
    assert row.vllm_accept is True
    assert row.ambiguous_boundary is False
    assert row.passed is True
    assert summary["passed"] is True


def test_mh_replay_tracks_boundary_ambiguity_separately():
    cur = _scores(norm_tokens=[[-4.0]], unnorm_tokens=[[-8.0]])
    prop = _scores(norm_tokens=[[-3.0]], unnorm_tokens=[[-9.0]])
    row = mh_replay_row(
        row_id="mh-boundary",
        cur_hf=cur,
        prop_hf=prop,
        cur_vllm=cur,
        prop_vllm=prop,
        row_index=0,
        suffix_len=1,
        temperature=0.5,
        u=0.1353352832366127,
    )

    assert row.ambiguous_boundary is True
    assert summarize_mh_replay([row])["passed"] is False


def test_write_calibration_artifacts(tmp_path):
    hf = _scores(norm_tokens=[[-1.0]], unnorm_tokens=[[-2.0]])
    rows = score_parity_rows(
        hf=hf,
        vllm=hf,
        prefix_lens=[1],
        target_segments=[[7]],
        temperature=0.5,
    )
    mh = mh_replay_row(
        row_id="mh-0",
        cur_hf=hf,
        prop_hf=hf,
        cur_vllm=hf,
        prop_vllm=hf,
        row_index=0,
        suffix_len=1,
        temperature=0.5,
        u=0.5,
    )

    summary = write_calibration_artifacts(
        tmp_path,
        score_rows=rows,
        mh_rows=[mh],
        full_chain_replay={
            "passed": True,
            "acceptance_count_match": True,
            "final_token_chain_match": True,
        },
    )

    assert summary["passed"] is True
    for rel in (
        "score_parity.jsonl",
        "mh_replay_parity.jsonl",
        "full_chain_replay.json",
        "calibration_summary.json",
        "calibration_report.md",
    ):
        assert (tmp_path / rel).exists()
    loaded = json.loads((tmp_path / "calibration_summary.json").read_text())
    assert loaded["passed"] is True


def test_validate_vllm_calibration_artifact_accepts_passing_summary(tmp_path):
    payload = {
        "passed": True,
        "score_parity": {"passed": True, "tolerance": 1e-3},
        "mh_replay_parity": {"passed": True},
        "full_chain_replay": {
            "passed": True,
            "tokenizer_parity": {"passed": True},
        },
    }
    path = tmp_path / "calibration_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = validate_vllm_calibration_artifact(tmp_path)

    assert loaded["passed"] is True
    assert loaded["_artifact_path"] == str(path)


def test_validate_vllm_calibration_artifact_checks_expected_model_id(tmp_path):
    payload = {
        "passed": True,
        "score_parity": {"passed": True, "tolerance": 1e-3},
        "mh_replay_parity": {"passed": True},
        "full_chain_replay": {
            "passed": True,
            "runtime_metadata": {
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            },
            "tokenizer_parity": {
                "passed": True,
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            },
        },
    }
    path = tmp_path / "calibration_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = validate_vllm_calibration_artifact(
        path,
        expected_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )

    assert loaded["passed"] is True
    with pytest.raises(CalibrationArtifactError, match="model_id does not match"):
        validate_vllm_calibration_artifact(
            path,
            expected_model_id="Qwen/Qwen2.5-Math-1.5B",
        )


def test_vllm_calibration_required_only_for_power_conditions():
    assert _RUN_CONDITION._condition_requires_vllm_calibration("single_prompt_power")
    assert _RUN_CONDITION._condition_requires_vllm_calibration("proicl_gepa_mcmc")
    assert _RUN_CONDITION._condition_requires_vllm_calibration(
        "proicl_gepa_mcmc_memory"
    )
    assert not _RUN_CONDITION._condition_requires_vllm_calibration("greedy")
    assert not _RUN_CONDITION._condition_requires_vllm_calibration("gepa_only")


def test_validate_vllm_calibration_artifact_rejects_failed_summary(tmp_path):
    path = tmp_path / "calibration_summary.json"
    path.write_text(
        json.dumps(
            {
                "passed": False,
                "score_parity": {"passed": False, "tolerance": 1e-3},
                "mh_replay_parity": {"passed": True},
                "full_chain_replay": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationArtifactError, match="score_parity did not pass"):
        validate_vllm_calibration_artifact(path)
