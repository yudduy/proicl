from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from polaris.infra.serving import ScoreBatch

DEFAULT_LOGPROB_ATOL = 1e-3
DEFAULT_AMBIGUOUS_RATE_LIMIT = 0.01


class CalibrationArtifactError(ValueError):
    """Raised when a vLLM/HF calibration artifact is missing or failed."""


@dataclass(frozen=True)
class ScoreParityRow:
    row_id: str
    target_token_id: int
    prefix_len: int
    token_pos: int
    temperature: float
    hf_lp_norm: float
    vllm_lp_norm: float
    hf_lp_unnorm: float
    vllm_lp_unnorm: float
    norm_abs_diff: float
    unnorm_abs_diff: float
    segment_norm_abs_diff: float
    segment_unnorm_abs_diff: float
    segment_tolerance: float
    segment_passed: bool
    passed: bool


@dataclass(frozen=True)
class MHReplayRow:
    row_id: str
    suffix_len: int
    temperature: float
    log_u: float
    hf_log_r: float
    vllm_log_r: float
    abs_diff: float
    tolerance: float
    hf_accept: bool
    vllm_accept: bool
    ambiguous_boundary: bool
    passed: bool


def mh_log_r(
    *,
    cur_norm: float,
    cur_unnorm: float,
    prop_norm: float,
    prop_unnorm: float,
) -> float:
    """RWS Metropolis-Hastings log acceptance ratio."""
    return float(prop_unnorm + cur_norm - cur_unnorm - prop_norm)


def score_parity_rows(
    *,
    hf: ScoreBatch,
    vllm: ScoreBatch,
    prefix_lens: list[int],
    target_segments: list[list[int]],
    temperature: float,
    atol: float = DEFAULT_LOGPROB_ATOL,
    row_prefix: str = "score",
) -> list[ScoreParityRow]:
    if len(hf.lp_norm_tokens) != len(vllm.lp_norm_tokens):
        raise ValueError("HF and vLLM score batches have different row counts")
    if len(prefix_lens) != len(target_segments) or len(prefix_lens) != len(hf.lp_norm_tokens):
        raise ValueError("prefix_lens, target_segments, and score rows must align")

    rows: list[ScoreParityRow] = []
    for row_idx, (prefix_len, targets) in enumerate(zip(prefix_lens, target_segments)):
        hf_norm = hf.lp_norm_tokens[row_idx]
        hf_unnorm = hf.lp_unnorm_tokens[row_idx]
        vllm_norm = vllm.lp_norm_tokens[row_idx]
        vllm_unnorm = vllm.lp_unnorm_tokens[row_idx]
        lengths = {len(targets), len(hf_norm), len(hf_unnorm), len(vllm_norm), len(vllm_unnorm)}
        if len(lengths) != 1:
            raise ValueError(
                "token/logprob length mismatch for parity row "
                f"{row_idx}: target={len(targets)} hf_norm={len(hf_norm)} "
                f"hf_unnorm={len(hf_unnorm)} vllm_norm={len(vllm_norm)} "
                f"vllm_unnorm={len(vllm_unnorm)}"
            )
        segment_norm_diff = abs(
            sum(float(x) for x in hf_norm) - sum(float(x) for x in vllm_norm)
        )
        segment_unnorm_diff = abs(
            sum(float(x) for x in hf_unnorm) - sum(float(x) for x in vllm_unnorm)
        )
        segment_tolerance = atol * max(1, len(targets))
        segment_passed = (
            segment_norm_diff <= segment_tolerance
            and segment_unnorm_diff <= segment_tolerance
        )
        for token_pos, token_id in enumerate(targets):
            norm_diff = abs(float(hf_norm[token_pos]) - float(vllm_norm[token_pos]))
            unnorm_diff = abs(float(hf_unnorm[token_pos]) - float(vllm_unnorm[token_pos]))
            rows.append(
                ScoreParityRow(
                    row_id=f"{row_prefix}-{row_idx}-{token_pos}",
                    target_token_id=int(token_id),
                    prefix_len=int(prefix_len),
                    token_pos=int(token_pos),
                    temperature=float(temperature),
                    hf_lp_norm=float(hf_norm[token_pos]),
                    vllm_lp_norm=float(vllm_norm[token_pos]),
                    hf_lp_unnorm=float(hf_unnorm[token_pos]),
                    vllm_lp_unnorm=float(vllm_unnorm[token_pos]),
                    norm_abs_diff=norm_diff,
                    unnorm_abs_diff=unnorm_diff,
                    segment_norm_abs_diff=float(segment_norm_diff),
                    segment_unnorm_abs_diff=float(segment_unnorm_diff),
                    segment_tolerance=float(segment_tolerance),
                    segment_passed=bool(segment_passed),
                    passed=norm_diff <= atol and unnorm_diff <= atol and segment_passed,
                )
            )
    return rows


def summarize_score_parity(
    rows: list[ScoreParityRow],
    *,
    atol: float = DEFAULT_LOGPROB_ATOL,
) -> dict[str, Any]:
    diffs = [max(row.norm_abs_diff, row.unnorm_abs_diff) for row in rows]
    return {
        "kind": "score_parity",
        "tolerance": float(atol),
        "n_rows": len(rows),
        "passed": bool(rows) and all(row.passed for row in rows),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": mean(diffs) if diffs else None,
        "max_segment_abs_diff": max(
            (
                max(row.segment_norm_abs_diff, row.segment_unnorm_abs_diff)
                for row in rows
            ),
            default=None,
        ),
    }


def mh_replay_row(
    *,
    row_id: str,
    cur_hf: ScoreBatch,
    prop_hf: ScoreBatch,
    cur_vllm: ScoreBatch,
    prop_vllm: ScoreBatch,
    row_index: int,
    suffix_len: int,
    temperature: float,
    u: float,
    atol_per_token: float = DEFAULT_LOGPROB_ATOL,
) -> MHReplayRow:
    if not 0.0 < u < 1.0:
        raise ValueError("u must be in the open interval (0, 1)")
    tolerance = float(atol_per_token) * max(1, int(suffix_len))
    hf_log_r = mh_log_r(
        cur_norm=cur_hf.lp_norm[row_index],
        cur_unnorm=cur_hf.lp_unnorm[row_index],
        prop_norm=prop_hf.lp_norm[row_index],
        prop_unnorm=prop_hf.lp_unnorm[row_index],
    )
    vllm_log_r = mh_log_r(
        cur_norm=cur_vllm.lp_norm[row_index],
        cur_unnorm=cur_vllm.lp_unnorm[row_index],
        prop_norm=prop_vllm.lp_norm[row_index],
        prop_unnorm=prop_vllm.lp_unnorm[row_index],
    )
    log_u = math.log(float(u))
    hf_accept = log_u < hf_log_r
    vllm_accept = log_u < vllm_log_r
    diff = abs(hf_log_r - vllm_log_r)
    ambiguous = abs(log_u - hf_log_r) <= tolerance
    return MHReplayRow(
        row_id=row_id,
        suffix_len=int(suffix_len),
        temperature=float(temperature),
        log_u=float(log_u),
        hf_log_r=float(hf_log_r),
        vllm_log_r=float(vllm_log_r),
        abs_diff=float(diff),
        tolerance=tolerance,
        hf_accept=bool(hf_accept),
        vllm_accept=bool(vllm_accept),
        ambiguous_boundary=bool(ambiguous),
        passed=diff <= tolerance and (hf_accept == vllm_accept or ambiguous),
    )


def summarize_mh_replay(
    rows: list[MHReplayRow],
    *,
    ambiguous_rate_limit: float = DEFAULT_AMBIGUOUS_RATE_LIMIT,
) -> dict[str, Any]:
    ambiguous_count = sum(1 for row in rows if row.ambiguous_boundary)
    ambiguous_rate = ambiguous_count / len(rows) if rows else 0.0
    return {
        "kind": "mh_replay_parity",
        "n_rows": len(rows),
        "passed": bool(rows)
        and all(row.passed for row in rows)
        and ambiguous_rate <= ambiguous_rate_limit,
        "max_abs_diff": max((row.abs_diff for row in rows), default=None),
        "ambiguous_boundary_count": ambiguous_count,
        "ambiguous_boundary_rate": ambiguous_rate,
        "ambiguous_boundary_rate_limit": float(ambiguous_rate_limit),
    }


def write_calibration_artifacts(
    out_dir: Path,
    *,
    score_rows: list[ScoreParityRow],
    mh_rows: list[MHReplayRow],
    full_chain_replay: dict[str, Any] | None = None,
    stochastic_sanity: dict[str, Any] | None = None,
    atol: float = DEFAULT_LOGPROB_ATOL,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "score_parity.jsonl", [asdict(row) for row in score_rows])
    _write_jsonl(out_dir / "mh_replay_parity.jsonl", [asdict(row) for row in mh_rows])
    full_chain_payload = full_chain_replay or {
        "passed": None,
        "reason": "not_run",
    }
    stochastic_payload = stochastic_sanity or {
        "passed": None,
        "reason": "not_run",
    }
    _write_json(out_dir / "full_chain_replay.json", full_chain_payload)
    score_summary = summarize_score_parity(score_rows, atol=atol)
    mh_summary = summarize_mh_replay(mh_rows)
    summary = {
        "passed": bool(
            score_summary["passed"]
            and mh_summary["passed"]
            and full_chain_payload.get("passed") is True
        ),
        "score_parity": score_summary,
        "mh_replay_parity": mh_summary,
        "full_chain_replay": full_chain_payload,
        "stochastic_sanity": stochastic_payload,
        "artifacts": {
            "score_parity": "score_parity.jsonl",
            "mh_replay_parity": "mh_replay_parity.jsonl",
            "full_chain_replay": "full_chain_replay.json",
            "summary": "calibration_summary.json",
            "report": "calibration_report.md",
        },
    }
    _write_json(out_dir / "calibration_summary.json", summary)
    (out_dir / "calibration_report.md").write_text(_calibration_report(summary), encoding="utf-8")
    return summary


def validate_vllm_calibration_artifact(
    artifact_path: Path | str | None,
    *,
    max_logprob_atol: float = DEFAULT_LOGPROB_ATOL,
    expected_model_id: str | None = None,
) -> dict[str, Any]:
    if artifact_path is None:
        raise CalibrationArtifactError(
            "vLLM backend requires --vllm-parity-artifact pointing to a "
            "passing calibration_summary.json"
        )
    path = Path(artifact_path)
    if path.is_dir():
        path = path / "calibration_summary.json"
    if not path.exists():
        raise CalibrationArtifactError(f"vLLM parity artifact does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationArtifactError(
            f"vLLM parity artifact is not valid JSON: {path}"
        ) from exc

    failures: list[str] = []
    if payload.get("passed") is not True:
        failures.append("top-level passed is not true")

    score = payload.get("score_parity", {})
    if score.get("passed") is not True:
        failures.append("score_parity did not pass")
    tolerance = score.get("tolerance")
    if tolerance is not None and float(tolerance) > max_logprob_atol:
        failures.append(f"score_parity tolerance {tolerance} exceeds {max_logprob_atol}")

    mh = payload.get("mh_replay_parity", {})
    if mh.get("passed") is not True:
        failures.append("mh_replay_parity did not pass")

    chain = payload.get("full_chain_replay", {})
    if chain.get("passed") is not True:
        failures.append("full_chain_replay did not pass")

    tokenizer = chain.get("tokenizer_parity")
    if isinstance(tokenizer, dict) and tokenizer.get("passed") is not True:
        failures.append("tokenizer_parity did not pass")

    observed_model_id = _calibration_model_id(payload)
    if expected_model_id is not None and observed_model_id != expected_model_id:
        failures.append(
            "calibration model_id does not match requested model: "
            f"expected={expected_model_id} observed={observed_model_id}"
        )

    if failures:
        raise CalibrationArtifactError(
            f"vLLM parity artifact failed gates at {path}: " + "; ".join(failures)
        )
    payload["_artifact_path"] = str(path)
    return payload


def _calibration_model_id(payload: dict[str, Any]) -> str | None:
    chain = payload.get("full_chain_replay")
    if isinstance(chain, dict):
        runtime = chain.get("runtime_metadata")
        if isinstance(runtime, dict) and runtime.get("model_id") is not None:
            return str(runtime["model_id"])
        vllm_runtime = chain.get("vllm_runtime_metadata")
        if isinstance(vllm_runtime, dict) and vllm_runtime.get("model_id") is not None:
            return str(vllm_runtime["model_id"])
        hf_runtime = chain.get("hf_runtime_metadata")
        if isinstance(hf_runtime, dict) and hf_runtime.get("model_id") is not None:
            return str(hf_runtime["model_id"])
        tokenizer = chain.get("tokenizer_parity")
        if isinstance(tokenizer, dict) and tokenizer.get("model_id") is not None:
            return str(tokenizer["model_id"])

    tokenizer = payload.get("tokenizer_parity")
    if isinstance(tokenizer, dict) and tokenizer.get("model_id") is not None:
        return str(tokenizer["model_id"])
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _calibration_report(summary: dict[str, Any]) -> str:
    score = summary["score_parity"]
    mh = summary["mh_replay_parity"]
    return (
        "# vLLM/HF calibration\n\n"
        f"- passed: {summary['passed']}\n"
        f"- score_rows: {score['n_rows']}\n"
        f"- score_max_abs_diff: {score['max_abs_diff']}\n"
        f"- score_mean_abs_diff: {score['mean_abs_diff']}\n"
        f"- mh_rows: {mh['n_rows']}\n"
        f"- mh_max_abs_diff: {mh['max_abs_diff']}\n"
        f"- mh_ambiguous_boundary_rate: {mh['ambiguous_boundary_rate']}\n"
    )
