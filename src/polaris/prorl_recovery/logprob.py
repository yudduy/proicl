from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TrajectoryForScoring:
    row_id: str
    task_family: str
    problem_id: str
    prompt_text: str
    response_text: str


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool | None = None) -> list[int]:
    if add_special_tokens is None:
        return list(tokenizer.encode(text))
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


def score_trajectories_batched(
    *,
    sampler: Any,
    trajectories: Iterable[TrajectoryForScoring],
    batch_size: int = 8,
    include_token_logprobs: bool = True,
) -> list[dict[str, Any]]:
    """Score whole responses with the HF/RWS `score_segments` contract.

    The caller chooses the sampler. For bucket assignment this must be the HF
    scorer; vLLM forced-score output is diagnostic unless it has passed a
    full-trajectory parity smoke.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tokenizer = getattr(sampler, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("sampler must expose a tokenizer for trajectory scoring")

    rows: list[dict[str, Any]] = []
    pending: list[TrajectoryForScoring] = []

    def flush() -> None:
        if not pending:
            return
        prefix_batch: list[list[int]] = []
        target_batch: list[list[int]] = []
        for trajectory in pending:
            prefix_batch.append(_encode(tokenizer, trajectory.prompt_text))
            target_batch.append(
                _encode(tokenizer, trajectory.response_text, add_special_tokens=False)
            )
        scores = sampler.score_segments(
            prefix_batch,
            target_batch,
            temperature=1.0,
        )
        for trajectory, target_ids, lp_sum, token_lps in zip(
            pending,
            target_batch,
            scores.lp_norm,
            scores.lp_norm_tokens,
        ):
            token_count = len(target_ids)
            out = {
                "row_id": trajectory.row_id,
                "task_family": trajectory.task_family,
                "problem_id": trajectory.problem_id,
                "lp_base_sum": float(lp_sum),
                "lp_base_mean": float(lp_sum) / token_count if token_count else 0.0,
                "token_count": token_count,
                "scorer": "hf_rws_score_segments_temperature_1",
            }
            if include_token_logprobs:
                out["token_logprobs"] = [float(x) for x in token_lps]
            rows.append(out)
        pending.clear()

    for trajectory in trajectories:
        pending.append(trajectory)
        if len(pending) >= batch_size:
            flush()
    flush()
    return rows
