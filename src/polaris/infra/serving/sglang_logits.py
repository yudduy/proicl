from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - only present inside the SGLang runtime image.
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
except Exception:  # pragma: no cover - local CPU/unit environment.
    CustomLogitProcessor = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class SegmentScoreRequest:
    input_ids: list[int]
    logprob_start_len: int
    target_len: int
    temperature: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": self.temperature,
            },
            "temp_scaled_logprobs": True,
            "return_logprob": True,
            "logprob_start_len": self.logprob_start_len,
            "top_logprobs_num": 0,
        }


@dataclass(frozen=True)
class NextTokenScoreRequest:
    input_ids: list[int]
    target_token_id: int
    temperature: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": self.temperature,
            },
            "return_logprob": True,
            "top_logprobs_num": 0,
            "token_ids_logprob": [self.target_token_id],
        }


@dataclass(frozen=True)
class ForcedSegmentScoreRequest:
    input_ids: list[int]
    target_ids: list[int]
    temperature: float
    score_path: str
    score_id: str
    custom_logit_processor: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "sampling_params": {
                "max_new_tokens": len(self.target_ids),
                "temperature": 1.0,
                "custom_params": {
                    "target_token_ids": self.target_ids,
                    "temperature": self.temperature,
                    "score_path": self.score_path,
                    "score_id": self.score_id,
                },
            },
            "custom_logit_processor": self.custom_logit_processor,
            "return_logprob": False,
        }


def build_forced_segment_score_request(
    prefix_ids: list[int],
    target_ids: list[int],
    *,
    temperature: float,
    score_path: str,
    score_id: str,
) -> ForcedSegmentScoreRequest:
    if not prefix_ids:
        raise ValueError("prefix_ids must be non-empty")
    if not target_ids:
        raise ValueError("target_ids must be non-empty")
    processor = POLARISForcedTokenLogitProcessor()
    to_str = getattr(processor, "to_str", None)
    if to_str is None:
        raise RuntimeError("SGLang CustomLogitProcessor is unavailable")
    return ForcedSegmentScoreRequest(
        input_ids=list(prefix_ids),
        target_ids=[int(x) for x in target_ids],
        temperature=float(temperature),
        score_path=score_path,
        score_id=score_id,
        custom_logit_processor=to_str(),
    )


def build_next_token_score_request(
    context_ids: list[int],
    target_token_id: int,
    *,
    temperature: float,
) -> NextTokenScoreRequest:
    if not context_ids:
        raise ValueError("context_ids must be non-empty")
    return NextTokenScoreRequest(
        input_ids=list(context_ids),
        target_token_id=int(target_token_id),
        temperature=temperature,
    )


def build_segment_score_request(
    prefix_ids: list[int],
    target_ids: list[int],
    *,
    temperature: float,
) -> SegmentScoreRequest:
    if not prefix_ids:
        raise ValueError("prefix_ids must be non-empty")
    return SegmentScoreRequest(
        input_ids=list(prefix_ids) + list(target_ids),
        logprob_start_len=len(prefix_ids) - 1,
        target_len=len(target_ids),
        temperature=temperature,
    )


def extract_input_token_logprobs(response: dict[str, Any], target_len: int) -> list[float]:
    """Extract target-segment input logprobs from common SGLang response shapes."""
    meta = response.get("meta_info") or response.get("meta") or response
    raw = (
        meta.get("input_token_logprobs")
        or meta.get("input_token_logprobs_val")
        or meta.get("input_logprobs")
        or []
    )
    vals = [_coerce_logprob(item) for item in raw]
    vals = [v for v in vals if v is not None]
    if len(vals) < target_len:
        raise ValueError(
            f"SGLang response returned {len(vals)} input logprobs, need {target_len}"
        )
    return [float(x) for x in vals[-target_len:]]


def extract_output_token_id_logprob(
    response: dict[str, Any], target_token_id: int
) -> float:
    """Extract `/generate` output-token logprob for `token_ids_logprob`."""
    meta = response.get("meta_info") or response.get("meta") or response
    raw = meta.get("output_token_ids_logprobs")
    found = _find_token_logprob(raw, int(target_token_id))
    if found is not None:
        return found

    vals = meta.get("output_token_ids_logprobs_val")
    idxs = meta.get("output_token_ids_logprobs_idx")
    found = _find_token_logprob_from_val_idx(vals, idxs, int(target_token_id))
    if found is not None:
        return found

    raise ValueError(
        f"SGLang response did not include logprob for target token {target_token_id}"
    )


class POLARISForcedTokenLogitProcessor(CustomLogitProcessor):  # type: ignore[misc]
    """SGLang custom-logit processor for MCMC segment scoring.

    It records original target-token scores before forcing the next token. The
    side channel is a JSONL file because SGLang's HTTP response does not expose
    custom processor state.
    """

    def __init__(self) -> None:
        self._positions: dict[str, int] = {}

    def __call__(self, logits, custom_param_list):  # pragma: no cover - GPU hook
        import torch
        import torch.nn.functional as F

        if custom_param_list is None:
            return logits
        for row_idx, params in enumerate(custom_param_list):
            if not params:
                continue
            targets = [int(x) for x in params["target_token_ids"]]
            score_id = str(params["score_id"])
            pos = self._positions.get(score_id, 0)
            if pos >= len(targets):
                continue
            target = targets[pos]
            temperature = float(params.get("temperature", 1.0))
            score_path = str(params["score_path"])
            row = logits[row_idx].float()
            lp_norm = F.log_softmax(row / temperature, dim=-1)[target]
            lp_unnorm = (1.0 / temperature) * F.log_softmax(row, dim=-1)[target]
            os.makedirs(os.path.dirname(score_path), exist_ok=True)
            with open(score_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "score_id": score_id,
                            "position": pos,
                            "target_token_id": target,
                            "lp_norm": float(lp_norm.detach().cpu()),
                            "lp_unnorm": float(lp_unnorm.detach().cpu()),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            self._positions[score_id] = pos + 1
            logits[row_idx].fill_(-torch.inf)
            logits[row_idx, target] = 0.0
        return logits


def _coerce_logprob(item: Any) -> float | None:
    if item is None:
        return None
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("logprob", "logp", "value"):
            if key in item and item[key] is not None:
                return float(item[key])
        return None
    if isinstance(item, (list, tuple)) and item:
        if isinstance(item[0], (int, float)):
            return float(item[0])
        if len(item) > 1 and isinstance(item[1], (int, float)):
            return float(item[1])
    return None


def _find_token_logprob(item: Any, target_token_id: int) -> float | None:
    if item is None:
        return None
    if isinstance(item, dict):
        token_id = item.get("token_id", item.get("token"))
        logprob = item.get("logprob", item.get("logp", item.get("value")))
        if token_id == target_token_id and logprob is not None:
            return float(logprob)
        for value in item.values():
            found = _find_token_logprob(value, target_token_id)
            if found is not None:
                return found
        return None
    if isinstance(item, (list, tuple)):
        if (
            len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], int)
            and item[1] == target_token_id
        ):
            return float(item[0])
        for value in item:
            found = _find_token_logprob(value, target_token_id)
            if found is not None:
                return found
    return None


def _find_token_logprob_from_val_idx(
    vals: Any, idxs: Any, target_token_id: int
) -> float | None:
    if vals is None or idxs is None:
        return None
    if isinstance(idxs, int) and isinstance(vals, (int, float)):
        return float(vals) if idxs == target_token_id else None
    if isinstance(vals, (list, tuple)) and isinstance(idxs, (list, tuple)):
        for val, idx in zip(vals, idxs):
            found = _find_token_logprob_from_val_idx(val, idx, target_token_id)
            if found is not None:
                return found
    return None
