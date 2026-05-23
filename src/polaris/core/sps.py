from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SPSConfig:
    """Scalable Power Sampling hyperparameters.

    The paper's main setting is alpha=4 with block_size=192 and
    top_k=rollouts_per_candidate=8. The runner derives block size from
    max_new_tokens/block_num so small smoke runs can stay cheap.
    """

    top_k: int = 8
    candidate_pool_size: int = 8
    rollouts_per_candidate: int = 8
    rollout_horizon: int | None = None
    jackknife: bool = True


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    return logsumexp(values) - math.log(len(values))


def softmax_from_log_weights(log_weights: Sequence[float]) -> list[float]:
    z = logsumexp(log_weights)
    if z == -math.inf:
        n = len(log_weights)
        return [1.0 / n for _ in log_weights] if n else []
    return [math.exp(w - z) for w in log_weights]


def _normalize_nonnegative(values: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(v)) for v in values]
    total = sum(clipped)
    if total > 0.0:
        return [v / total for v in clipped]
    n = len(values)
    return [1.0 / n for _ in values] if n else []


def sps_candidate_probabilities(
    *,
    candidate_logps: Sequence[float],
    rollout_logps_by_candidate: Sequence[Sequence[float]],
    alpha: float,
    jackknife: bool = True,
) -> list[float]:
    """Return SPS probabilities over candidate blocks.

    `candidate_logps[i]` is log p(block_i | prefix). Each rollout logp is
    log p(future | prefix, block_i) under the base model. The returned
    distribution is the jackknife-corrected finite-candidate approximation to
    p_pow over the candidate set.
    """

    if alpha <= 1.0:
        return softmax_from_log_weights(candidate_logps)
    if len(candidate_logps) != len(rollout_logps_by_candidate):
        raise ValueError("candidate_logps and rollout_logps_by_candidate length mismatch")
    if not candidate_logps:
        return []

    rollout_counts = {len(rows) for rows in rollout_logps_by_candidate}
    if rollout_counts == {0}:
        return softmax_from_log_weights([alpha * lp for lp in candidate_logps])
    if len(rollout_counts) != 1:
        raise ValueError("every candidate must have the same rollout count")
    m = rollout_counts.pop()
    if m == 0:
        return softmax_from_log_weights([alpha * lp for lp in candidate_logps])

    scaled_rollouts = [
        [(alpha - 1.0) * float(logp) for logp in rollout_logps]
        for rollout_logps in rollout_logps_by_candidate
    ]
    log_zeta = [logmeanexp(rows) for rows in scaled_rollouts]
    raw_probs = softmax_from_log_weights(
        [alpha * float(block_lp) + z for block_lp, z in zip(candidate_logps, log_zeta)]
    )
    if not jackknife or m <= 1:
        return raw_probs

    loo_probs_sum = [0.0 for _ in candidate_logps]
    for heldout_idx in range(m):
        loo_log_zeta = [
            logmeanexp([value for idx, value in enumerate(rows) if idx != heldout_idx])
            for rows in scaled_rollouts
        ]
        probs = softmax_from_log_weights(
            [
                alpha * float(block_lp) + z
                for block_lp, z in zip(candidate_logps, loo_log_zeta)
            ]
        )
        for i, prob in enumerate(probs):
            loo_probs_sum[i] += prob

    corrected = [
        m * raw - ((m - 1.0) / m) * loo_sum
        for raw, loo_sum in zip(raw_probs, loo_probs_sum)
    ]
    return _normalize_nonnegative(corrected)
