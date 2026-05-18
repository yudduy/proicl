"""Break-even N: deployment volume where POLARIS amortizes vs GRPO (PROPOSAL §5.4).

```text
RolloutTotal(POLARIS, N) = ArchiveConstructionRollouts + N * InferenceRolloutsPerQuery
RolloutTotal(GRPO,    N) = TrainingRollouts + N
```

Break-even N* is the smallest non-negative integer N such that POLARIS's
total rollouts are ≤ GRPO's at the same accuracy. Used in `metrics.json`
on any run that targets a published GRPO reference.
"""

from __future__ import annotations

import math


def break_even_n(
    *,
    archive_rollouts: int,
    inference_rollouts_per_query: int,
    grpo_training_rollouts: int,
    grpo_inference_rollouts_per_query: int = 1,
) -> int | None:
    """Largest N at which POLARIS total rollouts ≤ GRPO total rollouts.

    POLARIS pays `archive_rollouts` up front and `inference_rollouts_per_query`
    per query; GRPO pays `grpo_training_rollouts` up front and
    `grpo_inference_rollouts_per_query` per query. POLARIS wins at low N
    (where GRPO's training cost dominates) and loses at high N (where
    POLARIS's per-query rate dominates).

    Returns the largest non-negative N where POLARIS still wins, or `None`
    if POLARIS never wins (archive cost exceeds GRPO's total even at N=0).
    """
    archive_gap = grpo_training_rollouts - archive_rollouts
    per_query_gap = inference_rollouts_per_query - grpo_inference_rollouts_per_query
    if archive_gap < 0:
        return None  # POLARIS upfront already worse than GRPO total at N=0.
    if per_query_gap <= 0:
        # POLARIS is no worse per-query AND upfront cheaper → wins for all N.
        return math.inf  # type: ignore[return-value]
    return max(0, math.floor(archive_gap / per_query_gap))
