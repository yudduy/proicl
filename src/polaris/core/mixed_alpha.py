from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaSchedule:
    """Pre-registered policy mapping (prompt_id, sample_index) -> alpha.

    `policy_id` is recorded in the run manifest so the alpha schedule is
    unambiguous when reading artifacts later (proposal §"Drift-Lock Protocol"
    forbids unrecorded changes to locked hyperparameters).

    Two policies are pre-registered for the v1 MATH500 slice:
      - fixed_alpha_4:    every sample uses alpha=4 (RWS carryover baseline)
      - mixed_alpha_4_1:  half samples alpha=4, half alpha=1, split by parity
                          of sample_index. This is the POLARIS diversity-
                          preservation schedule (proposal §"core mechanism").

    Determinism: alpha is a pure function of (prompt_id, sample_index).
    No RNG involved — the schedule is reproducible across runs.
    """

    policy_id: str
    alphas: tuple[float, ...]  # cycled over sample_index

    def alpha(self, prompt_id: str, sample_index: int) -> float:
        return self.alphas[sample_index % len(self.alphas)]


FIXED_ALPHA_4 = AlphaSchedule(policy_id="fixed_alpha_4", alphas=(4.0,))
MIXED_ALPHA_4_1 = AlphaSchedule(
    policy_id="mixed_alpha_4_1",
    alphas=(4.0, 1.0),
)
# Decaying-α schedule for PROPOSAL §6.3 diversity-preservation ablation.
# Each prompt's per-sample-index budget walks 4 → 3 → 2 → 1, then cycles.
# Pre-register the schedule before any held-out evaluation that uses it.
DECAYING_ALPHA_4_TO_1 = AlphaSchedule(
    policy_id="decaying_alpha_4_to_1",
    alphas=(4.0, 3.0, 2.0, 1.0),
)
