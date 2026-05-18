from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BASE_MODEL_KEY = "deepseek-r1-distill-qwen-1.5b"
PRORL_V1_MODEL_KEY = "nemotron-prorl-v1"
PRORL_V2_MODEL_KEY = "nemotron-prorl-v2"
BRO_RL_MODEL_KEY = "nemotron-brorl"
PRORL_RECOVERY_CHECKPOINTS = (
    BASE_MODEL_KEY,
    PRORL_V1_MODEL_KEY,
    PRORL_V2_MODEL_KEY,
    BRO_RL_MODEL_KEY,
)

TRANSPLANT_THRESHOLD_PP = 10.0
HIGH_BASE_LOGPROB_QUANTILE = 0.75


@dataclass(frozen=True)
class ReplicationGate:
    gate_id: str
    model_key: str
    track: str
    condition: str
    alpha: float
    max_new_tokens: int
    block_num: int
    n_mcmc: int
    target_accuracy: float
    tolerance: float
    source: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    def accepts(self, observed_accuracy: float) -> bool:
        return abs(float(observed_accuracy) - self.target_accuracy) <= self.tolerance


KARAN_DU_REPLICATION_GATE = ReplicationGate(
    gate_id="karan_du_math500_qwen2.5_math_7b_power_sampling_v1",
    model_key="qwen2.5-math-7b",
    track="math500",
    condition="single_prompt_power",
    alpha=4.0,
    max_new_tokens=3072,
    block_num=16,
    n_mcmc=10,
    target_accuracy=0.748,
    tolerance=0.02,
    source="Reasoning with Sampling published MATH500 Power Sampling result",
)


def recoverable_fraction(
    *,
    base_accuracy: float,
    frozen_inference_accuracy: float,
    trained_accuracy: float,
) -> float:
    """Fraction of the training gain recovered by frozen-base inference.

    RF = (A_frozen - A_base) / (A_trained - A_base), clipped to [0, 1].
    A non-positive training gap is undefined for this audit.
    """

    denom = float(trained_accuracy) - float(base_accuracy)
    if denom <= 0.0:
        raise ValueError("trained_accuracy must exceed base_accuracy")
    raw = (float(frozen_inference_accuracy) - float(base_accuracy)) / denom
    return max(0.0, min(1.0, raw))


def memory_transplant_claim_passes(
    *, transplant_pass_at_16: float, control_pass_at_16: float
) -> bool:
    delta_pp = 100.0 * (float(transplant_pass_at_16) - float(control_pass_at_16))
    return delta_pp >= TRANSPLANT_THRESHOLD_PP
