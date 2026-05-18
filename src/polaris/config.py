from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

MODEL_ID = "Qwen/Qwen2.5-Math-7B"
MODEL_FAMILY = "qwen_math"
SEED = 0

# RWS sharpening hyperparameters (proposal §3 "Baseline sharpening setting").
ALPHA = 4
PROPOSAL_TEMPERATURE = 1 / ALPHA
MCMC_STEPS = 10
MAX_NEW_TOKENS = 3072
# RWS's actual knob is block_num; block_size is derived. See
# polaris.vendored.rws.power_samp_utils.mcmc_power_samp(..., block_num=16).
MCMC_BLOCK_NUM = 16
MCMC_BLOCK_SIZE = MAX_NEW_TOKENS // MCMC_BLOCK_NUM

BOOTSTRAP_RESAMPLES = 1000

INPUT_PRICE_PER_MILLION_TOKENS = 0.20
OUTPUT_PRICE_PER_MILLION_TOKENS = 0.60


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int

    @property
    def dollars(self) -> float:
        return (
            self.input_tokens * INPUT_PRICE_PER_MILLION_TOKENS
            + self.output_tokens * OUTPUT_PRICE_PER_MILLION_TOKENS
        ) / 1_000_000


def estimate_cost(input_tokens: int, output_tokens: int) -> CostEstimate:
    return CostEstimate(input_tokens=input_tokens, output_tokens=output_tokens)


# Direct sampling baselines plus the public ProRL/BroRL checkpoint audit matrix.
_PRORL_RECOVERY_TRACKS = (
    "math500",
    "gpqa_diamond",
    "reasoning_gym_boxnet",
    "reasoning_gym_graph_color",
    "reasoning_gym_family_relationships",
)

MODEL_REGISTRY: dict[str, dict] = {
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "family": "qwen_base",
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "default_tracks": ("humaneval_plus",),  # RWS HumanEval direct match
        "revision": None,
        "revision_commit": None,
    },
    "qwen2.5-math-7b": {
        "hf_id": "Qwen/Qwen2.5-Math-7B",
        "family": "qwen_math",
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "default_tracks": ("math500",),  # RWS MATH500 direct match
        "revision": None,
        "revision_commit": None,
    },
    "deepseek-math-7b": {
        "hf_id": "deepseek-ai/deepseek-math-7b-base",
        "family": "deepseek_math",
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "default_tracks": ("math500",),  # SPS direct match
        "revision": None,
        "revision_commit": None,
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "family": "deepseek_r1_distill_qwen",
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "default_tracks": _PRORL_RECOVERY_TRACKS,
        "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        "revision_commit": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    },
    "nemotron-prorl-v1": {
        "hf_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "family": "nemotron_reasoning_qwen",
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "default_tracks": _PRORL_RECOVERY_TRACKS,
        "revision": "v1",
        "revision_commit": "b89048893f95246c6b5749b287f0049e6df42ee9",
    },
    "nemotron-prorl-v2": {
        "hf_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "family": "nemotron_reasoning_qwen",
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "default_tracks": _PRORL_RECOVERY_TRACKS,
        "revision": "main",
        "revision_commit": "c62ac5e70bd578a9235aa9d8e11fff2f1f63d4a0",
        "artifact_etags": {
            "config.json": "4b1c5c5f54f6c0dc01260820fafb441453e30ff5",
            "generation_config.json": "57f3091b2ddba5b8c1d0cc608e5ad5590482cd47",
            "model-00001-of-00002.safetensors": (
                "f477d1409b354cf98084b7798ab25c42121e9e39c2d207fe9d8693506acf4c78"
            ),
            "model-00002-of-00002.safetensors": (
                "4a0531bf3db7fa4e94f863fdfbe71cb0083964eecc4b5ddc474863d26074f40c"
            ),
        },
    },
    "nemotron-brorl": {
        "hf_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "family": "nemotron_reasoning_qwen",
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "default_tracks": _PRORL_RECOVERY_TRACKS,
        "revision": "brorl",
        "revision_commit": "3441fcdf8c6e81a2959e6352ff50122e3c677d72",
    },
}

# Three task families per PROPOSAL §5.1. Track configs are populated when
# the loader+verifier pair lands for each track (v1=math500 wired now,
# v4=humaneval_plus, v5=gpqa_diamond).
TRACK_REGISTRY: dict[str, dict] = {
    "math500": {
        "regime": "diminished",
        "primary_model": "qwen2.5-math-7b",
        "verifier_id": "math/sympy-equivalence-v1",
        "dataset_module": "polaris.evals.datasets.math500",
        "verifier_module": "polaris.evals.verifiers.math",
        "inference_time_verifier": True,
    },
    "humaneval_plus": {
        "regime": "sustained",
        "primary_model": "qwen2.5-7b",
        "verifier_id": "code/humaneval-plus-v1",
        "dataset_module": "polaris.evals.datasets.humaneval_plus",
        "verifier_module": "polaris.evals.verifiers.code",
        "inference_time_verifier": True,
    },
    "gpqa_diamond": {
        "regime": "diminished_or_plateau",
        "primary_model": "qwen2.5-7b",
        "verifier_id": "gpqa/answer-key-oracle-v1",
        "dataset_module": "polaris.evals.datasets.gpqa_diamond",
        "verifier_module": "polaris.evals.verifiers.gpqa",
        "inference_time_verifier": False,  # offline only per PROPOSAL §4
    },
    "reasoning_gym_boxnet": {
        "regime": "sustained",
        "primary_model": "deepseek-r1-distill-qwen-1.5b",
        "verifier_id": "reasoning-gym/score-answer-v1",
        "dataset_module": "polaris.evals.datasets.reasoning_gym",
        "verifier_module": "polaris.evals.verifiers.reasoning_gym",
        "inference_time_verifier": True,
    },
    "reasoning_gym_graph_color": {
        "regime": "sustained",
        "primary_model": "deepseek-r1-distill-qwen-1.5b",
        "verifier_id": "reasoning-gym/score-answer-v1",
        "dataset_module": "polaris.evals.datasets.reasoning_gym",
        "verifier_module": "polaris.evals.verifiers.reasoning_gym",
        "inference_time_verifier": True,
    },
    "reasoning_gym_family_relationships": {
        "regime": "sustained",
        "primary_model": "deepseek-r1-distill-qwen-1.5b",
        "verifier_id": "reasoning-gym/score-answer-v1",
        "dataset_module": "polaris.evals.datasets.reasoning_gym",
        "verifier_module": "polaris.evals.verifiers.reasoning_gym",
        "inference_time_verifier": True,
    },
}
