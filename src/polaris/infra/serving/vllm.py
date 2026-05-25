"""vLLM V0-backed sampler/scorer.

RWS correctness needs full-vocabulary logits at each target token. vLLM V0
supports per-request logits processors, so POLARIS scores fixed segments by
forcing the known target tokens while recording the original raw logits.

The `native_segment` scoring mode is the upstreamable target surface: a vLLM
fork/branch may expose `LLM.score_sequences(...)`, which returns teacher-forced
target-token logprobs without routing through generation logits processors.
HF/RWS remains the oracle; this path is scale-valid only after the parity gates
write passing calibration artifacts.

Source checks:
- vLLM V1 rejects per-request logits processors.
- vLLM rejects `SamplingParams(max_tokens=0)`, so scoring uses forced decode.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import os
import random
import time
import math
from dataclasses import dataclass
from typing import Any

from polaris.config import (
    MAX_NEW_TOKENS,
    MCMC_BLOCK_NUM,
    MCMC_STEPS,
    MODEL_ID,
    PROPOSAL_TEMPERATURE,
    SEED,
    estimate_cost,
)
from polaris.core.sps import SPSConfig, sps_candidate_probabilities
from polaris.infra.serving import ScoreBatch

VLLM_SCORING_MODES = ("forced_decode_v0", "native_segment")


@dataclass
class Generation:
    generation: str
    prompt_text: str
    response_contains_prompt: bool
    prompt_token_count: int
    generation_token_count: int
    wall_clock_seconds: float
    estimated_dollar_cost: float
    acceptance_ratio: float | None = None
    token_ids: list[int] | None = None
    logprobs_norm: list[float] | None = None
    logprobs_unnorm: list[float] | None = None


@dataclass
class _MCMCChainState:
    prompt_text: str
    prefix_ids: list[int]
    gen: list[int]
    log_probs_norm: list[float]
    log_probs_unnorm: list[float]
    rng: random.Random
    attempts: int = 0
    acceptances: int = 0
    active: bool = True


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_batch_limit(*names: str) -> int:
    for name in names:
        value = _env_positive_int(name, 0)
        if value > 0:
            return value
    return 0


def _chunks(seq: list[Any], size: int):
    if size <= 0 or len(seq) <= size:
        yield seq
        return
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


class VLLMForcedTokenProcessor:
    """Record target-token logprobs, then force that token for decode."""

    def __init__(self, target_ids: list[int], temperature: float) -> None:
        self.target_ids = [int(x) for x in target_ids]
        self.temperature = float(temperature)
        self.lp_norm: list[float] = []
        self.lp_unnorm: list[float] = []

    def clone(self) -> "VLLMForcedTokenProcessor":
        return self

    def __call__(self, prompt_token_ids, past_token_ids, logits):
        import torch
        import torch.nn.functional as F

        pos = len(past_token_ids)
        if pos >= len(self.target_ids):
            return logits
        target = self.target_ids[pos]
        row = logits.float()
        self.lp_norm.append(
            float(F.log_softmax(row / self.temperature, dim=-1)[target].detach().cpu())
        )
        self.lp_unnorm.append(
            float(
                (
                    (1.0 / self.temperature)
                    * F.log_softmax(row, dim=-1)[target]
                ).detach().cpu()
            )
        )
        forced = torch.full_like(logits, -torch.inf)
        forced[target] = 0.0
        return forced


class VLLMSamplingRecorderProcessor:
    """Sample from the proposal distribution, record scores, then force token.

    This fuses RWS `generate(..., output_logits=True)` behavior into vLLM:
    the processor observes raw full-vocab logits, samples from
    `softmax(logits / temperature)`, records both proposal and target scores,
    and forces the sampled token so vLLM emits exactly the recorded path.
    """

    def __init__(self, temperature: float, seed: int = 0) -> None:
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.token_ids: list[int] = []
        self.lp_norm: list[float] = []
        self.lp_unnorm: list[float] = []
        self._generators = {}

    def clone(self) -> "VLLMSamplingRecorderProcessor":
        return self

    def __call__(self, prompt_token_ids, past_token_ids, logits):
        import torch
        import torch.nn.functional as F

        row = logits.float()
        lp_norm = F.log_softmax(row / self.temperature, dim=-1)
        lp_unnorm = (1.0 / self.temperature) * F.log_softmax(row, dim=-1)
        device_key = str(row.device)
        generator = self._generators.get(device_key)
        if generator is None:
            generator = torch.Generator(device=row.device)
            generator.manual_seed(self.seed)
            self._generators[device_key] = generator
        probs = torch.exp(lp_norm)
        target = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        self.token_ids.append(target)
        self.lp_norm.append(float(lp_norm[target].detach().cpu()))
        self.lp_unnorm.append(float(lp_unnorm[target].detach().cpu()))
        forced = torch.full_like(logits, -torch.inf)
        forced[target] = 0.0
        return forced


class VLLMValidTokenMaskProcessor:
    """Suppress model-head padding rows that are not real tokenizer ids."""

    def __init__(self, valid_vocab_size: int) -> None:
        self.valid_vocab_size = int(valid_vocab_size)

    def clone(self) -> "VLLMValidTokenMaskProcessor":
        return self

    def __call__(self, prompt_token_ids, past_token_ids, logits):
        if self.valid_vocab_size > 0 and logits.shape[-1] > self.valid_vocab_size:
            logits = logits.clone()
            logits[..., self.valid_vocab_size :] = -float("inf")
        return logits


class VLLMGenerator:
    """vLLM V0 sampler with source-audited forced-token scoring."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        seed: int = SEED,
        tensor_parallel_size: int = 1,
        dtype: str = "float32",
        model_impl: str = "transformers",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        enforce_eager: bool = True,
        disable_async_output_proc: bool = True,
        local_files_only: bool = False,
        fused_sampling_recorder: bool = False,
        revision: str | None = None,
        scoring_mode: str = "forced_decode_v0",
        parity_artifact_path: str | None = None,
        enable_prefix_caching: bool = True,
        reset_prefix_cache_for_scoring: bool = True,
    ) -> None:
        if scoring_mode not in VLLM_SCORING_MODES:
            raise ValueError(
                f"unknown vLLM scoring mode {scoring_mode!r}; "
                f"expected one of {VLLM_SCORING_MODES}"
            )
        os.environ.setdefault("VLLM_USE_V1", "0")
        import transformers
        from vllm import LLM, SamplingParams

        random.seed(seed)
        self.model_id = model_id
        self.revision = revision
        self.seed = seed
        self.dtype = dtype
        self.model_impl = model_impl
        self.fused_sampling_recorder = fused_sampling_recorder
        self.scoring_mode = scoring_mode
        self.parity_artifact_path = parity_artifact_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.reset_prefix_cache_for_scoring = bool(reset_prefix_cache_for_scoring)
        self.SamplingParams = SamplingParams
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            self.valid_vocab_size = int(len(self.tokenizer))
        except TypeError:
            self.valid_vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or 0)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "tokenizer": model_id,
            "dtype": dtype,
            "trust_remote_code": False,
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed,
            "enable_prefix_caching": self.enable_prefix_caching,
            "model_impl": model_impl,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "disable_async_output_proc": disable_async_output_proc,
        }
        if revision is not None:
            kwargs["revision"] = revision
            kwargs["tokenizer_revision"] = revision
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)

    def runtime_metadata(self) -> dict[str, Any]:
        try:
            vllm_version = importlib_metadata.version("vllm")
        except importlib_metadata.PackageNotFoundError:
            vllm_version = None
        return {
            "backend": "vllm",
            "vllm_scoring_mode": self.scoring_mode,
            "vllm_version": vllm_version,
            "vllm_commit": os.environ.get("VLLM_COMMIT")
            or os.environ.get("VLLM_GIT_COMMIT"),
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1"),
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "model_id": self.model_id,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "dtype": self.dtype,
            "model_impl": self.model_impl,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "prefix_caching": self.enable_prefix_caching,
            "reset_prefix_cache_for_scoring": self.reset_prefix_cache_for_scoring,
            "sps_vllm_batch_size": _env_batch_limit(
                "PROICL_SPS_VLLM_BATCH_SIZE",
                "SPS_VLLM_BATCH_SIZE",
            )
            or None,
            "native_segment_available": hasattr(self.llm, "score_sequences"),
            "parity_artifact_path": self.parity_artifact_path,
            "tokenizer_special_tokens": {
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            },
            "valid_vocab_size": self.valid_vocab_size,
        }

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _sampling_params(self, **kwargs):
        return self.SamplingParams(**kwargs)

    def _with_valid_token_mask(self, processors: list[Any] | None = None) -> list[Any]:
        return [VLLMValidTokenMaskProcessor(self.valid_vocab_size), *(processors or [])]

    def _validate_token_ids(self, token_ids: list[int], *, context: str) -> None:
        if self.valid_vocab_size <= 0:
            return
        invalid = [
            int(x)
            for x in token_ids
            if int(x) < 0 or int(x) >= self.valid_vocab_size
        ]
        if invalid:
            sample = invalid[:8]
            raise ValueError(
                f"{context} contains token ids outside tokenizer vocabulary: "
                f"valid_vocab_size={self.valid_vocab_size} invalid_sample={sample}"
            )

    def _generate_ids(
        self,
        prefix_ids: list[int],
        *,
        temperature: float,
        max_new_tokens: int,
        do_sample: bool,
        exact_length: bool = False,
    ) -> list[int]:
        self._validate_token_ids(prefix_ids, context="generation prefix_ids")
        params_kwargs: dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "top_p": 1.0,
            "ignore_eos": exact_length,
            "skip_special_tokens": False,
            "detokenize": False,
        }
        if exact_length:
            params_kwargs["min_tokens"] = max_new_tokens
        if do_sample:
            params_kwargs["temperature"] = temperature
            params_kwargs["logits_processors"] = self._with_valid_token_mask()
        else:
            params_kwargs["temperature"] = 0.0
            params_kwargs["logits_processors"] = self._with_valid_token_mask()
        params = self._sampling_params(**params_kwargs)
        outputs = self.llm.generate(
            prompts=[{"prompt_token_ids": list(prefix_ids)}],
            sampling_params=params,
            use_tqdm=False,
        )
        return [int(x) for x in outputs[0].outputs[0].token_ids]

    def _generate_ids_batch(
        self,
        prefix_ids_batch: list[list[int]],
        *,
        temperature: float,
        max_new_tokens: int | list[int],
        seed_offsets: list[int] | None = None,
        seed_base: int | None = None,
        exact_length: bool = False,
    ) -> list[list[int]]:
        n = len(prefix_ids_batch)
        if n == 0:
            return []
        if isinstance(max_new_tokens, int):
            lengths = [int(max_new_tokens) for _ in prefix_ids_batch]
        else:
            lengths = [int(x) for x in max_new_tokens]
            if len(lengths) != n:
                raise ValueError("max_new_tokens length must match prefix batch length")
        if seed_offsets is None:
            offsets = list(range(n))
        else:
            offsets = [int(x) for x in seed_offsets]
            if len(offsets) != n:
                raise ValueError("seed_offsets length must match prefix batch length")

        base_seed = self.seed if seed_base is None else int(seed_base)
        results: list[list[int]] = [[] for _ in prefix_ids_batch]
        scheduled: list[tuple[int, int]] = []
        prompts: list[dict[str, list[int]]] = []
        params_list: list[Any] = []

        for row_idx, (prefix_ids, length, offset) in enumerate(
            zip(prefix_ids_batch, lengths, offsets)
        ):
            self._validate_token_ids(
                list(prefix_ids),
                context=f"generation prefix_ids_batch[{row_idx}]",
            )
            if length < 0:
                raise ValueError(f"max_new_tokens must be >= 0, got {length}")
            if length == 0:
                continue
            params = self._sampling_params(
                max_tokens=length,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                seed=base_seed + offset,
                ignore_eos=exact_length,
                skip_special_tokens=False,
                detokenize=False,
                logits_processors=self._with_valid_token_mask(),
            )
            if exact_length:
                params.min_tokens = length
            prompts.append({"prompt_token_ids": list(prefix_ids)})
            params_list.append(params)
            scheduled.append((row_idx, length))

        if not scheduled:
            return results

        batch_limit = _env_batch_limit(
            "PROICL_SPS_VLLM_BATCH_SIZE",
            "SPS_VLLM_BATCH_SIZE",
        )
        outputs = []
        indices = list(range(len(prompts)))
        for chunk in _chunks(indices, batch_limit):
            outputs.extend(
                self.llm.generate(
                    prompts=[prompts[idx] for idx in chunk],
                    sampling_params=[params_list[idx] for idx in chunk],
                    use_tqdm=False,
                )
            )
        for output, (row_idx, expected_len) in zip(outputs, scheduled):
            token_ids = [int(x) for x in output.outputs[0].token_ids]
            if exact_length and len(token_ids) != expected_len:
                raise RuntimeError(
                    "vLLM native sampler returned unexpected token count: "
                    f"tokens={len(token_ids)} expected={expected_len}"
                )
            results[row_idx] = token_ids
        return results

    def _sample_ids_with_scores(
        self,
        prefix_ids: list[int],
        *,
        temperature: float,
        max_new_tokens: int,
        seed_offset: int = 0,
    ) -> tuple[list[int], list[float], list[float]]:
        return self._sample_ids_with_scores_batch(
            [prefix_ids],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed_offsets=[seed_offset],
        )[0]

    def _sample_ids_with_scores_batch_fused(
        self,
        prefix_ids_batch: list[list[int]],
        *,
        temperature: float,
        max_new_tokens: int | list[int],
        seed_offsets: list[int] | None = None,
        seed_base: int | None = None,
        exact_length: bool = False,
    ) -> list[tuple[list[int], list[float], list[float]]]:
        """Sample a batch while recording proposal and target logprobs."""
        n = len(prefix_ids_batch)
        if n == 0:
            return []
        if isinstance(max_new_tokens, int):
            lengths = [int(max_new_tokens) for _ in prefix_ids_batch]
        else:
            lengths = [int(x) for x in max_new_tokens]
            if len(lengths) != n:
                raise ValueError("max_new_tokens length must match prefix batch length")
        if seed_offsets is None:
            offsets = list(range(n))
        else:
            offsets = [int(x) for x in seed_offsets]
            if len(offsets) != n:
                raise ValueError("seed_offsets length must match prefix batch length")

        base_seed = self.seed if seed_base is None else int(seed_base)
        results: list[tuple[list[int], list[float], list[float]]] = [
            ([], [], []) for _ in prefix_ids_batch
        ]
        scheduled: list[tuple[int, int, VLLMSamplingRecorderProcessor]] = []
        prompts: list[dict[str, list[int]]] = []
        params_list: list[Any] = []

        for row_idx, (prefix_ids, length, offset) in enumerate(
            zip(prefix_ids_batch, lengths, offsets)
        ):
            self._validate_token_ids(
                list(prefix_ids),
                context=f"fused generation prefix_ids_batch[{row_idx}]",
            )
            if length < 0:
                raise ValueError(f"max_new_tokens must be >= 0, got {length}")
            if length == 0:
                continue
            processor = VLLMSamplingRecorderProcessor(
                temperature=temperature,
                seed=base_seed + offset,
            )
            params = self._sampling_params(
                max_tokens=length,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                ignore_eos=exact_length,
                skip_special_tokens=False,
                detokenize=False,
                logits_processors=self._with_valid_token_mask([processor]),
            )
            if exact_length:
                params.min_tokens = length
            prompts.append({"prompt_token_ids": list(prefix_ids)})
            params_list.append(params)
            scheduled.append((row_idx, length, processor))

        if not scheduled:
            return results

        batch_limit = _env_batch_limit(
            "PROICL_SPS_VLLM_BATCH_SIZE",
            "SPS_VLLM_BATCH_SIZE",
        )
        outputs = []
        indices = list(range(len(prompts)))
        for chunk in _chunks(indices, batch_limit):
            outputs.extend(
                self.llm.generate(
                    prompts=[prompts[idx] for idx in chunk],
                    sampling_params=[params_list[idx] for idx in chunk],
                    use_tqdm=False,
                )
            )
        unmatched = set(range(len(scheduled)))
        for output_idx, output in enumerate(outputs):
            token_ids = [int(x) for x in output.outputs[0].token_ids]
            exact_matches = [
                idx
                for idx in unmatched
                if scheduled[idx][2].token_ids[: len(token_ids)] == token_ids
            ]
            if len(exact_matches) == 1:
                scheduled_idx = exact_matches[0]
            elif output_idx in unmatched and output_idx in exact_matches:
                scheduled_idx = output_idx
            else:
                debug = []
                for idx in list(unmatched)[:4]:
                    row_idx, expected_len, processor = scheduled[idx]
                    first_mismatch = _first_mismatch(processor.token_ids, token_ids)
                    debug.append(
                        {
                            "scheduled_idx": idx,
                            "row_idx": row_idx,
                            "recorded_len": len(processor.token_ids),
                            "emitted_len": len(token_ids),
                            "expected_len": expected_len,
                            "first_mismatch": first_mismatch,
                            "recorded_head": processor.token_ids[:8],
                        }
                    )
                raise RuntimeError(
                    "vLLM sampled-token recorder could not be matched to emitted tokens: "
                    f"output_idx={output_idx} emitted_head={token_ids[:8]} "
                    f"candidate_debug={debug}"
                )
            unmatched.remove(scheduled_idx)
            row_idx, expected_len, processor = scheduled[scheduled_idx]
            if (
                len(processor.lp_norm) < len(token_ids)
                or len(processor.lp_unnorm) < len(token_ids)
            ):
                raise RuntimeError(
                    "vLLM sampled-token recorder did not record every emitted token: "
                    f"recorded_norm={len(processor.lp_norm)} "
                    f"recorded_unnorm={len(processor.lp_unnorm)} "
                    f"emitted={len(token_ids)}"
                )
            if exact_length and len(token_ids) != expected_len:
                raise RuntimeError(
                    "vLLM sampled-token recorder emitted unexpected token count: "
                    f"emitted={len(token_ids)} expected={expected_len}"
                )
            results[row_idx] = (
                token_ids,
                list(processor.lp_norm[: len(token_ids)]),
                list(processor.lp_unnorm[: len(token_ids)]),
            )
        return results

    def _sample_ids_with_scores_batch(
        self,
        prefix_ids_batch: list[list[int]],
        *,
        temperature: float,
        max_new_tokens: int | list[int],
        seed_offsets: list[int] | None = None,
        seed_base: int | None = None,
        exact_length: bool = False,
    ) -> list[tuple[list[int], list[float], list[float]]]:
        """Sample with vLLM, then score sampled segments with forced decode.

        The one-pass fused recorder is kept behind an opt-in flag. It is faster
        in principle but currently fails GPU validation on long batched decodes:
        vLLM can diverge from the token forced inside the recorder after ~100
        generated tokens. The default path is the GPU-accepted correctness path.
        """
        if self.fused_sampling_recorder:
            return self._sample_ids_with_scores_batch_fused(
                prefix_ids_batch,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed_offsets=seed_offsets,
                seed_base=seed_base,
                exact_length=exact_length,
            )

        n = len(prefix_ids_batch)
        if n == 0:
            return []
        if isinstance(max_new_tokens, int):
            lengths = [int(max_new_tokens) for _ in prefix_ids_batch]
        else:
            lengths = [int(x) for x in max_new_tokens]
            if len(lengths) != n:
                raise ValueError("max_new_tokens length must match prefix batch length")
        if seed_offsets is None:
            offsets = list(range(n))
        else:
            offsets = [int(x) for x in seed_offsets]
            if len(offsets) != n:
                raise ValueError("seed_offsets length must match prefix batch length")

        token_ids_batch = self._generate_ids_batch(
            prefix_ids_batch,
            temperature=temperature,
            max_new_tokens=lengths,
            seed_offsets=offsets,
            seed_base=seed_base,
            exact_length=exact_length,
        )
        scores = self.score_segments(
            prefix_ids_batch,
            token_ids_batch,
            temperature=temperature,
        )
        results: list[tuple[list[int], list[float], list[float]]] = []
        for row_idx, token_ids in enumerate(token_ids_batch):
            norm = scores.lp_norm_tokens[row_idx]
            unnorm = scores.lp_unnorm_tokens[row_idx]
            if len(norm) != len(token_ids) or len(unnorm) != len(token_ids):
                raise RuntimeError(
                    "vLLM forced scoring returned misaligned token/logprob lengths: "
                    f"tokens={len(token_ids)} norm={len(norm)} unnorm={len(unnorm)}"
                )
            results.append((token_ids, list(norm), list(unnorm)))
        return results

    def generate_greedy(
        self, prompt_text: str, *, max_new_tokens: int = MAX_NEW_TOKENS
    ) -> Generation:
        prefix_ids = self._encode(prompt_text)
        started = time.monotonic()
        token_ids = self._generate_ids(
            prefix_ids,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        elapsed = time.monotonic() - started
        cost = estimate_cost(len(prefix_ids), len(token_ids))
        return Generation(
            generation=self._decode(token_ids),
            prompt_text=prompt_text,
            response_contains_prompt=False,
            prompt_token_count=len(prefix_ids),
            generation_token_count=len(token_ids),
            wall_clock_seconds=elapsed,
            estimated_dollar_cost=cost.dollars,
            acceptance_ratio=None,
            token_ids=token_ids,
        )

    def generate_low_temp(
        self,
        prompt_text: str,
        *,
        temperature: float = PROPOSAL_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self.generate_low_temp_batch(
            [prompt_text],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )[0]

    def generate_low_temp_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float = PROPOSAL_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Generation]:
        if not prompt_texts:
            return []
        if seed_offsets is None:
            offsets = list(range(len(prompt_texts)))
        else:
            offsets = [int(x) for x in seed_offsets]
            if len(offsets) != len(prompt_texts):
                raise ValueError("seed_offsets length must match prompt_texts length")
        prefix_ids_batch = [self._encode(prompt_text) for prompt_text in prompt_texts]
        started = time.monotonic()
        token_ids_batch = self._generate_ids_batch(
            prefix_ids_batch,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed_offsets=offsets,
            seed_base=seed_base,
            exact_length=False,
        )
        elapsed = time.monotonic() - started
        per_candidate_wall = elapsed / len(prompt_texts)
        generations: list[Generation] = []
        for prompt_text, prefix_ids, token_ids in zip(
            prompt_texts, prefix_ids_batch, token_ids_batch
        ):
            cost = estimate_cost(len(prefix_ids), len(token_ids))
            generations.append(
                Generation(
                    generation=self._decode(token_ids),
                    prompt_text=prompt_text,
                    response_contains_prompt=False,
                    prompt_token_count=len(prefix_ids),
                    generation_token_count=len(token_ids),
                    wall_clock_seconds=per_candidate_wall,
                    estimated_dollar_cost=cost.dollars,
                    acceptance_ratio=None,
                    token_ids=token_ids,
                    logprobs_norm=[],
                    logprobs_unnorm=[],
                )
            )
        return generations

    def generate_power(
        self,
        prompt_text: str,
        *,
        temperature: float = PROPOSAL_TEMPERATURE,
        mcmc_steps: int = MCMC_STEPS,
        max_new_tokens: int = MAX_NEW_TOKENS,
        block_num: int = MCMC_BLOCK_NUM,
    ) -> Generation:
        return self.generate_power_batch(
            [prompt_text],
            temperature=temperature,
            mcmc_steps=mcmc_steps,
            max_new_tokens=max_new_tokens,
            block_num=block_num,
        )[0]

    def generate_power_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float = PROPOSAL_TEMPERATURE,
        mcmc_steps: int = MCMC_STEPS,
        max_new_tokens: int = MAX_NEW_TOKENS,
        block_num: int = MCMC_BLOCK_NUM,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Generation]:
        if not prompt_texts:
            return []
        if block_num <= 0:
            raise ValueError(f"block_num must be > 0, got {block_num}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if seed_offsets is None:
            offsets = list(range(len(prompt_texts)))
        else:
            offsets = [int(x) for x in seed_offsets]
            if len(offsets) != len(prompt_texts):
                raise ValueError("seed_offsets length must match prompt_texts length")

        base_seed = self.seed if seed_base is None else int(seed_base)
        prefix_ids_batch = [self._encode(prompt_text) for prompt_text in prompt_texts]
        block_base, block_remainder = divmod(max_new_tokens, block_num)
        block_sizes = [
            block_base + (1 if block_idx < block_remainder else 0)
            for block_idx in range(block_num)
        ]
        states = [
            _MCMCChainState(
                prompt_text=prompt_text,
                prefix_ids=prefix_ids,
                gen=[],
                log_probs_norm=[],
                log_probs_unnorm=[],
                rng=random.Random(base_seed + offsets[chain_idx] * 1_000_003),
            )
            for chain_idx, (prompt_text, prefix_ids) in enumerate(
                zip(prompt_texts, prefix_ids_batch)
            )
        ]
        started = time.monotonic()

        for block_idx, jump_size in enumerate(block_sizes):
            active_pairs = [
                (chain_idx, state)
                for chain_idx, state in enumerate(states)
                if state.active and jump_size > 0
            ]
            if not active_pairs:
                if not any(state.active for state in states):
                    break
                continue

            draft_prefixes = [state.prefix_ids + state.gen for _, state in active_pairs]
            draft_offsets = [
                (block_idx + 1) * 1_000_003 + offsets[chain_idx]
                for chain_idx, _ in active_pairs
            ]
            draft_rows = self._sample_ids_with_scores_batch(
                draft_prefixes,
                temperature=temperature,
                max_new_tokens=jump_size,
                seed_offsets=draft_offsets,
                seed_base=base_seed,
                exact_length=True,
            )
            for (_, state), (draft_ids, draft_norm, draft_unnorm) in zip(
                active_pairs, draft_rows
            ):
                state.gen.extend(draft_ids)
                state.log_probs_norm.extend(draft_norm)
                state.log_probs_unnorm.extend(draft_unnorm)

            for step_idx in range(mcmc_steps):
                proposal_meta: list[tuple[int, _MCMCChainState, int, int]] = []
                proposal_prefixes: list[list[int]] = []
                proposal_lengths: list[int] = []
                proposal_offsets: list[int] = []
                for chain_idx, state in enumerate(states):
                    if not state.active:
                        continue
                    t = len(state.gen)
                    if t == 0:
                        continue
                    state.attempts += 1
                    idx = state.rng.randint(0, t - 1)
                    proposal_meta.append((chain_idx, state, idx, t))
                    proposal_prefixes.append(state.prefix_ids + state.gen[:idx])
                    proposal_lengths.append(t - idx)
                    proposal_offsets.append(
                        10_000_019
                        + (block_idx + 1) * 1_000_003
                        + (step_idx + 1) * 10_007
                        + offsets[chain_idx]
                    )
                if not proposal_meta:
                    continue

                proposal_rows = self._sample_ids_with_scores_batch(
                    proposal_prefixes,
                    temperature=temperature,
                    max_new_tokens=proposal_lengths,
                    seed_offsets=proposal_offsets,
                    seed_base=base_seed,
                    exact_length=True,
                )
                for (_, state, idx, t), (
                    prop_ids,
                    log_prob_prop,
                    target_log_prob_prop,
                ) in zip(proposal_meta, proposal_rows):
                    log_prob_cur = state.log_probs_norm[idx:t]
                    target_log_prob_cur = state.log_probs_unnorm[idx:t]
                    if (
                        len(prop_ids) != t - idx
                        or len(log_prob_prop) != len(prop_ids)
                        or len(target_log_prob_prop) != len(prop_ids)
                    ):
                        raise RuntimeError(
                            "vLLM proposal returned misaligned token/logprob lengths: "
                            f"tokens={len(prop_ids)} norm={len(log_prob_prop)} "
                            f"unnorm={len(target_log_prob_prop)} expected={t - idx}"
                        )
                    log_r = (
                        sum(target_log_prob_prop)
                        + sum(log_prob_cur)
                        - sum(target_log_prob_cur)
                        - sum(log_prob_prop)
                    )
                    if log_r >= 0.0 or math.log(state.rng.random()) < log_r:
                        state.acceptances += 1
                        state.gen = state.gen[:idx] + prop_ids
                        state.log_probs_norm[idx:] = log_prob_prop
                        state.log_probs_unnorm[idx:] = target_log_prob_prop

            eos_id = self.tokenizer.eos_token_id
            for state in states:
                if eos_id in state.gen:
                    eos_idx = state.gen.index(eos_id) + 1
                    state.gen = state.gen[:eos_idx]
                    state.log_probs_norm = state.log_probs_norm[:eos_idx]
                    state.log_probs_unnorm = state.log_probs_unnorm[:eos_idx]
                    state.active = False
            if not any(state.active for state in states):
                break

        elapsed = time.monotonic() - started
        per_candidate_wall = elapsed / len(prompt_texts)
        generations: list[Generation] = []
        for state in states:
            if (
                len(state.gen) != len(state.log_probs_norm)
                or len(state.gen) != len(state.log_probs_unnorm)
            ):
                raise RuntimeError(
                    "vLLM MCMC generated misaligned token/logprob trajectory: "
                    f"tokens={len(state.gen)} norm={len(state.log_probs_norm)} "
                    f"unnorm={len(state.log_probs_unnorm)}"
                )
            cost = estimate_cost(len(state.prefix_ids), len(state.gen))
            generations.append(
                Generation(
                    generation=self._decode(state.gen),
                    prompt_text=state.prompt_text,
                    response_contains_prompt=False,
                    prompt_token_count=len(state.prefix_ids),
                    generation_token_count=len(state.gen),
                    wall_clock_seconds=per_candidate_wall,
                    estimated_dollar_cost=cost.dollars,
                    acceptance_ratio=(
                        state.acceptances / state.attempts if state.attempts else 0.0
                    ),
                    token_ids=state.gen,
                    logprobs_norm=state.log_probs_norm,
                    logprobs_unnorm=state.log_probs_unnorm,
                )
            )
        return generations

    def generate_sps_power(
        self,
        prompt_text: str,
        *,
        temperature: float = PROPOSAL_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        block_num: int = MCMC_BLOCK_NUM,
        top_k: int = 8,
        candidate_pool_size: int = 8,
        rollouts_per_candidate: int = 8,
        rollout_horizon: int | None = None,
        seed_base: int | None = None,
        seed_offset: int = 0,
    ) -> Generation:
        return self.generate_sps_power_batch(
            [prompt_text],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            block_num=block_num,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            rollouts_per_candidate=rollouts_per_candidate,
            rollout_horizon=rollout_horizon,
            seed_base=seed_base,
            seed_offsets=[seed_offset],
        )[0]

    def generate_sps_power_batch(
        self,
        prompt_texts: list[str],
        *,
        temperature: float = PROPOSAL_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        block_num: int = MCMC_BLOCK_NUM,
        top_k: int = 8,
        candidate_pool_size: int = 8,
        rollouts_per_candidate: int = 8,
        rollout_horizon: int | None = None,
        seed_base: int | None = None,
        seed_offsets: list[int] | None = None,
    ) -> list[Generation]:
        """Approximate power sampling with block-level SPS lookahead.

        This implements the batched/block variant of Ji et al. (2026) on top of
        the existing vLLM sampled-segment scoring path. Candidate blocks and
        future rollouts are sampled from the base distribution, scored under the
        base model, then jackknife-corrected into a finite-candidate
        approximation to the alpha-power distribution.
        """
        if not prompt_texts:
            return []
        max_chains_per_batch = _env_positive_int("PROICL_SPS_CHAIN_BATCH_SIZE", 0)
        if max_chains_per_batch > 0 and len(prompt_texts) > max_chains_per_batch:
            generations: list[Generation] = []
            if seed_offsets is None:
                offsets = list(range(len(prompt_texts)))
            else:
                offsets = [int(x) for x in seed_offsets]
                if len(offsets) != len(prompt_texts):
                    raise ValueError("seed_offsets length must match prompt_texts length")
            for start in range(0, len(prompt_texts), max_chains_per_batch):
                stop = start + max_chains_per_batch
                generations.extend(
                    self.generate_sps_power_batch(
                        prompt_texts[start:stop],
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        block_num=block_num,
                        top_k=top_k,
                        candidate_pool_size=candidate_pool_size,
                        rollouts_per_candidate=rollouts_per_candidate,
                        rollout_horizon=rollout_horizon,
                        seed_base=seed_base,
                        seed_offsets=offsets[start:stop],
                    )
                )
            return generations
        if block_num <= 0:
            raise ValueError(f"block_num must be > 0, got {block_num}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if candidate_pool_size <= 0:
            raise ValueError(
                f"candidate_pool_size must be > 0, got {candidate_pool_size}"
            )
        if rollouts_per_candidate < 0:
            raise ValueError(
                f"rollouts_per_candidate must be >= 0, got {rollouts_per_candidate}"
            )
        if seed_offsets is None:
            offsets = list(range(len(prompt_texts)))
        else:
            offsets = [int(x) for x in seed_offsets]
            if len(offsets) != len(prompt_texts):
                raise ValueError("seed_offsets length must match prompt_texts length")

        alpha = 1.0 / float(temperature)
        if alpha <= 1.0:
            return self.generate_low_temp_batch(
                prompt_texts,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                seed_base=seed_base,
                seed_offsets=offsets,
            )

        base_seed = self.seed if seed_base is None else int(seed_base)
        config = SPSConfig(
            top_k=top_k,
            candidate_pool_size=max(candidate_pool_size, top_k),
            rollouts_per_candidate=rollouts_per_candidate,
            rollout_horizon=rollout_horizon,
        )
        prefix_ids_batch = [self._encode(prompt_text) for prompt_text in prompt_texts]
        block_base, block_remainder = divmod(max_new_tokens, block_num)
        block_sizes = [
            block_base + (1 if block_idx < block_remainder else 0)
            for block_idx in range(block_num)
        ]
        states = [
            _MCMCChainState(
                prompt_text=prompt_text,
                prefix_ids=prefix_ids,
                gen=[],
                log_probs_norm=[],
                log_probs_unnorm=[],
                rng=random.Random(base_seed + offsets[chain_idx] * 1_000_003),
            )
            for chain_idx, (prompt_text, prefix_ids) in enumerate(
                zip(prompt_texts, prefix_ids_batch)
            )
        ]
        dollars_by_state = [0.0 for _ in states]
        started = time.monotonic()

        for block_idx, block_size in enumerate(block_sizes):
            if block_size <= 0:
                continue
            active_indices = [
                chain_idx
                for chain_idx, state in enumerate(states)
                if state.active and len(state.gen) < max_new_tokens
            ]
            if not active_indices:
                break

            candidate_prefixes: list[list[int]] = []
            candidate_meta: list[tuple[int, int]] = []
            candidate_offsets: list[int] = []
            for chain_idx in active_indices:
                state = states[chain_idx]
                remaining = max_new_tokens - len(state.gen)
                length = min(block_size, remaining)
                if length <= 0:
                    state.active = False
                    continue
                for cand_idx in range(config.candidate_pool_size):
                    candidate_prefixes.append(state.prefix_ids + state.gen)
                    candidate_meta.append((chain_idx, cand_idx))
                    candidate_offsets.append(
                        20_000_003
                        + (block_idx + 1) * 1_000_003
                        + cand_idx * 10_007
                        + offsets[chain_idx]
                    )
            if not candidate_prefixes:
                continue

            candidate_rows = self._sample_ids_with_scores_batch(
                candidate_prefixes,
                temperature=1.0,
                max_new_tokens=block_size,
                seed_offsets=candidate_offsets,
                seed_base=base_seed,
                exact_length=True,
            )
            grouped: dict[int, list[dict[str, Any]]] = {idx: [] for idx in active_indices}
            for prefix, (chain_idx, cand_idx), (ids, norm, unnorm) in zip(
                candidate_prefixes, candidate_meta, candidate_rows
            ):
                grouped[chain_idx].append(
                    {
                        "candidate_index": cand_idx,
                        "ids": ids,
                        "norm": norm,
                        "unnorm": unnorm,
                        "logp": float(sum(norm)),
                    }
                )
                dollars_by_state[chain_idx] += estimate_cost(
                    len(prefix), len(ids)
                ).dollars

            selected_by_chain: dict[int, list[dict[str, Any]]] = {}
            rollout_prefixes: list[list[int]] = []
            rollout_meta: list[tuple[int, int, int]] = []
            rollout_offsets: list[int] = []
            rollout_lengths: list[int] = []
            for chain_idx in active_indices:
                state = states[chain_idx]
                candidates = sorted(
                    grouped[chain_idx],
                    key=lambda row: row["logp"],
                    reverse=True,
                )[: config.top_k]
                selected_by_chain[chain_idx] = candidates
                remaining_after_candidate = max_new_tokens - (
                    len(state.gen) + len(candidates[0]["ids"]) if candidates else len(state.gen)
                )
                horizon = min(
                    config.rollout_horizon
                    if config.rollout_horizon is not None
                    else block_size,
                    max(0, remaining_after_candidate),
                )
                if horizon <= 0 or config.rollouts_per_candidate == 0:
                    continue
                for cand_pos, candidate in enumerate(candidates):
                    prefix = state.prefix_ids + state.gen + candidate["ids"]
                    for rollout_idx in range(config.rollouts_per_candidate):
                        rollout_prefixes.append(prefix)
                        rollout_meta.append((chain_idx, cand_pos, rollout_idx))
                        rollout_lengths.append(horizon)
                        rollout_offsets.append(
                            40_000_009
                            + (block_idx + 1) * 1_000_003
                            + cand_pos * 10_007
                            + rollout_idx * 101
                            + offsets[chain_idx]
                        )

            rollout_logps: dict[tuple[int, int], list[float]] = {}
            if rollout_prefixes:
                rollout_rows = self._sample_ids_with_scores_batch(
                    rollout_prefixes,
                    temperature=1.0,
                    max_new_tokens=rollout_lengths,
                    seed_offsets=rollout_offsets,
                    seed_base=base_seed,
                    exact_length=True,
                )
                for prefix, (chain_idx, cand_pos, _), (ids, norm, _) in zip(
                    rollout_prefixes, rollout_meta, rollout_rows
                ):
                    rollout_logps.setdefault((chain_idx, cand_pos), []).append(
                        float(sum(norm))
                    )
                    dollars_by_state[chain_idx] += estimate_cost(
                        len(prefix), len(ids)
                    ).dollars

            eos_id = self.tokenizer.eos_token_id
            for chain_idx, candidates in selected_by_chain.items():
                if not candidates:
                    states[chain_idx].active = False
                    continue
                probs = sps_candidate_probabilities(
                    candidate_logps=[row["logp"] for row in candidates],
                    rollout_logps_by_candidate=[
                        rollout_logps.get((chain_idx, cand_pos), [])
                        for cand_pos in range(len(candidates))
                    ],
                    alpha=alpha,
                    jackknife=config.jackknife,
                )
                u = states[chain_idx].rng.random()
                cumulative = 0.0
                chosen_pos = len(probs) - 1
                for pos, prob in enumerate(probs):
                    cumulative += prob
                    if u <= cumulative:
                        chosen_pos = pos
                        break
                chosen = candidates[chosen_pos]
                state = states[chain_idx]
                state.gen.extend(chosen["ids"])
                state.log_probs_norm.extend(chosen["norm"])
                state.log_probs_unnorm.extend(chosen["unnorm"])
                if eos_id in state.gen:
                    eos_idx = state.gen.index(eos_id) + 1
                    state.gen = state.gen[:eos_idx]
                    state.log_probs_norm = state.log_probs_norm[:eos_idx]
                    state.log_probs_unnorm = state.log_probs_unnorm[:eos_idx]
                    state.active = False

        elapsed = time.monotonic() - started
        per_candidate_wall = elapsed / len(prompt_texts)
        generations: list[Generation] = []
        for state, dollars in zip(states, dollars_by_state):
            generations.append(
                Generation(
                    generation=self._decode(state.gen),
                    prompt_text=state.prompt_text,
                    response_contains_prompt=False,
                    prompt_token_count=len(state.prefix_ids),
                    generation_token_count=len(state.gen),
                    wall_clock_seconds=per_candidate_wall,
                    estimated_dollar_cost=dollars,
                    acceptance_ratio=None,
                    token_ids=state.gen,
                    logprobs_norm=state.log_probs_norm,
                    logprobs_unnorm=state.log_probs_unnorm,
                )
            )
        return generations

    def score_segments(
        self,
        prefix_ids_batch: list[list[int]],
        target_segments_batch: list[list[int]],
        *,
        temperature: float,
    ) -> ScoreBatch:
        if self.scoring_mode == "native_segment":
            return self._score_segments_native(
                prefix_ids_batch,
                target_segments_batch,
                temperature=temperature,
            )
        return self._score_segments_forced_decode(
            prefix_ids_batch,
            target_segments_batch,
            temperature=temperature,
        )

    def _score_segments_forced_decode(
        self,
        prefix_ids_batch: list[list[int]],
        target_segments_batch: list[list[int]],
        *,
        temperature: float,
    ) -> ScoreBatch:
        if len(prefix_ids_batch) != len(target_segments_batch):
            raise ValueError("prefix_ids_batch and target_segments_batch length mismatch")

        lp_norm: list[float] = []
        lp_unnorm: list[float] = []
        lp_norm_tokens: list[list[float]] = []
        lp_unnorm_tokens: list[list[float]] = []
        scored: list[tuple[int, list[int], VLLMForcedTokenProcessor]] = []
        prompts: list[dict[str, list[int]]] = []
        params_list: list[Any] = []

        for row_idx, (prefix_ids, target_ids) in enumerate(
            zip(prefix_ids_batch, target_segments_batch)
        ):
            if not prefix_ids:
                raise ValueError("prefix_ids must be non-empty to score next-token targets")
            if not target_ids:
                lp_norm.append(0.0)
                lp_unnorm.append(0.0)
                lp_norm_tokens.append([])
                lp_unnorm_tokens.append([])
                continue
            self._validate_token_ids(list(prefix_ids), context="score prefix_ids")
            self._validate_token_ids(list(target_ids), context="score target_ids")

            processor = VLLMForcedTokenProcessor(list(target_ids), temperature)
            params = self._sampling_params(
                max_tokens=len(target_ids),
                # Do not set min_tokens to the segment length here. vLLM enforces
                # min_tokens by suppressing EOS, which breaks teacher-forced
                # scoring for EOS-containing targets.
                min_tokens=0,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                ignore_eos=True,
                skip_special_tokens=False,
                detokenize=False,
                logits_processors=[processor],
            )
            lp_norm.append(float("nan"))
            lp_unnorm.append(float("nan"))
            lp_norm_tokens.append([])
            lp_unnorm_tokens.append([])
            prompts.append({"prompt_token_ids": list(prefix_ids)})
            params_list.append(params)
            scored.append((row_idx, [int(x) for x in target_ids], processor))

        if not scored:
            return ScoreBatch(
                lp_norm=lp_norm,
                lp_unnorm=lp_unnorm,
                lp_norm_tokens=lp_norm_tokens,
                lp_unnorm_tokens=lp_unnorm_tokens,
            )

        batch_limit = _env_batch_limit(
            "PROICL_SPS_VLLM_BATCH_SIZE",
            "SPS_VLLM_BATCH_SIZE",
        )
        self._reset_prefix_cache_for_scoring()
        try:
            outputs = []
            indices = list(range(len(prompts)))
            for chunk in _chunks(indices, batch_limit):
                outputs.extend(
                    self.llm.generate(
                        prompts=[prompts[idx] for idx in chunk],
                        sampling_params=[params_list[idx] for idx in chunk],
                        use_tqdm=False,
                    )
                )
        finally:
            self._reset_prefix_cache_for_scoring()
        for output, (row_idx, expected, processor) in zip(outputs, scored):
            observed = [int(x) for x in output.outputs[0].token_ids]
            if observed != expected:
                raise RuntimeError(
                    "vLLM forced-token scoring generated unexpected token ids: "
                    f"expected={expected[:8]} observed={observed[:8]}"
                )
            norm_list = list(processor.lp_norm)
            unnorm_list = list(processor.lp_unnorm)
            if len(norm_list) != len(expected) or len(unnorm_list) != len(expected):
                raise RuntimeError(
                    "vLLM logits processor did not record every target token: "
                    f"recorded={len(norm_list)} expected={len(expected)}"
                )
            lp_norm_tokens[row_idx] = norm_list
            lp_unnorm_tokens[row_idx] = unnorm_list
            lp_norm[row_idx] = float(sum(norm_list))
            lp_unnorm[row_idx] = float(sum(unnorm_list))

        return ScoreBatch(
            lp_norm=lp_norm,
            lp_unnorm=lp_unnorm,
            lp_norm_tokens=lp_norm_tokens,
            lp_unnorm_tokens=lp_unnorm_tokens,
        )

    def _reset_prefix_cache_for_scoring(self) -> None:
        if not self.enable_prefix_caching or not self.reset_prefix_cache_for_scoring:
            return
        engine = getattr(self.llm, "llm_engine", None)
        reset = getattr(engine, "reset_prefix_cache", None)
        if callable(reset):
            reset()

    def _score_segments_native(
        self,
        prefix_ids_batch: list[list[int]],
        target_segments_batch: list[list[int]],
        *,
        temperature: float,
    ) -> ScoreBatch:
        if len(prefix_ids_batch) != len(target_segments_batch):
            raise ValueError("prefix_ids_batch and target_segments_batch length mismatch")
        score_sequences = getattr(self.llm, "score_sequences", None)
        if score_sequences is None:
            raise RuntimeError(
                "vLLM scoring_mode='native_segment' requires a vLLM fork/branch "
                "with LLM.score_sequences(prompt_token_ids, target_token_ids, "
                "temperatures). Use scoring_mode='forced_decode_v0' until that "
                "native scorer is installed and HF parity artifacts pass."
            )

        lp_norm: list[float] = []
        lp_unnorm: list[float] = []
        lp_norm_tokens: list[list[float]] = []
        lp_unnorm_tokens: list[list[float]] = []
        scored: list[tuple[int, list[int]]] = []
        prompts: list[list[int]] = []
        targets: list[list[int]] = []

        for row_idx, (prefix_ids, target_ids) in enumerate(
            zip(prefix_ids_batch, target_segments_batch)
        ):
            if not prefix_ids:
                raise ValueError("prefix_ids must be non-empty to score next-token targets")
            target_ids = [int(x) for x in target_ids]
            if not target_ids:
                lp_norm.append(0.0)
                lp_unnorm.append(0.0)
                lp_norm_tokens.append([])
                lp_unnorm_tokens.append([])
                continue
            lp_norm.append(float("nan"))
            lp_unnorm.append(float("nan"))
            lp_norm_tokens.append([])
            lp_unnorm_tokens.append([])
            prompts.append([int(x) for x in prefix_ids])
            targets.append(target_ids)
            scored.append((row_idx, target_ids))

        if not scored:
            return ScoreBatch(
                lp_norm=lp_norm,
                lp_unnorm=lp_unnorm,
                lp_norm_tokens=lp_norm_tokens,
                lp_unnorm_tokens=lp_unnorm_tokens,
            )

        batch_limit = _env_batch_limit(
            "PROICL_SPS_VLLM_BATCH_SIZE",
            "SPS_VLLM_BATCH_SIZE",
        )
        outputs = []
        indices = list(range(len(prompts)))
        for chunk in _chunks(indices, batch_limit):
            chunk_prompts = [prompts[idx] for idx in chunk]
            chunk_targets = [targets[idx] for idx in chunk]
            try:
                chunk_outputs = score_sequences(
                    prompt_token_ids=chunk_prompts,
                    target_token_ids=chunk_targets,
                    temperatures=[float(temperature)] * len(chunk_targets),
                )
            except TypeError:
                chunk_outputs = score_sequences(
                    chunk_prompts,
                    chunk_targets,
                    [float(temperature)] * len(chunk_targets),
                )
            outputs.extend(list(chunk_outputs))
        if len(outputs) != len(scored):
            raise RuntimeError(
                "vLLM native score_sequences returned the wrong number of rows: "
                f"expected={len(scored)} observed={len(outputs)}"
            )

        for result, (row_idx, expected) in zip(outputs, scored):
            observed = _native_result_field(
                result,
                ("target_token_ids", "emitted_token_ids", "token_ids"),
            )
            if observed is not None:
                observed_ids = _coerce_int_list(observed)
                if observed_ids != expected:
                    raise RuntimeError(
                        "vLLM native scoring returned unexpected target token ids: "
                        f"expected={expected[:8]} observed={observed_ids[:8]}"
                    )

            temp_tokens = _coerce_float_list(
                _native_required_field(
                    result,
                    ("lp_temp", "logprobs_temp", "temperature_logprobs", "lp_norm"),
                )
            )
            unnorm_obj = _native_result_field(
                result,
                ("lp_unnorm", "logprobs_unnorm", "target_logprobs"),
            )
            if unnorm_obj is None:
                base_tokens = _coerce_float_list(
                    _native_required_field(
                        result,
                        ("lp_base", "logprobs_base", "base_logprobs"),
                    )
                )
                unnorm_tokens = [float(x) / float(temperature) for x in base_tokens]
            else:
                unnorm_tokens = _coerce_float_list(unnorm_obj)

            if len(temp_tokens) != len(expected) or len(unnorm_tokens) != len(expected):
                raise RuntimeError(
                    "vLLM native scoring returned misaligned token/logprob lengths: "
                    f"tokens={len(expected)} temp={len(temp_tokens)} "
                    f"unnorm={len(unnorm_tokens)}"
                )
            lp_norm_tokens[row_idx] = temp_tokens
            lp_unnorm_tokens[row_idx] = unnorm_tokens
            lp_norm[row_idx] = float(sum(temp_tokens))
            lp_unnorm[row_idx] = float(sum(unnorm_tokens))

        return ScoreBatch(
            lp_norm=lp_norm,
            lp_unnorm=lp_unnorm,
            lp_norm_tokens=lp_norm_tokens,
            lp_unnorm_tokens=lp_unnorm_tokens,
        )


def _native_result_field(result: Any, names: tuple[str, ...]) -> Any:
    if isinstance(result, dict):
        for name in names:
            if name in result:
                return result[name]
        return None
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return None


def _native_required_field(result: Any, names: tuple[str, ...]) -> Any:
    value = _native_result_field(result, names)
    if value is None:
        raise RuntimeError(
            "vLLM native score_sequences result is missing required logprob field; "
            f"accepted field names={names}"
        )
    return value


def _coerce_float_list(values: Any) -> list[float]:
    return [float(x) for x in list(values)]


def _coerce_int_list(values: Any) -> list[int]:
    return [int(x) for x in list(values)]
