from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from polaris.config import (
    MAX_NEW_TOKENS,
    MCMC_BLOCK_NUM,
    MCMC_STEPS,
    MODEL_ID,
    PROPOSAL_TEMPERATURE,
    SEED,
    estimate_cost,
)
from polaris.infra.serving import ScoreBatch


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


class RWSGenerator:
    """Thin wrapper around the official RWS MCMC sampler."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        seed: int = SEED,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> None:
        import torch
        import transformers
        from polaris.vendored.rws.power_samp_utils import AutoregressiveSampler

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.autoreg_sampler = AutoregressiveSampler(self.model, self.tokenizer, device)

    def runtime_metadata(self) -> dict:
        return {
            "backend": "hf",
            "scoring_mode": "hf_forward_or_rws",
            "model_id": self.model_id,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "device": self.device,
            "tokenizer_special_tokens": {
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            },
        }

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def encode(self, text: str):
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def generate_greedy(
        self,
        prompt_text: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self._hf_generate(
            prompt_text=prompt_text,
            do_sample=False,
            temperature=None,
            max_new_tokens=max_new_tokens,
        )

    def generate_rws_std(
        self,
        prompt_text: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self._hf_generate(
            prompt_text=prompt_text,
            do_sample=True,
            temperature=None,
            max_new_tokens=max_new_tokens,
        )

    def generate_low_temp(
        self,
        prompt_text: str,
        temperature: float = PROPOSAL_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Generation:
        return self._hf_generate(
            prompt_text=prompt_text,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def generate_power(
        self,
        prompt_text: str,
        temperature: float = PROPOSAL_TEMPERATURE,
        mcmc_steps: int = MCMC_STEPS,
        max_new_tokens: int = MAX_NEW_TOKENS,
        block_num: int = MCMC_BLOCK_NUM,
    ) -> Generation:
        from polaris.vendored.rws.power_samp_utils import mcmc_power_samp

        input_ids = self.encode(prompt_text)
        prefix = [idx.item() for idx in input_ids[0]]
        started = time.monotonic()
        full_ids, _, _, acceptance_ratio = mcmc_power_samp(
            self.autoreg_sampler,
            prefix,
            temperature,
            mcmc_steps,
            max_new_tokens=max_new_tokens,
            block_num=block_num,
        )
        elapsed = time.monotonic() - started
        decoded = self.tokenizer.decode(
            self.torch.tensor(full_ids, dtype=self.torch.long).to("cpu"),
            skip_special_tokens=True,
        )
        output_tokens = max(0, len(full_ids) - len(prefix))
        cost = estimate_cost(len(prefix), output_tokens)
        return Generation(
            generation=decoded,
            prompt_text=prompt_text,
            response_contains_prompt=True,
            prompt_token_count=len(prefix),
            generation_token_count=output_tokens,
            wall_clock_seconds=elapsed,
            estimated_dollar_cost=cost.dollars,
            acceptance_ratio=float(acceptance_ratio),
        )

    def score_segments(
        self,
        prefix_ids_batch: list[list[int]],
        target_segments_batch: list[list[int]],
        *,
        temperature: float,
    ) -> ScoreBatch:
        """HF correctness-oracle scoring for already chosen target segments."""
        import torch.nn.functional as F

        if len(prefix_ids_batch) != len(target_segments_batch):
            raise ValueError("prefix_ids_batch and target_segments_batch length mismatch")

        lp_norm: list[float] = []
        lp_unnorm: list[float] = []
        lp_norm_tokens: list[list[float]] = []
        lp_unnorm_tokens: list[list[float]] = []
        for prefix_ids, target_ids in zip(prefix_ids_batch, target_segments_batch):
            if not prefix_ids:
                raise ValueError("prefix_ids must be non-empty to score next-token targets")
            if not target_ids:
                lp_norm.append(0.0)
                lp_unnorm.append(0.0)
                lp_norm_tokens.append([])
                lp_unnorm_tokens.append([])
                continue

            full_ids = list(prefix_ids) + list(target_ids)
            input_ids = self.torch.tensor([full_ids], dtype=self.torch.long).to(self.device)
            attn = self.torch.ones_like(input_ids)
            with self.torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            start = len(prefix_ids)
            end = len(full_ids)
            rows = out.logits[0, start - 1 : end - 1, :].float()
            targets = self.torch.tensor(target_ids, dtype=self.torch.long, device=rows.device)
            norm_vals = (
                F.log_softmax(rows / temperature, dim=-1)
                .gather(-1, targets.unsqueeze(-1))
                .squeeze(-1)
            )
            unnorm_vals = (
                (1.0 / temperature)
                * F.log_softmax(rows, dim=-1)
                .gather(-1, targets.unsqueeze(-1))
                .squeeze(-1)
            )
            norm_list = [float(x) for x in norm_vals.detach().cpu().tolist()]
            unnorm_list = [float(x) for x in unnorm_vals.detach().cpu().tolist()]
            lp_norm_tokens.append(norm_list)
            lp_unnorm_tokens.append(unnorm_list)
            lp_norm.append(float(sum(norm_list)))
            lp_unnorm.append(float(sum(unnorm_list)))
            del out
        return ScoreBatch(
            lp_norm=lp_norm,
            lp_unnorm=lp_unnorm,
            lp_norm_tokens=lp_norm_tokens,
            lp_unnorm_tokens=lp_unnorm_tokens,
        )

    def _hf_generate(
        self,
        prompt_text: str,
        do_sample: bool,
        temperature: float | None,
        max_new_tokens: int,
    ) -> Generation:
        input_ids = self.encode(prompt_text)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": self.torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": False,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        started = time.monotonic()
        output = self.model.generate(**kwargs)
        elapsed = time.monotonic() - started
        generated_ids = output.sequences[0][len(input_ids[0]) :].detach().to("cpu")
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        cost = estimate_cost(len(input_ids[0]), len(generated_ids))
        return Generation(
            generation=text,
            prompt_text=prompt_text,
            response_contains_prompt=False,
            prompt_token_count=len(input_ids[0]),
            generation_token_count=len(generated_ids),
            wall_clock_seconds=elapsed,
            estimated_dollar_cost=cost.dollars,
            acceptance_ratio=None,
        )
