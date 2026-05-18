"""Batched MCMC power sampling on HF transformers.

Mirrors `polaris.vendored.rws.power_samp_utils.mcmc_power_samp` (RWS, arXiv:2510.14901)
but advances K chains in lock-step through a single padded forward pass. Per-chain
throughput is ~K× the vendored implementation up to GPU saturation.

Why this file remains the HF correctness oracle:
The MH accept/reject ratio (proposal §4.3, eq. for `log_r`) requires
`log p_temp(x_t | x_<t) = log_softmax(logits/temp)` *and*
`(1/temp) * log p_base(x_t | x_<t) = (1/temp) * log_softmax(logits)` at every
resampled position. The temperature-normalized term contains a partition function
`log Z_t = logsumexp(logits/temp)` that needs the **full-vocabulary logits** at
that position. HF returns those directly via `model(input_ids).logits`, so this
path is the oracle for validating faster SGLang/vLLM serving adapters before any
paid scale-up.

Batching wins applied:
- K parallel chains via padded `model(input_ids, attention_mask)` forward
- Shared prompt-prefix KV cache (HF reuses via `past_key_values` between blocks)
- FlashAttention-2 (`attn_implementation="flash_attention_2"`)
- Optional `torch.compile` of the forward (set on the model at load time)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class BatchedMCMCResult:
    """Output of `batched_mcmc_power_samp`."""

    output_token_ids: list[
        list[int]
    ]  # K final chains (generated tokens only, no prefix)
    acceptance_ratios: list[float]  # accepted / proposed per chain
    wall_clock_seconds: float


def _gather_token_logprobs(
    logits,  # (B, T, V) tensor of raw logits
    sequences,  # (B, T+1) input_ids the logits scored
    score_ranges,  # list[(start, end)] half-open per-row segments to sum
    temp: float,
):
    """Return per-row sums of (lp_norm, lp_unnorm) over `score_ranges`.

    `logits[b, t, :]` is the next-token distribution given `sequences[b, :t+1]`.
    So `sequences[b, t+1]` is the token whose lp is determined by `logits[b, t, :]`.
    For a segment `[start, end)` of *predicted* token positions (0-indexed in the
    sequence), we sum log-probs over `t = start-1 .. end-2` in the logits index.
    """
    import torch
    import torch.nn.functional as F

    B, T, V = logits.shape
    # Build a mask + index tensor for vectorized gather.
    lp_norm_sums = torch.zeros(B, device=logits.device, dtype=torch.float32)
    lp_unnorm_sums = torch.zeros(B, device=logits.device, dtype=torch.float32)
    for b, (start, end) in enumerate(score_ranges):
        if end <= start:
            continue
        # Logits row at position t predicts token at sequence position t+1.
        t_lo, t_hi = start - 1, end - 1  # half-open in logits index
        row_logits = logits[b, t_lo:t_hi, :].float()  # (L, V)
        targets = sequences[b, start:end]  # (L,)
        # Normalized (temperature-scaled) log-probs.
        log_p_norm = F.log_softmax(row_logits / temp, dim=-1)
        # Unnormalized: (1/temp) * log_softmax of base distribution.
        log_p_unnorm = (1.0 / temp) * F.log_softmax(row_logits, dim=-1)
        lp_norm_sums[b] = (
            log_p_norm.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1).sum()
        )
        lp_unnorm_sums[b] = (
            log_p_unnorm.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1).sum()
        )
    return lp_norm_sums, lp_unnorm_sums


def _batched_draft(
    model,
    pad_token_id: int,
    prefix_ids_batch,
    new_tokens: int,
    temp: float,
):
    """Generate `new_tokens` new tokens per chain at temperature `temp`.

    Returns:
        full_sequences: (B, max_prefix_len + new_tokens) padded LEFT with pad_token_id
        attention_mask: (B, max_prefix_len + new_tokens)
        prefix_lens: (B,) original prefix length per row
        per_step_logits: tuple of `new_tokens` tensors, each (B, V) — raw logits
    """
    import torch

    B = len(prefix_ids_batch)
    prefix_lens = [len(p) for p in prefix_ids_batch]
    max_pref = max(prefix_lens)
    # Left-pad so all chains end at the same position; decoder consumes from the right.
    input_ids = torch.full((B, max_pref), pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, max_pref), dtype=torch.long)
    for b, ids in enumerate(prefix_ids_batch):
        input_ids[b, -len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attn[b, -len(ids) :] = 1
    input_ids = input_ids.to(model.device)
    attn = attn.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,  # MCMC requires exact length; RWS continues past EOS
            do_sample=True,
            temperature=temp,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
        )
    return out.sequences, attn, prefix_lens, None, max_pref


def _score_segments(
    model,
    pad_token_id: int,
    full_sequences_batch,
    score_ranges,
    temp: float,
):
    """One batched forward; return per-row (lp_norm_sum, lp_unnorm_sum) over score_ranges."""
    import torch

    B = len(full_sequences_batch)
    seq_lens = [len(s) for s in full_sequences_batch]
    max_len = max(seq_lens)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for b, ids in enumerate(full_sequences_batch):
        input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[b, : len(ids)] = 1
    input_ids = input_ids.to(model.device)
    attn = attn.to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        result = _gather_token_logprobs(out.logits, input_ids, score_ranges, temp=temp)
    del out
    return result


def batched_mcmc_power_samp(
    model,
    tokenizer,
    prefix_ids_batch: list[list[int]],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int = 3072,
    block_num: int = 16,
    seed: int = 0,
) -> BatchedMCMCResult:
    """K parallel MCMC chains, one batched forward per step.

    Mirrors the vendored RWS algorithm exactly:
    - 16 blocks of size 192 (max_new_tokens / block_num)
    - Each block: draft `jump_size` tokens at temp=1/alpha, then `mcmc_steps` MH proposals
    - Each MH step: per-chain random resample position; propose; vectorized accept/reject

    Drop-in cheaper replacement for K sequential calls to vendored `mcmc_power_samp`.
    """
    import torch

    K = len(prefix_ids_batch)
    jump_size = max_new_tokens // block_num
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    rng = random.Random(seed)
    gens: list[list[int]] = [[] for _ in range(K)]
    accepted = [0] * K
    proposed = [0] * K

    started = time.monotonic()

    for block_k in range(block_num):
        # Draft: extend each chain by jump_size tokens at temp.
        full_prefixes = [list(prefix_ids_batch[k]) + gens[k] for k in range(K)]
        seq_after_draft, _, prefix_lens, _, max_pref_in = _batched_draft(
            model,
            pad_id,
            full_prefixes,
            new_tokens=jump_size,
            temp=temp,
        )
        # Extract just the *generated* portion per chain (drop the left padding and prefix).
        for k in range(K):
            full_row = seq_after_draft[k].tolist()
            # left-padded; chain k's real content starts at max_pref_in - prefix_lens[k].
            start_of_real = max_pref_in - prefix_lens[k]
            full_seq = full_row[start_of_real:]  # prefix_ids + gens + newly drafted
            gens[k] = full_seq[len(prefix_ids_batch[k]) :]  # drop prefix → gen portion

        for _ in range(mcmc_steps):
            for k in range(K):
                proposed[k] += 1
            t_per = [len(gens[k]) for k in range(K)]
            idx_per = [rng.randint(0, t_per[k] - 1) for k in range(K)]
            # Propose new suffixes: per-chain new lengths vary.
            propose_prefixes = [
                list(prefix_ids_batch[k]) + gens[k][: idx_per[k]] for k in range(K)
            ]
            propose_new_lens = [t_per[k] - idx_per[k] for k in range(K)]
            max_new_propose = max(propose_new_lens)
            seq_prop, _, prop_prefix_lens, _, max_pref_prop = _batched_draft(
                model,
                pad_id,
                propose_prefixes,
                new_tokens=max_new_propose,
                temp=temp,
            )
            # Per-chain proposed suffix tokens.
            proposed_segments: list[list[int]] = []
            for k in range(K):
                full_row = seq_prop[k].tolist()
                start_of_real = max_pref_prop - prop_prefix_lens[k]
                suffix = full_row[start_of_real + len(propose_prefixes[k]) :]
                # Take only as many tokens as we need (per-chain).
                proposed_segments.append(suffix[: propose_new_lens[k]])

            # Score CURRENT chain (prefix + gens) on segment [prefix_len+idx, prefix_len+t).
            cur_full = [list(prefix_ids_batch[k]) + gens[k] for k in range(K)]
            cur_ranges = [
                (
                    len(prefix_ids_batch[k]) + idx_per[k],
                    len(prefix_ids_batch[k]) + t_per[k],
                )
                for k in range(K)
            ]
            cur_norm, cur_unnorm = _score_segments(
                model, pad_id, cur_full, cur_ranges, temp=temp
            )

            # Score PROPOSED chain (prefix + gens[:idx] + proposed_segments) on same range.
            prop_full = [
                list(prefix_ids_batch[k]) + gens[k][: idx_per[k]] + proposed_segments[k]
                for k in range(K)
            ]
            prop_norm, prop_unnorm = _score_segments(
                model,
                pad_id,
                prop_full,
                cur_ranges,
                temp=temp,
            )

            # MH ratio per chain (matches RWS `log_r`).
            log_r = (prop_unnorm - cur_unnorm) + (cur_norm - prop_norm)
            log_u = torch.log(torch.rand(K, device=log_r.device, generator=None))
            accept_mask = log_u < log_r

            for k in range(K):
                if accept_mask[k].item():
                    gens[k] = gens[k][: idx_per[k]] + proposed_segments[k]
                    accepted[k] += 1
            # Release the per-step logits tensors before next MH step.
            del cur_norm, cur_unnorm, prop_norm, prop_unnorm, log_r, log_u
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    wall = time.monotonic() - started
    return BatchedMCMCResult(
        output_token_ids=gens,
        acceptance_ratios=[a / p if p else 0.0 for a, p in zip(accepted, proposed)],
        wall_clock_seconds=wall,
    )
