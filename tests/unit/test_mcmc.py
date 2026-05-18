"""Unit tests for the batched MCMC math helpers.

Full `batched_mcmc_power_samp` is GPU-only (needs a real HF model + generate).
This file locks the load-bearing arithmetic of `_gather_token_logprobs` against
synthetic logits — that math is what the MH accept/reject ratio depends on, so
getting it wrong silently corrupts every result.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from types import SimpleNamespace

import polaris.infra.mcmc as mcmc_mod
from polaris.infra.mcmc import _gather_token_logprobs


def test_gather_uniform_logits_yields_uniform_logprob():
    """For uniform logits the per-token log-prob is -log(V)."""
    B, T, V = 1, 5, 4
    logits = torch.zeros(B, T, V)  # uniform → log_softmax = -log(V)
    seqs = torch.tensor([[0, 1, 2, 3, 0, 1]])  # length T+1 = 6
    score_ranges = [(1, T + 1)]  # score positions 1..T+1 (i.e. 5 tokens)
    lp_norm, lp_unnorm = _gather_token_logprobs(logits, seqs, score_ranges, temp=1.0)
    expected_per_token = -math.log(V)
    assert lp_norm[0].item() == pytest.approx(5 * expected_per_token, abs=1e-5)
    # At temp=1, unnorm == norm because (1/temp)*log_softmax(logits) == log_softmax(logits)
    assert lp_unnorm[0].item() == pytest.approx(lp_norm[0].item(), abs=1e-5)


def test_gather_temperature_scaling_lp_unnorm():
    """lp_unnorm = (1/temp) * log_softmax(logits) at the gathered token."""
    B, T, V = 1, 1, 4
    logits = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])  # (1, 1, 4)
    seqs = torch.tensor([[0, 0]])
    # Sequence position 1 is the predicted token. score_range (1, 2) → t_lo=0, t_hi=1.
    lp_norm, lp_unnorm = _gather_token_logprobs(logits, seqs, [(1, 2)], temp=0.25)
    base_lp = F.log_softmax(logits[0, 0], dim=-1)[
        0
    ].item()  # log_softmax of raw logits at token 0
    expected_unnorm = (1.0 / 0.25) * base_lp
    expected_norm = F.log_softmax(logits[0, 0] / 0.25, dim=-1)[0].item()
    assert lp_unnorm[0].item() == pytest.approx(expected_unnorm, abs=1e-5)
    assert lp_norm[0].item() == pytest.approx(expected_norm, abs=1e-5)


def test_gather_empty_range_is_zero():
    logits = torch.randn(2, 3, 5)
    seqs = torch.zeros(2, 4, dtype=torch.long)
    lp_norm, lp_unnorm = _gather_token_logprobs(
        logits, seqs, [(1, 1), (2, 2)], temp=0.25
    )
    assert lp_norm.tolist() == [0.0, 0.0]
    assert lp_unnorm.tolist() == [0.0, 0.0]


def test_gather_per_row_independent_ranges():
    """Two rows with different score_ranges: outputs must depend only on the per-row segment."""
    torch.manual_seed(0)
    B, T, V = 2, 4, 6
    logits = torch.randn(B, T, V)
    seqs = torch.randint(0, V, (B, T + 1))
    # Row 0 scores positions 1..3 (2 tokens); row 1 scores positions 2..5 (3 tokens).
    score_ranges = [(1, 3), (2, 5)]
    lp_norm, lp_unnorm = _gather_token_logprobs(logits, seqs, score_ranges, temp=0.5)

    # Compute by hand for row 0 over positions 1..3 → logits index 0..2 → 2 tokens.
    row = logits[0].float()
    lp_norm_hand = (
        F.log_softmax(row[0] / 0.5, dim=-1)[seqs[0, 1]]
        + F.log_softmax(row[1] / 0.5, dim=-1)[seqs[0, 2]]
    ).item()
    lp_unnorm_hand = (
        (1 / 0.5) * F.log_softmax(row[0], dim=-1)[seqs[0, 1]]
        + (1 / 0.5) * F.log_softmax(row[1], dim=-1)[seqs[0, 2]]
    ).item()
    assert lp_norm[0].item() == pytest.approx(lp_norm_hand, abs=1e-5)
    assert lp_unnorm[0].item() == pytest.approx(lp_unnorm_hand, abs=1e-5)


def test_mh_ratio_invariance_at_temp_one():
    """At temp=1, lp_norm == lp_unnorm everywhere → MH log_r = 0 for any swap."""
    torch.manual_seed(1)
    B, T, V = 4, 6, 8
    logits = torch.randn(B, T, V)
    seqs = torch.randint(0, V, (B, T + 1))
    score_ranges = [(1, T) for _ in range(B)]
    lp_norm, lp_unnorm = _gather_token_logprobs(logits, seqs, score_ranges, temp=1.0)
    # MH ratio uses (prop_unnorm - cur_unnorm) + (cur_norm - prop_norm).
    # At temp=1 the two terms are negatives of each other for any single configuration,
    # so any matched pair would cancel; locking it here as a sanity invariant.
    assert torch.allclose(lp_norm, lp_unnorm, atol=1e-5)


def test_batched_mcmc_resamples_from_full_generated_suffix(monkeypatch):
    """RWS picks any generated token after the fixed prompt, not only the latest block."""

    call_prefix_lens = []

    def fake_batched_draft(model, pad_token_id, prefix_ids_batch, new_tokens, temp):
        call_prefix_lens.append([len(x) for x in prefix_ids_batch])
        max_pref = max(len(x) for x in prefix_ids_batch)
        rows = []
        for row_idx, prefix in enumerate(prefix_ids_batch):
            pad = [pad_token_id] * (max_pref - len(prefix))
            rows.append(pad + list(prefix) + [10 + row_idx] * new_tokens)
        return (
            torch.tensor(rows, dtype=torch.long),
            None,
            [len(x) for x in prefix_ids_batch],
            None,
            max_pref,
        )

    def fake_score_segments(model, pad_token_id, full_sequences_batch, score_ranges, temp):
        return (
            torch.zeros(len(full_sequences_batch)),
            torch.zeros(len(full_sequences_batch)),
        )

    monkeypatch.setattr(mcmc_mod, "_batched_draft", fake_batched_draft)
    monkeypatch.setattr(mcmc_mod, "_score_segments", fake_score_segments)

    result = mcmc_mod.batched_mcmc_power_samp(
        SimpleNamespace(device="cpu"),
        SimpleNamespace(pad_token_id=0, eos_token_id=999),
        prefix_ids_batch=[[42]],
        temp=0.25,
        mcmc_steps=1,
        max_new_tokens=4,
        block_num=2,
        seed=1,
    )

    assert result.output_token_ids
    # Calls are: block-1 draft, block-1 proposal, block-2 draft, block-2 proposal.
    # With seed=1, the second RWS resample index is 0 in generated-token coordinates,
    # so the proposal prefix is just the fixed prompt (length 1). The previous buggy
    # implementation only sampled from the latest block and produced length 3 here.
    assert call_prefix_lens[3] == [1]
