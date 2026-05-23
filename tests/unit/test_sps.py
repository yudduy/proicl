from __future__ import annotations

import pytest
import math
from pathlib import Path

from polaris.core.sps import sps_candidate_probabilities


def test_sps_future_scaling_can_override_local_block_likelihood():
    probs = sps_candidate_probabilities(
        candidate_logps=(-1.0, -0.5),
        rollout_logps_by_candidate=((-0.1, -0.2, -0.1), (-3.0, -3.0, -3.0)),
        alpha=4.0,
    )

    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1]


def test_sps_without_rollouts_reduces_to_low_temperature_over_candidates():
    probs = sps_candidate_probabilities(
        candidate_logps=(-1.0, -2.0),
        rollout_logps_by_candidate=((), ()),
        alpha=4.0,
    )

    assert probs[0] == pytest.approx(1.0 / (1.0 + math.exp(-4.0)))
    assert sum(probs) == pytest.approx(1.0)


def test_sps_jackknife_matches_leave_one_out_bias_correction():
    candidate_logps = (-1.0, -1.2)
    rollouts = ((-0.1, -2.0), (-0.8, -0.9))
    raw = sps_candidate_probabilities(
        candidate_logps=candidate_logps,
        rollout_logps_by_candidate=rollouts,
        alpha=4.0,
        jackknife=False,
    )
    loo0 = sps_candidate_probabilities(
        candidate_logps=candidate_logps,
        rollout_logps_by_candidate=((rollouts[0][1],), (rollouts[1][1],)),
        alpha=4.0,
        jackknife=False,
    )
    loo1 = sps_candidate_probabilities(
        candidate_logps=candidate_logps,
        rollout_logps_by_candidate=((rollouts[0][0],), (rollouts[1][0],)),
        alpha=4.0,
        jackknife=False,
    )

    expected = [
        2.0 * raw[i] - 0.5 * (loo0[i] + loo1[i])
        for i in range(len(candidate_logps))
    ]
    expected = [max(0.0, value) for value in expected]
    total = sum(expected)
    expected = [value / total for value in expected]

    corrected = sps_candidate_probabilities(
        candidate_logps=candidate_logps,
        rollout_logps_by_candidate=rollouts,
        alpha=4.0,
        jackknife=True,
    )

    assert corrected == pytest.approx(expected)
    assert sum(corrected) == pytest.approx(1.0)


def test_eval_config_records_sps_paper_contract():
    text = (Path(__file__).resolve().parents[2] / "configs" / "eval.yaml").read_text(
        encoding="utf-8"
    )

    assert "implementation: scalable_power_sampling" in text
    assert "target_distribution: p_alpha" in text
    assert "approximation: scaled_low_temperature_with_future_lookahead" in text
    assert "verifier_free_sampling: true" in text
    assert "jackknife_correction: true" in text
    assert "alpha: 4" in text
    assert "top_k: 8" in text
    assert "candidate_pool_size: 8" in text
    assert "rollouts_per_candidate: 8" in text
