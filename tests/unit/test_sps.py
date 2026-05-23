from __future__ import annotations

import pytest
import math

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
