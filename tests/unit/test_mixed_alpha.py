from __future__ import annotations

from polaris.core.mixed_alpha import FIXED_ALPHA_4, MIXED_ALPHA_4_1, AlphaSchedule


def test_fixed_alpha_4_constant():
    for sample_index in range(20):
        assert FIXED_ALPHA_4.alpha("any_prompt_id", sample_index) == 4.0


def test_mixed_alpha_4_1_parity_cycling():
    # alphas=(4.0, 1.0) — parity by sample_index, prompt_id ignored
    for i in range(0, 20, 2):
        assert MIXED_ALPHA_4_1.alpha("p", i) == 4.0
    for i in range(1, 20, 2):
        assert MIXED_ALPHA_4_1.alpha("p", i) == 1.0


def test_mixed_alpha_4_1_prompt_id_invariance():
    # alpha is a pure function of sample_index, not prompt_id
    for i in range(8):
        assert MIXED_ALPHA_4_1.alpha("a", i) == MIXED_ALPHA_4_1.alpha("b", i)


def test_policy_id_preserved():
    assert FIXED_ALPHA_4.policy_id == "fixed_alpha_4"
    assert MIXED_ALPHA_4_1.policy_id == "mixed_alpha_4_1"


def test_custom_schedule_cycles_alpha_tuple():
    sched = AlphaSchedule(policy_id="custom", alphas=(2.0, 3.0, 5.0))
    expected = [2.0, 3.0, 5.0, 2.0, 3.0, 5.0, 2.0]
    got = [sched.alpha("p", i) for i in range(7)]
    assert got == expected
