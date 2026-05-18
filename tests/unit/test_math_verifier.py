from __future__ import annotations

from polaris.evals.verifiers.math import VERIFIER_ID, score_math


def test_verifier_id_constant():
    assert VERIFIER_ID == "math/sympy-equivalence-v1"


def test_correct_boxed_answer_passes():
    res = score_math("The answer is \\boxed{42}.", "42")
    assert res["passed"] is True
    assert res["score"] == 1.0
    assert res["extracted"] == "42"
    assert res["reference"] == "42"
    assert res["verifier_id"] == VERIFIER_ID


def test_wrong_boxed_answer_fails():
    res = score_math("The answer is \\boxed{43}.", "42")
    assert res["passed"] is False
    assert res["score"] == 0.0


def test_no_boxed_answer_fails_gracefully():
    res = score_math("no boxed marker here", "42")
    assert res["passed"] is False
    assert res["score"] == 0.0


def test_symbolic_equivalence_via_sympy():
    # 1/2 and 0.5 should grade equivalent under sympy normalization
    res = score_math("answer is \\boxed{1/2}", "0.5")
    assert res["passed"] is True


def test_latex_fraction_equivalence():
    res = score_math("answer is \\boxed{\\frac{1}{2}}", "1/2")
    assert res["passed"] is True


def test_result_is_a_plain_dict_with_required_keys():
    res = score_math("\\boxed{1}", "1")
    assert isinstance(res, dict)
    for key in ("score", "extracted", "reference", "passed", "verifier_id"):
        assert key in res
