from __future__ import annotations

from polaris.evals.datasets.math500 import (
    MATH500_TEST_SLICE,
    Problem,
    load_math500_all,
    load_math500_slice,
)


def test_load_math500_all_returns_500_problems():
    problems = load_math500_all()
    assert len(problems) == 500


def test_problems_have_required_fields():
    problems = load_math500_all()
    first = problems[0]
    assert isinstance(first, Problem)
    assert first.prompt
    assert first.answer
    assert first.problem_id


def test_load_math500_slice_indexing():
    sub = load_math500_slice(0, 5)
    assert len(sub) == 5
    full = load_math500_all()
    for i, p in enumerate(sub):
        assert p.problem_id == full[i].problem_id


def test_test_slice_constant_is_full_math500():
    start, end = MATH500_TEST_SLICE
    assert (start, end) == (0, 500)
    sub = load_math500_slice(start, end)
    assert len(sub) == 500


def test_load_order_is_deterministic():
    a = [p.problem_id for p in load_math500_all()]
    b = [p.problem_id for p in load_math500_all()]
    assert a == b


def test_problem_is_frozen():
    p = load_math500_all()[0]
    import dataclasses

    assert dataclasses.is_dataclass(p)
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        p.prompt = "mutated"
