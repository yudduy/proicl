from __future__ import annotations

import pytest

from polaris.core.descriptor import (
    DESCRIPTOR_CATEGORIES,
    DESCRIPTOR_EXTRACTOR_VERSION,
    classify_trace,
)


def test_version_constant_is_set():
    assert DESCRIPTOR_EXTRACTOR_VERSION
    # versioned per proposal §4 (measurement instrument must be versioned)
    assert isinstance(DESCRIPTOR_EXTRACTOR_VERSION, str)


def test_categories_are_the_four_proposal_cells():
    assert DESCRIPTOR_CATEGORIES == (
        "direct_computation",
        "algebraic_transformation",
        "backward_verification",
        "stepwise_decomposition",
    )


def test_stepwise_trace_classified_as_stepwise():
    trace = (
        "Step 1: Compute 2 + 2 = 4. "
        "Step 2: Multiply by 3 to get 12. "
        "Step 3: Add 1 to get 13."
    )
    label, confidence = classify_trace(trace)
    assert label == "stepwise_decomposition"
    assert 0.0 <= confidence <= 1.0


def test_algebraic_trace_classified_as_algebraic():
    trace = "Let x = 3. Then 2x + 1 = 7. Solving for x gives x = 3."
    label, _ = classify_trace(trace)
    assert label == "algebraic_transformation"


def test_verification_trace_classified_as_backward_verification():
    trace = "The answer is 5. Verify by substituting back: 2*5 + 1 = 11. Correct."
    label, _ = classify_trace(trace)
    assert label == "backward_verification"


def test_direct_trace_falls_back_to_direct_computation():
    trace = "42."
    label, _ = classify_trace(trace)
    assert label == "direct_computation"


def test_empty_trace_returns_direct_computation():
    label, conf = classify_trace("")
    assert label == "direct_computation"
    assert conf == 0.0


def test_confidence_in_unit_interval():
    for trace in (
        "Step 1: ...\nStep 2: ...",
        "Let x = 5. Substitute back.",
        "Verify the answer.",
        "",
        "random",
    ):
        _, conf = classify_trace(trace)
        assert 0.0 <= conf <= 1.0


def test_classify_returns_tuple_of_str_and_float():
    out = classify_trace("Step 1: do thing")
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], str) and isinstance(out[1], float)
