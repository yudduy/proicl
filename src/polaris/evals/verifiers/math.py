from __future__ import annotations

from polaris.vendored.rws.grader_utils.math_grader import grade_answer
from polaris.vendored.rws.grader_utils.parse_utils import parse_answer

VERIFIER_ID = "math/sympy-equivalence-v1"


def score_math(generation: str, reference: str) -> dict:
    """Extract \\boxed{} answer from generation and check sympy equivalence to reference.

    Returns a dict with score (1.0/0.0), extracted answer, reference, passed bool, verifier_id.
    Wraps vendored RWS math_grader (proposal §"Verifier policy" item 1).
    """
    try:
        extracted = parse_answer(generation) or ""
    except Exception:
        extracted = ""
    try:
        ok = bool(grade_answer(extracted, reference))
    except Exception:
        ok = False
    return {
        "score": 1.0 if ok else 0.0,
        "extracted": extracted,
        "reference": reference,
        "passed": ok,
        "verifier_id": VERIFIER_ID,
    }
