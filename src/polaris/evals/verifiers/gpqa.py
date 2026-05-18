"""GPQA-Diamond evaluator (PROPOSAL §5.1, §4 offline-only rules).

Answer-key oracle is for offline analysis only. Inference-time selection
must use `select_gpqa_non_oracle` (majority vote with likelihood tiebreak),
not the answer key. Mixing the two without an explicit `oracle_selection`
label is a §4 violation.
"""

from __future__ import annotations

import re

ORACLE_VERIFIER_ID = "gpqa/answer-key-oracle-v1"
NON_ORACLE_SELECTOR_ID = "gpqa/majority-vote-likelihood-tiebreak-v1"


def extract_gpqa_answer(text: str) -> str | None:
    matches = re.findall(r"(?:answer\s*(?:is|:)?\s*)?\(?\b([ABCD])\b\)?", text, re.I)
    return matches[-1].upper() if matches else None


def score_gpqa_oracle(generation: str, reference: str) -> dict:
    """Offline-only oracle score. NEVER call this at inference-time selection
    unless the experiment is explicitly labeled `oracle_selection`."""
    predicted = extract_gpqa_answer(generation)
    gold = reference.strip().upper()
    passed = predicted == gold
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "verifier_id": ORACLE_VERIFIER_ID,
        "predicted_answer": predicted,
        "reference_answer": gold,
        "offline_only": True,
    }


def select_gpqa_non_oracle(candidates: list[dict]) -> dict:
    """Non-oracle selector per PROPOSAL §4: majority vote over extracted
    multiple-choice answers with normalized model likelihood as tiebreaker.
    """
    if not candidates:
        raise ValueError("select_gpqa_non_oracle requires at least one candidate")
    enriched: list[tuple[dict, str, float]] = []
    counts: dict[str, int] = {}
    for idx, candidate in enumerate(candidates):
        answer = candidate.get("answer") or extract_gpqa_answer(
            str(candidate.get("generation", ""))
        )
        if answer is None:
            answer = ""
        answer = str(answer).strip().upper()
        likelihood = float(
            candidate.get(
                "lp_norm_sum",
                candidate.get("log_likelihood", candidate.get("score", 0.0)),
            )
        )
        counts[answer] = counts.get(answer, 0) + 1
        enriched.append((candidate, answer, likelihood - idx * 1e-12))

    def key(item: tuple[dict, str, float]) -> tuple[int, float]:
        _, answer, likelihood = item
        return (counts[answer], likelihood)

    selected, answer, _ = max(enriched, key=key)
    out = dict(selected)
    out["selected_answer"] = answer or None
    out["selector_id"] = NON_ORACLE_SELECTOR_ID
    out["oracle_used"] = False
    return out
