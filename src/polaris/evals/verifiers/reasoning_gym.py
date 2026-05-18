from __future__ import annotations

import json


VERIFIER_ID = "reasoning-gym/score-answer-v1"


def _extract_answer_for_scoring(generation: str) -> str:
    try:
        from reasoning_gym.utils import extract_answer
    except ImportError:
        extracted = None
    else:
        extracted = extract_answer(generation)
    return generation.strip() if extracted is None else extracted


def score_reasoning_gym(generation: str, reference: str) -> dict:
    payload = json.loads(reference)
    task = payload["task"]
    entry = payload["entry"]
    answer = _extract_answer_for_scoring(generation)
    scorer_error = None
    try:
        import reasoning_gym
    except ImportError:
        expected = str(payload.get("answer", "")).strip().lower()
        got = answer.strip().lower()
        score = 1.0 if got == expected else 0.0
    else:
        dataset = reasoning_gym.create_dataset(task, size=1, seed=0)
        try:
            score = float(dataset.score_answer(answer=answer, entry=entry))
            scorer_error = None
        except Exception as exc:  # Badly shaped model answers should fail the candidate, not the run.
            score = 0.0
            scorer_error = f"{type(exc).__name__}: {exc}"
    result = {
        "score": score,
        "passed": score >= 1.0,
        "verifier_id": VERIFIER_ID,
        "task": task,
        "extracted_answer": answer,
    }
    if scorer_error is not None:
        result["scorer_error"] = scorer_error
    return result
