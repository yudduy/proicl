from __future__ import annotations

import json
import re


VERIFIER_ID = "reasoning-gym/score-answer-v1"
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER_RE = re.compile(r"answer\s*:", re.IGNORECASE)


def _last_tagged_answer(generation: str) -> str | None:
    matches = [match.strip() for match in _ANSWER_RE.findall(generation)]
    if not matches:
        return None
    for answer in reversed(matches):
        if answer and answer not in {"...", "…"}:
            return answer
    return matches[-1]


def _json_prefix(text: str) -> str | None:
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    return text[:end].strip()


def _json_anywhere(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        try:
            _, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        return text[start : start + end].strip()
    return None


def _last_answer_marker(generation: str) -> str | None:
    matches = list(_ANSWER_MARKER_RE.finditer(generation))
    if not matches:
        return None
    tail = generation[matches[-1].end() :].strip()
    if not tail:
        return None
    json_answer = _json_prefix(tail) or _json_anywhere(tail)
    if json_answer is not None:
        return json_answer
    for line in tail.splitlines():
        answer = line.strip()
        if answer:
            answer = answer.strip("`*_ ")
            answer = answer.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z -]*[A-Za-z.]?", answer):
                answer = answer.rstrip(".").lower()
            return answer
    return None


def _extract_answer_for_scoring(generation: str) -> str:
    tagged = _last_tagged_answer(generation)
    if tagged is not None:
        return tagged
    marked = _last_answer_marker(generation)
    if marked is not None:
        return marked
    try:
        from reasoning_gym.utils import extract_answer
    except ImportError:
        extracted = None
    else:
        extracted = extract_answer(generation)
    return generation.strip() if extracted is None else extracted


def _json_loads_or_none(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _base_feedback(*, answer: str, score: float, scorer_error: str | None) -> dict:
    format_valid = bool(answer.strip())
    failure_type = None
    repair_hint = None
    if scorer_error is not None:
        format_valid = False
        failure_type = "scorer_exception"
        repair_hint = "Return only the final answer in the exact format requested by the task."
    elif score < 1.0:
        failure_type = "answer_mismatch"
        repair_hint = "Re-check the task constraints and produce a verifier-parseable final answer."
    return {
        "format_valid": format_valid,
        "failure_type": failure_type,
        "first_failure_step": None,
        "constraint_violation": None,
        "repair_hint": repair_hint,
    }


def _task_feedback(*, task: str, answer: str, score: float, scorer_error: str | None) -> dict:
    feedback = _base_feedback(answer=answer, score=score, scorer_error=scorer_error)
    if score >= 1.0:
        return feedback
    if task == "graph_color":
        parsed = _json_loads_or_none(answer)
        if parsed is None:
            feedback.update(
                {
                    "format_valid": False,
                    "failure_type": "graph_color_invalid_json",
                    "constraint_violation": "coloring was not parseable as JSON",
                    "repair_hint": (
                        "Return a JSON object mapping every node to a color, then verify "
                        "that adjacent nodes have different colors."
                    ),
                }
            )
        elif not isinstance(parsed, dict):
            feedback.update(
                {
                    "format_valid": False,
                    "failure_type": "graph_color_wrong_shape",
                    "constraint_violation": "coloring answer was not a node-to-color object",
                    "repair_hint": "Use an object like {\"0\":\"red\", \"1\":\"blue\"}.",
                }
            )
        else:
            feedback.update(
                {
                    "failure_type": "graph_color_constraint_conflict",
                    "constraint_violation": "one or more graph constraints failed verifier scoring",
                    "repair_hint": "Check each edge and recolor any adjacent nodes sharing a color.",
                }
            )
    elif task == "family_relationships":
        feedback.update(
            {
                "failure_type": "family_relationship_answer_mismatch",
                "constraint_violation": "final relation/name did not match the family graph",
                "repair_hint": (
                    "Build the family graph explicitly, preserve relation direction, "
                    "then answer with only the requested relation or person."
                ),
            }
        )
    elif task == "boxnet":
        parsed = _json_loads_or_none(answer)
        if parsed is None:
            feedback.update(
                {
                    "format_valid": False,
                    "failure_type": "boxnet_invalid_json",
                    "constraint_violation": "action plan was not parseable as JSON",
                    "repair_hint": "Return only a JSON action plan using the task's required schema.",
                }
            )
        elif not isinstance(parsed, (list, dict)):
            feedback.update(
                {
                    "format_valid": False,
                    "failure_type": "boxnet_wrong_action_shape",
                    "constraint_violation": "action plan shape did not match verifier expectations",
                    "repair_hint": (
                        "Serialize the action sequence exactly; include each move as a "
                        "discrete object or list item."
                    ),
                }
            )
        else:
            feedback.update(
                {
                    "failure_type": "boxnet_state_or_legality_failure",
                    "first_failure_step": "unknown",
                    "constraint_violation": (
                        "the parsed action plan failed a boxnet legality or final-state check"
                    ),
                    "repair_hint": (
                        "Simulate the grid after each action, check collisions and box "
                        "positions, then repair the first illegal move."
                    ),
                }
            )
    return feedback


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
    result.update(
        _task_feedback(task=task, answer=answer, score=score, scorer_error=scorer_error)
    )
    if scorer_error is not None:
        result["scorer_error"] = scorer_error
    return result
