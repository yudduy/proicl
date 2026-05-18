"""HumanEval+ code verifier (PROPOSAL §5.1 verifier policy item 2).

Wraps vendored `evalplus` sandboxed unit-test execution. Per PROPOSAL §7.2,
this is the load-bearing verifier for the Sustained-regime track — false
positives must stay low because they poison memory under v3+ runs.
"""

from __future__ import annotations

import json
import multiprocessing
import queue

VERIFIER_ID = "code/humaneval-plus-v1"


def _run_assertion_code(code: str, test_code: str, out: multiprocessing.Queue) -> None:
    ns: dict[str, object] = {}
    try:
        exec(code, ns)
        exec(test_code, ns)
    except BaseException as exc:
        out.put({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    else:
        out.put({"passed": True, "error": None})


def _score_with_test_code(generation: str, test_code: str, timeout_seconds: float) -> dict:
    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_assertion_code, args=(generation, test_code, q)
    )
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(0.2)
        if proc.is_alive():
            proc.kill()
        return {
            "score": 0.0,
            "passed": False,
            "verifier_id": VERIFIER_ID,
            "exec_details": {"status": "timeout"},
        }
    try:
        result = q.get_nowait()
    except queue.Empty:
        result = {"passed": False, "error": "no verifier result returned"}
    return {
        "score": 1.0 if result["passed"] else 0.0,
        "passed": bool(result["passed"]),
        "verifier_id": VERIFIER_ID,
        "exec_details": result,
    }


def _compose_solution(generation: str, payload: dict) -> str:
    prompt = payload.get("prompt", "")
    entry_point = payload.get("entry_point", "")
    stripped = generation.lstrip()
    if prompt and entry_point and f"def {entry_point}" not in stripped:
        return prompt + generation
    return generation


def _score_evalplus_payload(generation: str, payload: dict) -> dict:
    from polaris.vendored.evalplus.eval import PASS, untrusted_check
    from polaris.vendored.evalplus.gen.util import trusted_exec

    prompt = payload.get("prompt", "")
    canonical_solution = payload.get("canonical_solution", "")
    entry_point = payload["entry_point"]
    inputs = list(payload.get("base_input", [])) + list(payload.get("plus_input", []))
    if not inputs:
        return _score_with_test_code(
            _compose_solution(generation, payload),
            payload.get("test_code", ""),
            timeout_seconds=2.0,
        )
    expected, ref_time = trusted_exec(
        prompt + canonical_solution,
        inputs,
        entry_point,
        record_time=True,
    )
    status, details = untrusted_check(
        "humaneval",
        _compose_solution(generation, payload),
        inputs,
        entry_point,
        expected=expected,
        atol=payload.get("atol", 0),
        ref_time=ref_time,
        fast_check=True,
        min_time_limit=0.1,
        gt_time_limit_factor=4.0,
    )
    passed = status == PASS
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "verifier_id": VERIFIER_ID,
        "exec_details": {"status": status, "details": list(details)},
    }


def score_code(generation: str, reference: str) -> dict:
    """Run generation under HumanEval+ unit tests; return pass/score dict.

    Returns: {score: 1.0/0.0, passed: bool, verifier_id, exec_details}.
    Sandboxed execution; reference holds the test harness payload.

    `reference` is JSON emitted by `load_humaneval_plus_slice`. For local
    infrastructure smokes it may contain a direct `test_code` assertion block.
    For real EvalPlus rows it contains canonical solution and base/plus inputs,
    which are executed through the vendored EvalPlus checker.
    """
    payload = json.loads(reference)
    if payload.get("test_code"):
        return _score_with_test_code(
            _compose_solution(generation, payload),
            payload["test_code"],
            timeout_seconds=float(payload.get("timeout_seconds", 2.0)),
        )
    return _score_evalplus_payload(generation, payload)
