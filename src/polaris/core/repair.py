from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from polaris.core.inference import Candidate


@dataclass(frozen=True)
class RepairConfig:
    max_attempts: int = 2
    temperature: float = 1.0
    mode: str = "verifier_guided"


def build_repair_prompt(*, original_prompt: str, failed_generation: str, verifier_result: dict) -> str:
    hint = verifier_result.get("repair_hint") or "Repair the answer so it passes the verifier."
    failure = verifier_result.get("failure_type") or "verifier_failed"
    violation = verifier_result.get("constraint_violation") or "not provided"
    return (
        f"{original_prompt}\n\n"
        "The previous answer failed external verification.\n"
        f"Failure type: {failure}\n"
        f"Constraint violation: {violation}\n"
        f"Repair hint: {hint}\n\n"
        "Previous answer:\n"
        f"{failed_generation}\n\n"
        "Return a corrected final answer in the original task format."
    )


def run_verifier_guided_repair(
    *,
    parent: Candidate,
    sampler: Any,
    scorer: Callable[[str, str], dict],
    reference: str,
    max_new_tokens: int,
    config: RepairConfig = RepairConfig(),
) -> list[Candidate]:
    if parent.verifier_result.get("passed", False):
        return []
    repairs: list[Candidate] = []
    prompt = build_repair_prompt(
        original_prompt=parent.prompt_text,
        failed_generation=parent.generation,
        verifier_result=parent.verifier_result,
    )
    for attempt in range(1, config.max_attempts + 1):
        gen = sampler.generate_low_temp(
            prompt,
            temperature=config.temperature,
            max_new_tokens=max_new_tokens,
        )
        full_response = (
            gen.generation if gen.response_contains_prompt else prompt + gen.generation
        )
        verifier_result = scorer(full_response, reference)
        repairs.append(
            Candidate(
                prompt_id=parent.prompt_id,
                sample_index=parent.sample_index,
                alpha=parent.alpha,
                prompt_text=prompt,
                generation=gen.generation,
                response_contains_prompt=gen.response_contains_prompt,
                prompt_token_count=gen.prompt_token_count,
                generation_token_count=gen.generation_token_count,
                wall_clock_seconds=gen.wall_clock_seconds,
                estimated_dollar_cost=gen.estimated_dollar_cost,
                acceptance_ratio=getattr(gen, "acceptance_ratio", None),
                verifier_result=verifier_result,
                retrieved_memory_ids=list(parent.retrieved_memory_ids),
                token_ids=list(getattr(gen, "token_ids", []) or []),
                logprobs_norm=list(getattr(gen, "logprobs_norm", []) or []),
                logprobs_unnorm=list(getattr(gen, "logprobs_unnorm", []) or []),
                search_trace_id=f"{parent.prompt_id}:{parent.sample_index}:repair:{attempt}",
                parent_candidate_id=f"{parent.prompt_id}:{parent.sample_index}",
                repair_attempt=attempt,
                search_strategy=config.mode,
            )
        )
        if verifier_result.get("passed", False):
            break
        prompt = build_repair_prompt(
            original_prompt=parent.prompt_text,
            failed_generation=gen.generation,
            verifier_result=verifier_result,
        )
    return repairs
