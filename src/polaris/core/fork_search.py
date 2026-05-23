from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from polaris.core.inference import Candidate


@dataclass(frozen=True)
class ForkSearchConfig:
    top_k: int = 4
    max_depth: int = 2
    entropy_threshold: float = 2.5
    temperature: float = 1.0
    decision_suffix: str = "\n\nNext decision:"


def entropy_from_probs(probs: Iterable[float]) -> float:
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(p)
    return total


def _branch_distribution(sampler: Any, prompt: str, top_k: int) -> tuple[float, list[tuple[str, float]]]:
    fn = getattr(sampler, "next_token_distribution", None)
    if not callable(fn):
        return 0.0, []
    rows = fn(prompt, top_k=top_k)
    branches = [(str(token), float(prob)) for token, prob in rows[:top_k]]
    return entropy_from_probs(prob for _, prob in branches), branches


def run_entropy_gated_fork_search(
    *,
    prompt_id: str,
    sample_index_start: int,
    prompt_text: str,
    sampler: Any,
    scorer: Callable[[str, str], dict],
    reference: str,
    max_new_tokens: int,
    config: ForkSearchConfig = ForkSearchConfig(),
) -> list[Candidate]:
    decision_prompt = prompt_text + config.decision_suffix
    entropy, branches = _branch_distribution(sampler, decision_prompt, config.top_k)
    if not branches or entropy < config.entropy_threshold:
        return []
    candidates: list[Candidate] = []
    frontier = [(decision_prompt, token, prob, 1) for token, prob in branches]
    next_sample = sample_index_start
    while frontier:
        branch_prompt, token, prob, depth = frontier.pop(0)
        continuation_prompt = f"{branch_prompt} {token}"
        gen = sampler.generate_low_temp(
            continuation_prompt,
            temperature=config.temperature,
            max_new_tokens=max_new_tokens,
        )
        full_response = (
            gen.generation
            if gen.response_contains_prompt
            else continuation_prompt + gen.generation
        )
        verifier_result = scorer(full_response, reference)
        candidates.append(
            Candidate(
                prompt_id=prompt_id,
                sample_index=next_sample,
                alpha=1.0,
                prompt_text=continuation_prompt,
                generation=gen.generation,
                response_contains_prompt=gen.response_contains_prompt,
                prompt_token_count=gen.prompt_token_count,
                generation_token_count=gen.generation_token_count,
                wall_clock_seconds=gen.wall_clock_seconds,
                estimated_dollar_cost=gen.estimated_dollar_cost,
                acceptance_ratio=getattr(gen, "acceptance_ratio", None),
                verifier_result=verifier_result,
                token_ids=list(getattr(gen, "token_ids", []) or []),
                logprobs_norm=list(getattr(gen, "logprobs_norm", []) or []),
                logprobs_unnorm=list(getattr(gen, "logprobs_unnorm", []) or []),
                search_trace_id=f"{prompt_id}:fork:{next_sample}",
                parent_candidate_id=f"{prompt_id}:fork-root",
                fork_depth=depth,
                fork_entropy=entropy,
                fork_token=token,
                fork_probability=prob,
                search_strategy="entropy_gated_fork_search",
            )
        )
        next_sample += 1
        if verifier_result.get("passed", False) or depth >= config.max_depth:
            continue
        child_entropy, child_branches = _branch_distribution(
            sampler,
            continuation_prompt + config.decision_suffix,
            config.top_k,
        )
        if child_entropy >= config.entropy_threshold:
            frontier.extend(
                (continuation_prompt + config.decision_suffix, child, child_prob, depth + 1)
                for child, child_prob in child_branches
            )
    return candidates
