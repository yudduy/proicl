from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from polaris.core.archive import FrozenArchive
from polaris.core.memory import MemoryEntry, MemoryStore
from polaris.core.mixed_alpha import AlphaSchedule
from polaris.io.trajectory_cache import TrajectoryCache, TrajectoryKey, TrajectoryRecord


@dataclass
class Candidate:
    """One generated candidate plus its verifier outcome (proposal §7)."""

    prompt_id: str
    sample_index: int
    alpha: float
    prompt_text: str
    generation: str
    response_contains_prompt: bool
    prompt_token_count: int
    generation_token_count: int
    wall_clock_seconds: float
    estimated_dollar_cost: float
    acceptance_ratio: float | None
    verifier_result: dict
    retrieved_memory_ids: list[str] = field(default_factory=list)
    admitted_memory_id: str | None = None
    token_ids: list[int] = field(default_factory=list)
    logprobs_norm: list[float] = field(default_factory=list)
    logprobs_unnorm: list[float] = field(default_factory=list)


class _SamplerLike(Protocol):
    def generate_power(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int,
        mcmc_steps: int = ...,
        block_num: int = ...,
    ) -> Any: ...

    def generate_low_temp(
        self, prompt_text: str, *, temperature: float, max_new_tokens: int
    ) -> Any: ...


def _list_attr(obj: Any, name: str) -> list:
    value = getattr(obj, name, None)
    return list(value) if value is not None else []


def _compose_with_memory(*, entry, question: str, memories: list[MemoryEntry]) -> str:
    if not memories:
        return entry.compose(question)
    memory_lines = "\n".join(f"- {m.strategy_text}" for m in memories)
    memory_block = f"\nRelevant verified strategies:\n{memory_lines}\n"
    return f"{entry.prefix}{memory_block}{question}{entry.suffix}"


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def polaris_inference(
    *,
    question: str,
    reference: str,
    archive: FrozenArchive,
    sampler: _SamplerLike,
    alpha_schedule: AlphaSchedule,
    total_samples: int,
    max_new_tokens: int,
    scorer: Callable[[str, str], dict],
    skip_alpha_one_mcmc: bool = True,
    trajectory_cache: TrajectoryCache | None = None,
    cache_model_id: str | None = None,
    cache_track: str | None = None,
    cache_problem_id: str | None = None,
    cache_seed: int = 0,
    memory_store: MemoryStore | None = None,
    admit_memory: bool = False,
    memory_independent_check: Callable[[str], bool] | None = None,
    token_counter: Callable[[str], int] = _token_count,
    mcmc_steps: int | None = None,
    mcmc_block_num: int | None = None,
) -> tuple[Candidate, list[Candidate]]:
    """Run POLARIS inference for one query under a fixed budget (proposal §7).

    No online memory in v1 — `archive.max_retrieved_memory_entries` should be 0.
    Selector is argmax verifier score with iteration-order tiebreak.

    `skip_alpha_one_mcmc=True` short-circuits alpha=1 to plain temperature-1
    sampling instead of running an MCMC chain with no sharpening. Mathematically
    equivalent in expectation; saves wall time.

    `trajectory_cache` is optional and inactive by default. When provided, the
    caller must also pass model/track/problem metadata so replay keys are
    unambiguous across experiments.

    Returns (best_candidate, all_candidates).
    """
    if trajectory_cache is not None and (
        not cache_model_id or not cache_track or not cache_problem_id
    ):
        raise ValueError(
            "trajectory_cache requires cache_model_id, cache_track, and cache_problem_id"
        )

    allocation = archive.allocate(total_samples)
    candidates: list[Candidate] = []

    entry_by_id = {e.id: e for e in archive.entries}
    for entry_id, n_samples in allocation.items():
        entry = entry_by_id[entry_id]
        memories: list[MemoryEntry] = []
        if memory_store is not None and archive.max_retrieved_memory_entries > 0:
            memories = memory_store.retrieve(
                archive_prompt_id=entry_id,
                descriptor_filter=entry.descriptor_hint,
                max_entries=archive.max_retrieved_memory_entries,
                max_tokens=archive.max_retrieved_memory_tokens,
            )
        prompt_text = _compose_with_memory(
            entry=entry, question=question, memories=memories
        )
        retrieved_memory_ids = [m.id for m in memories]
        for sample_index in range(n_samples):
            alpha = alpha_schedule.alpha(entry_id, sample_index)
            cached = None
            if trajectory_cache is not None:
                cached = trajectory_cache.get(
                    model_id=cache_model_id or "",
                    track=cache_track or "",
                    problem_id=cache_problem_id or "",
                    prompt_id=entry_id,
                    sample_idx=sample_index,
                    alpha=alpha,
                    seed=cache_seed,
                )

            if cached is not None:
                verifier_result = cached.verifier_result
                if verifier_result is None:
                    full_response = (
                        cached.generation
                        if cached.response_contains_prompt
                        else prompt_text + cached.generation
                    )
                    verifier_result = scorer(full_response, reference)
                    trajectory_cache.mark_verified(
                        cached.key, verifier_result=verifier_result
                    )
                candidates.append(
                    Candidate(
                        prompt_id=entry_id,
                        sample_index=sample_index,
                        alpha=alpha,
                        prompt_text=prompt_text,
                        generation=cached.generation,
                        response_contains_prompt=cached.response_contains_prompt,
                        prompt_token_count=cached.prompt_token_count,
                        generation_token_count=cached.generation_token_count,
                        wall_clock_seconds=cached.wall_clock_seconds,
                        estimated_dollar_cost=cached.dollar_cost,
                        acceptance_ratio=cached.acceptance_ratio,
                        verifier_result=verifier_result,
                        retrieved_memory_ids=retrieved_memory_ids,
                        token_ids=list(cached.token_ids),
                        logprobs_norm=list(cached.logprobs_norm),
                        logprobs_unnorm=list(cached.logprobs_unnorm),
                    )
                )
                continue

            if alpha <= 1.0 and skip_alpha_one_mcmc:
                gen = sampler.generate_low_temp(
                    prompt_text,
                    temperature=1.0,
                    max_new_tokens=max_new_tokens,
                )
            else:
                power_kwargs: dict[str, Any] = {}
                if mcmc_steps is not None:
                    power_kwargs["mcmc_steps"] = mcmc_steps
                if mcmc_block_num is not None:
                    power_kwargs["block_num"] = mcmc_block_num
                gen = sampler.generate_power(
                    prompt_text,
                    temperature=1.0 / alpha,
                    max_new_tokens=max_new_tokens,
                    **power_kwargs,
                )
            full_response = (
                gen.generation
                if gen.response_contains_prompt
                else prompt_text + gen.generation
            )
            verifier_result = scorer(full_response, reference)
            admitted_memory_id = None
            if memory_store is not None and retrieved_memory_ids:
                memory_store.update_reliability(
                    retrieved_memory_ids,
                    1 if verifier_result.get("passed", False) else 0,
                )
            if (
                memory_store is not None
                and admit_memory
                and verifier_result.get("passed", False)
            ):
                check = memory_independent_check or (
                    lambda trace: bool(verifier_result.get("passed", False))
                )
                entry_obj = memory_store.admit(
                    candidate_trace=full_response,
                    archive_prompt_id=entry_id,
                    descriptor=entry.descriptor_hint,
                    source_query_id=cache_problem_id or "unknown",
                    independent_check=check,
                    token_counter=token_counter,
                    entry_id=f"{entry_id}:{cache_problem_id or 'query'}:{sample_index}",
                )
                admitted_memory_id = entry_obj.id if entry_obj is not None else None
            candidate = Candidate(
                prompt_id=entry_id,
                sample_index=sample_index,
                alpha=alpha,
                prompt_text=prompt_text,
                generation=gen.generation,
                response_contains_prompt=gen.response_contains_prompt,
                prompt_token_count=gen.prompt_token_count,
                generation_token_count=gen.generation_token_count,
                wall_clock_seconds=gen.wall_clock_seconds,
                estimated_dollar_cost=gen.estimated_dollar_cost,
                acceptance_ratio=gen.acceptance_ratio,
                verifier_result=verifier_result,
                retrieved_memory_ids=retrieved_memory_ids,
                admitted_memory_id=admitted_memory_id,
                token_ids=_list_attr(gen, "token_ids"),
                logprobs_norm=_list_attr(gen, "logprobs_norm"),
                logprobs_unnorm=_list_attr(gen, "logprobs_unnorm"),
            )
            candidates.append(candidate)
            if trajectory_cache is not None:
                trajectory_cache.put(
                    TrajectoryRecord(
                        key=TrajectoryKey(
                            model_id=cache_model_id or "",
                            track=cache_track or "",
                            problem_id=cache_problem_id or "",
                            prompt_id=entry_id,
                            sample_idx=sample_index,
                            alpha=alpha,
                            seed=cache_seed,
                        ),
                        prompt_hash=TrajectoryCache.prompt_hash(prompt_text),
                        generation=gen.generation,
                        response_contains_prompt=gen.response_contains_prompt,
                        token_ids=_list_attr(gen, "token_ids"),
                        logprobs_norm=_list_attr(gen, "logprobs_norm"),
                        logprobs_unnorm=_list_attr(gen, "logprobs_unnorm"),
                        acceptance_ratio=gen.acceptance_ratio,
                        prompt_token_count=gen.prompt_token_count,
                        generation_token_count=gen.generation_token_count,
                        wall_clock_seconds=gen.wall_clock_seconds,
                        dollar_cost=gen.estimated_dollar_cost,
                        verifier_result=verifier_result,
                    )
                )

    if not candidates:
        raise ValueError("polaris_inference produced no candidates (total_samples=0?)")

    # argmax verifier score with iteration-order tiebreak (first wins).
    best_idx = 0
    best_score = candidates[0].verifier_result.get("score", 0.0)
    for i in range(1, len(candidates)):
        s = candidates[i].verifier_result.get("score", 0.0)
        if s > best_score:
            best_score = s
            best_idx = i
    return candidates[best_idx], candidates


def candidate_to_jsonable(c: Candidate) -> dict:
    return asdict(c)
