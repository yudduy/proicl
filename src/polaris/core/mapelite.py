from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from polaris.core.archive import FrozenArchive, PromptEntry
from polaris.core.descriptor import DESCRIPTOR_CATEGORIES


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


@dataclass
class MapEliteGrid:
    """Descriptor-cell quality-diversity grid (proposal §R10).

    Cells are the four descriptor categories from §4. Each cell holds the
    highest-fitness prompt observed so far. Mutation iterations (v2) propose
    new prompts; placement is by trace descriptor; replacement requires
    strict fitness improvement.
    """

    cells: dict[str, tuple[PromptEntry, float]] = field(default_factory=dict)

    def update(self, descriptor: str, prompt: PromptEntry, fitness: float) -> bool:
        if descriptor not in DESCRIPTOR_CATEGORIES:
            raise ValueError(
                f"unknown descriptor category: {descriptor!r} "
                f"(must be one of {DESCRIPTOR_CATEGORIES})"
            )
        incumbent = self.cells.get(descriptor)
        if incumbent is None or fitness > incumbent[1]:
            self.cells[descriptor] = (prompt, fitness)
            return True
        return False

    def cell_fitness(self) -> dict[str, float]:
        return {descriptor: fitness for descriptor, (_, fitness) in self.cells.items()}

    def freeze(self) -> FrozenArchive:
        """Emit a FrozenArchive ordered by DESCRIPTOR_CATEGORIES."""
        entries = tuple(
            self.cells[cat][0] for cat in DESCRIPTOR_CATEGORIES if cat in self.cells
        )
        return FrozenArchive(entries=entries)


def _evaluate_prompt_on_dev(
    prompt: PromptEntry,
    dev_set: Sequence[Any],
    sampler: _SamplerLike,
    scorer: Callable[[str, str], dict],
    temperature: float,
    max_new_tokens: int,
    samples_per_eval: int,
) -> float:
    """Mean accuracy of `prompt` across `dev_set` (samples_per_eval per problem)."""
    if not dev_set:
        return 0.0
    per_problem_scores: list[float] = []
    for problem in dev_set:
        prompt_text = prompt.compose(problem.prompt)
        sample_scores: list[float] = []
        for _ in range(samples_per_eval):
            gen = sampler.generate_power(
                prompt_text, temperature=temperature, max_new_tokens=max_new_tokens
            )
            full_response = (
                gen.generation
                if gen.response_contains_prompt
                else prompt_text + gen.generation
            )
            sample_scores.append(
                scorer(full_response, problem.answer).get("score", 0.0)
            )
        per_problem_scores.append(sum(sample_scores) / len(sample_scores))
    return sum(per_problem_scores) / len(per_problem_scores)


def run_mapelite(
    *,
    seeds: Sequence[PromptEntry],
    dev_set: Sequence[Any],
    sampler: _SamplerLike,
    scorer: Callable[[str, str], dict],
    descriptor_fn: Callable[[str], tuple[str, float]],
    temperature: float = 0.25,
    max_new_tokens: int = 3072,
    n_iterations: int = 0,
    samples_per_eval: int = 1,
    reflection_lm: Any = None,
) -> MapEliteGrid:
    """Construct the prompt-archive MAP-Elites grid (proposal §6).

    `n_iterations=0`: evaluate each seed on `dev_set` and place in its cell.
    This is the v1 path — the seed grid IS the frozen archive.

    `n_iterations>0`: bounded reflective mutation path for infrastructure
    smokes. `reflection_lm` must be an injected callable so local tests do not
    spend on an LLM. Signature: `(prompt, iteration, grid) -> PromptEntry`.
    """
    grid = MapEliteGrid()
    for seed in seeds:
        fitness = _evaluate_prompt_on_dev(
            prompt=seed,
            dev_set=dev_set,
            sampler=sampler,
            scorer=scorer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            samples_per_eval=samples_per_eval,
        )
        grid.update(seed.descriptor_hint, seed, fitness)

    for iteration in range(n_iterations):
        if not callable(reflection_lm):
            raise ValueError(
                "n_iterations>0 requires a callable reflection_lm/proposer for "
                "bounded infrastructure runs"
            )
        archive = grid.freeze()
        if archive.entries:
            parent = max(
                archive.entries,
                key=lambda p: grid.cell_fitness().get(p.descriptor_hint, float("-inf")),
            )
        else:
            parent = seeds[iteration % len(seeds)]
        proposal = reflection_lm(parent, iteration, grid)
        if not isinstance(proposal, PromptEntry):
            raise TypeError("reflection_lm must return a PromptEntry")
        fitness = _evaluate_prompt_on_dev(
            prompt=proposal,
            dev_set=dev_set,
            sampler=sampler,
            scorer=scorer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            samples_per_eval=samples_per_eval,
        )
        grid.update(proposal.descriptor_hint, proposal, fitness)

    return grid
