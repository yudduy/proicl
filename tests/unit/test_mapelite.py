from __future__ import annotations

from dataclasses import dataclass

import pytest

from polaris.core.archive import FrozenArchive, PromptEntry
from polaris.core.descriptor import DESCRIPTOR_CATEGORIES, classify_trace
from polaris.core.mapelite import MapEliteGrid, run_mapelite


def _p(id_, prefix="P:", descriptor="direct_computation"):
    return PromptEntry(id=id_, prefix=prefix, suffix="", descriptor_hint=descriptor)


def test_grid_starts_empty():
    g = MapEliteGrid()
    assert g.cell_fitness() == {}
    assert g.freeze().k == 0


def test_grid_update_first_time_in_cell_accepts():
    g = MapEliteGrid()
    accepted = g.update("direct_computation", _p("a"), 0.4)
    assert accepted is True
    assert g.cell_fitness()["direct_computation"] == 0.4


def test_grid_update_replaces_strictly_higher_fitness():
    g = MapEliteGrid()
    g.update("direct_computation", _p("a"), 0.4)
    accepted = g.update("direct_computation", _p("b"), 0.5)
    assert accepted is True
    archive = g.freeze()
    assert archive.entries[0].id == "b"


def test_grid_update_rejects_equal_or_lower_fitness():
    g = MapEliteGrid()
    g.update("direct_computation", _p("a"), 0.4)
    assert g.update("direct_computation", _p("b"), 0.4) is False
    assert g.update("direct_computation", _p("c"), 0.3) is False
    assert g.freeze().entries[0].id == "a"


def test_grid_update_rejects_unknown_descriptor():
    g = MapEliteGrid()
    with pytest.raises(ValueError, match="descriptor"):
        g.update("not_a_real_category", _p("a"), 0.5)


def test_grid_freeze_orders_cells_by_descriptor_category_order():
    g = MapEliteGrid()
    # insert out of order
    g.update("stepwise_decomposition", _p("step"), 0.3)
    g.update("direct_computation", _p("direct"), 0.4)
    g.update("backward_verification", _p("verify"), 0.5)
    archive = g.freeze()
    ids = [e.id for e in archive.entries]
    # output order matches DESCRIPTOR_CATEGORIES, skipping empty cells
    assert ids == ["direct", "verify", "step"]


# --- run_mapelite (iterations=0) ---


@dataclass
class _DummyGen:
    generation: str
    response_contains_prompt: bool = False
    prompt_token_count: int = 1
    generation_token_count: int = 1
    wall_clock_seconds: float = 0.0
    estimated_dollar_cost: float = 0.0
    acceptance_ratio: float | None = None


class _CannedSampler:
    """Returns canned generations from a (prompt_id, problem_idx) -> str map."""

    def __init__(self, gens_by_problem):
        self._gens = gens_by_problem  # dict[(prompt_id, problem_idx), str]
        self._counter = 0

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        # extract prompt_id from prefix "P_<id>: " (test fixture)
        gen = self._gens.get(self._counter, "no-answer")
        self._counter += 1
        return _DummyGen(generation=gen)


@dataclass
class _DummyProblem:
    prompt: str
    answer: str
    problem_id: str = "x"


def _exact_match_scorer(full_response, reference):
    return {"score": 1.0 if reference in full_response else 0.0}


def test_run_mapelite_iterations_zero_evaluates_each_seed():
    seeds = (
        _p("direct", prefix="D: ", descriptor="direct_computation"),
        _p("algebraic", prefix="A: ", descriptor="algebraic_transformation"),
    )
    dev = [
        _DummyProblem(prompt="q1", answer="42"),
        _DummyProblem(prompt="q2", answer="7"),
    ]
    # direct gets both right (2/2 = 1.0), algebraic gets 1/2 = 0.5
    gens = {0: "42", 1: "7", 2: "42", 3: "wrong"}
    sampler = _CannedSampler(gens)
    grid = run_mapelite(
        seeds=seeds,
        dev_set=dev,
        sampler=sampler,
        scorer=_exact_match_scorer,
        descriptor_fn=classify_trace,
        n_iterations=0,
    )
    fitness = grid.cell_fitness()
    assert fitness["direct_computation"] == 1.0
    assert fitness["algebraic_transformation"] == 0.5


def test_run_mapelite_iterations_greater_than_zero_uses_reflection_hook():
    def proposer(prompt, iteration, grid):
        return _p(
            f"{prompt.id}_mutated_{iteration}",
            prefix="M: ",
            descriptor="direct_computation",
        )

    grid = run_mapelite(
        seeds=(_p("a", prefix="A: "),),
        dev_set=[_DummyProblem(prompt="q", answer="42")],
        sampler=_CannedSampler({0: "wrong", 1: "42"}),
        scorer=_exact_match_scorer,
        descriptor_fn=classify_trace,
        n_iterations=1,
        reflection_lm=proposer,
    )
    assert grid.freeze().entries[0].id == "a_mutated_0"


def test_run_mapelite_returns_grid_with_frozen_archive_emission():
    seeds = (_p("d", descriptor="direct_computation"),)
    dev = [_DummyProblem(prompt="q", answer="42")]
    sampler = _CannedSampler({0: "42"})
    grid = run_mapelite(
        seeds=seeds,
        dev_set=dev,
        sampler=sampler,
        scorer=_exact_match_scorer,
        descriptor_fn=classify_trace,
        n_iterations=0,
    )
    archive = grid.freeze()
    assert isinstance(archive, FrozenArchive)
    assert archive.k == 1
    assert archive.entries[0].id == "d"
