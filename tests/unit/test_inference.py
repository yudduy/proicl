from __future__ import annotations

from dataclasses import dataclass

import pytest

from polaris.core.archive import FrozenArchive, PromptEntry
from polaris.core.inference import polaris_inference
from polaris.core.mixed_alpha import FIXED_ALPHA_4, AlphaSchedule
from polaris.io.trajectory_cache import TrajectoryCache, TrajectoryKey, TrajectoryRecord


@dataclass
class _DummyGen:
    generation: str
    response_contains_prompt: bool = False
    prompt_token_count: int = 5
    generation_token_count: int = 5
    wall_clock_seconds: float = 0.01
    estimated_dollar_cost: float = 0.0
    acceptance_ratio: float | None = None


class _RecordingSampler:
    """Returns canned generations indexed by (prompt_text, call_idx).

    Records all calls so tests can assert on path taken (power vs low_temp).
    """

    def __init__(self, generations):
        self._generations = generations
        self.power_calls = []
        self.low_temp_calls = []

    def _pop(self):
        return _DummyGen(generation=self._generations.pop(0))

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        self.power_calls.append((prompt_text, temperature, mcmc_steps, block_num))
        return self._pop()

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        self.low_temp_calls.append((prompt_text, temperature))
        return self._pop()


def _scorer(score_map):
    """Return a scorer that maps full_response -> score via dict lookup, default 0.0."""

    def _score(full_response, reference):
        return {"score": score_map.get(full_response.strip(), 0.0)}

    return _score


def _archive(ids):
    return FrozenArchive(
        entries=tuple(
            PromptEntry(id=i, prefix=f"[{i}] ", suffix="", descriptor_hint="d")
            for i in ids
        )
    )


def test_argmax_selector_picks_highest_score():
    arc = _archive(["a", "b"])
    sampler = _RecordingSampler(["gen-a", "gen-b"])
    # full_response = prompt_text + generation when response_contains_prompt=False
    # full_response is plain concat (no separator): prompt_text + generation
    scorer = _scorer({"[a] Qgen-a": 0.3, "[b] Qgen-b": 0.9})
    best, all_c = polaris_inference(
        question="Q",
        reference="ref",
        archive=arc,
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=2,
        max_new_tokens=64,
        scorer=scorer,
    )
    assert len(all_c) == 2
    assert best.prompt_id == "b"
    assert best.verifier_result["score"] == 0.9


def test_argmax_selector_iteration_tiebreak_first_wins():
    arc = _archive(["a", "b"])
    sampler = _RecordingSampler(["g1", "g2"])
    scorer = _scorer({"[a] Qg1": 0.5, "[b] Qg2": 0.5})  # tied
    best, all_c = polaris_inference(
        question="Q",
        reference="ref",
        archive=arc,
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=2,
        max_new_tokens=64,
        scorer=scorer,
    )
    # iteration order is archive_id-order, "a" comes first, "a" wins the tie
    assert best.prompt_id == "a"


def test_alpha_one_short_circuits_to_low_temp_path():
    arc = _archive(["a"])
    sampler = _RecordingSampler(["g"])
    scorer = _scorer({"[a] Q g": 1.0})
    schedule_alpha_1 = AlphaSchedule(policy_id="alpha_1_only", alphas=(1.0,))
    polaris_inference(
        question="Q",
        reference="ref",
        archive=arc,
        sampler=sampler,
        alpha_schedule=schedule_alpha_1,
        total_samples=1,
        max_new_tokens=64,
        scorer=scorer,
        skip_alpha_one_mcmc=True,
    )
    assert len(sampler.low_temp_calls) == 1
    assert len(sampler.power_calls) == 0


def test_alpha_one_takes_mcmc_path_when_skip_disabled():
    arc = _archive(["a"])
    sampler = _RecordingSampler(["g"])
    scorer = _scorer({"[a] Q g": 1.0})
    schedule_alpha_1 = AlphaSchedule(policy_id="alpha_1_only", alphas=(1.0,))
    polaris_inference(
        question="Q",
        reference="ref",
        archive=arc,
        sampler=sampler,
        alpha_schedule=schedule_alpha_1,
        total_samples=1,
        max_new_tokens=64,
        scorer=scorer,
        skip_alpha_one_mcmc=False,
    )
    assert len(sampler.power_calls) == 1
    assert len(sampler.low_temp_calls) == 0


def test_alpha_4_uses_power_path():
    arc = _archive(["a"])
    sampler = _RecordingSampler(["g"])
    scorer = _scorer({"[a] Q g": 1.0})
    polaris_inference(
        question="Q",
        reference="ref",
        archive=arc,
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=1,
        max_new_tokens=64,
        scorer=scorer,
    )
    assert len(sampler.power_calls) == 1
    # temperature passed to power path is 1/alpha = 0.25
    assert sampler.power_calls[0][1] == pytest.approx(0.25)


def test_memory_enabled_without_store_behaves_like_no_memory():
    arc_with_memory = FrozenArchive(
        entries=(PromptEntry(id="a", prefix="", suffix="", descriptor_hint="d"),),
        max_retrieved_memory_entries=1,
    )
    sampler = _RecordingSampler(["g"])
    best, candidates = polaris_inference(
        question="Q",
        reference="ref",
        archive=arc_with_memory,
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=1,
        max_new_tokens=64,
        scorer=_scorer({"Qg": 1.0}),
    )
    assert len(candidates) == 1
    assert best.retrieved_memory_ids == []


def test_zero_total_samples_raises():
    arc = _archive(["a"])
    with pytest.raises(ValueError, match="no candidates"):
        polaris_inference(
            question="Q",
            reference="ref",
            archive=arc,
            sampler=_RecordingSampler([]),
            alpha_schedule=FIXED_ALPHA_4,
            total_samples=0,
            max_new_tokens=64,
            scorer=_scorer({}),
        )


def test_response_contains_prompt_uses_generation_as_full_response():
    arc = _archive(["a"])
    sampler = _RecordingSampler([])

    def gen_with_prompt(prompt_text, *, temperature, max_new_tokens):
        return _DummyGen(generation="full-with-prompt", response_contains_prompt=True)

    sampler.generate_power = gen_with_prompt
    seen = []

    def scorer(full_response, reference):
        seen.append(full_response)
        return {"score": 1.0}

    polaris_inference(
        question="Q",
        reference="ref",
        archive=arc,
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=1,
        max_new_tokens=64,
        scorer=scorer,
    )
    # because response_contains_prompt=True, scorer sees generation as-is (no concat)
    assert seen == ["full-with-prompt"]


def test_cache_hit_replays_candidate_without_sampling(tmp_path):
    arc = _archive(["a"])
    cache = TrajectoryCache(tmp_path / "trajectories.sqlite")
    prompt_text = "[a] Q"
    cache.put(
        TrajectoryRecord(
            key=TrajectoryKey(
                model_id="model",
                track="math500",
                problem_id="p0",
                prompt_id="a",
                sample_idx=0,
                alpha=4.0,
                seed=17,
            ),
            prompt_hash=TrajectoryCache.prompt_hash(prompt_text),
            generation="cached",
            response_contains_prompt=False,
            token_ids=[1, 2],
            logprobs_norm=[-1.0],
            logprobs_unnorm=[-0.25],
            acceptance_ratio=0.5,
            prompt_token_count=3,
            generation_token_count=2,
            wall_clock_seconds=9.0,
            dollar_cost=0.2,
            verifier_result={"score": 1.0, "passed": True},
        )
    )
    sampler = _RecordingSampler(["should-not-be-used"])

    best, all_c = polaris_inference(
        question="Q",
        reference="ref",
        archive=arc,
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=1,
        max_new_tokens=64,
        scorer=_scorer({}),
        trajectory_cache=cache,
        cache_model_id="model",
        cache_track="math500",
        cache_problem_id="p0",
        cache_seed=17,
    )

    assert len(sampler.power_calls) == 0
    assert len(sampler.low_temp_calls) == 0
    assert best.generation == "cached"
    assert best.acceptance_ratio == 0.5
    assert all_c[0].estimated_dollar_cost == 0.2


def test_cache_requires_unambiguous_context(tmp_path):
    arc = _archive(["a"])
    with pytest.raises(ValueError, match="cache_model_id"):
        polaris_inference(
            question="Q",
            reference="ref",
            archive=arc,
            sampler=_RecordingSampler(["g"]),
            alpha_schedule=FIXED_ALPHA_4,
            total_samples=1,
            max_new_tokens=64,
            scorer=_scorer({}),
            trajectory_cache=TrajectoryCache(tmp_path / "trajectories.sqlite"),
        )
