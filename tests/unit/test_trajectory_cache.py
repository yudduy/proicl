from __future__ import annotations

import pytest

from polaris.io.trajectory_cache import (
    TrajectoryCache,
    TrajectoryKey,
    TrajectoryRecord,
)


def _record(generation: str = "g", passed: bool | None = True) -> TrajectoryRecord:
    key = TrajectoryKey(
        model_id="m",
        track="math500",
        problem_id="p0",
        prompt_id="direct",
        sample_idx=0,
        alpha=4.0,
        seed=17,
    )
    verifier_result = None if passed is None else {"passed": passed, "score": float(passed)}
    return TrajectoryRecord(
        key=key,
        prompt_hash=TrajectoryCache.prompt_hash("prompt"),
        generation=generation,
        response_contains_prompt=False,
        token_ids=[1, 2, 3],
        logprobs_norm=[-1.0, -2.0],
        logprobs_unnorm=[-0.25, -0.5],
        acceptance_ratio=0.75,
        prompt_token_count=5,
        generation_token_count=3,
        wall_clock_seconds=1.5,
        dollar_cost=0.001,
        verifier_result=verifier_result,
    )


def test_put_get_roundtrip(tmp_path):
    cache = TrajectoryCache(tmp_path / "trajectories.sqlite")
    rec = _record()
    cache.put(rec)
    got = cache.get(
        model_id="m",
        track="math500",
        problem_id="p0",
        prompt_id="direct",
        sample_idx=0,
        alpha=4.0,
        seed=17,
    )
    assert got is not None
    assert got.key == rec.key
    assert got.generation == "g"
    assert got.token_ids == [1, 2, 3]
    assert got.logprobs_norm == pytest.approx([-1.0, -2.0], abs=1e-3)
    assert got.logprobs_unnorm == pytest.approx([-0.25, -0.5], abs=1e-3)
    assert got.verifier_result == {"passed": True, "score": 1.0}


def test_duplicate_rejected_unless_overwrite(tmp_path):
    cache = TrajectoryCache(tmp_path / "trajectories.sqlite")
    cache.put(_record("first"))
    with pytest.raises(ValueError, match="already cached"):
        cache.put(_record("second"))
    cache.put(_record("second"), overwrite=True)
    got = cache.get(
        model_id="m",
        track="math500",
        problem_id="p0",
        prompt_id="direct",
        sample_idx=0,
        alpha=4.0,
        seed=17,
    )
    assert got is not None
    assert got.generation == "second"


def test_mark_verified_updates_existing_row(tmp_path):
    cache = TrajectoryCache(tmp_path / "trajectories.sqlite")
    rec = _record(passed=None)
    cache.put(rec)
    cache.mark_verified(rec.key, verifier_result={"passed": False, "score": 0.0})
    got = cache.get(
        model_id="m",
        track="math500",
        problem_id="p0",
        prompt_id="direct",
        sample_idx=0,
        alpha=4.0,
        seed=17,
    )
    assert got is not None
    assert got.verifier_result == {"passed": False, "score": 0.0}


def test_iter_for_prompt_orders_by_alpha_then_sample(tmp_path):
    cache = TrajectoryCache(tmp_path / "trajectories.sqlite")
    rec = _record("a")
    cache.put(rec)
    cache.put(
        TrajectoryRecord(
            key=TrajectoryKey(
                model_id="m",
                track="math500",
                problem_id="p0",
                prompt_id="direct",
                sample_idx=1,
                alpha=1.0,
                seed=17,
            ),
            prompt_hash=rec.prompt_hash,
            generation="b",
            response_contains_prompt=False,
            token_ids=[],
            logprobs_norm=[],
            logprobs_unnorm=[],
            acceptance_ratio=None,
            prompt_token_count=5,
            generation_token_count=1,
            wall_clock_seconds=0.1,
            dollar_cost=0.0,
            verifier_result={"passed": True, "score": 1.0},
        )
    )
    rows = list(
        cache.iter_for_prompt(
            model_id="m",
            track="math500",
            problem_id="p0",
            prompt_id="direct",
            seed=17,
        )
    )
    assert [row.generation for row in rows] == ["b", "a"]
