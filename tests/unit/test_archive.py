from __future__ import annotations

import pytest

from polaris.core.archive import MATH500_ARCHIVE_V1, FrozenArchive, PromptEntry


def _archive(ids):
    return FrozenArchive(
        entries=tuple(
            PromptEntry(id=i, prefix=f"{i}: ", suffix="", descriptor_hint="d")
            for i in ids
        )
    )


def test_allocate_parity():
    arc = _archive(["a", "b", "c", "d"])
    assert arc.allocate(8) == {"a": 2, "b": 2, "c": 2, "d": 2}


def test_allocate_remainder_follows_archive_id_order():
    # §199: floor(B/k) to each entry, remainder by archive_id order
    arc = _archive(["a", "b", "c", "d"])
    assert arc.allocate(9) == {"a": 3, "b": 2, "c": 2, "d": 2}
    assert arc.allocate(10) == {"a": 3, "b": 3, "c": 2, "d": 2}
    assert arc.allocate(3) == {"a": 1, "b": 1, "c": 1, "d": 0}


def test_allocate_zero_budget():
    arc = _archive(["a", "b"])
    assert arc.allocate(0) == {"a": 0, "b": 0}


def test_allocate_negative_raises():
    arc = _archive(["a"])
    with pytest.raises(ValueError):
        arc.allocate(-1)


def test_empty_archive_allocates_empty_dict():
    arc = FrozenArchive(entries=())
    assert arc.k == 0
    assert arc.allocate(5) == {}


def test_duplicate_ids_raises():
    with pytest.raises(ValueError, match="duplicate prompt ids"):
        FrozenArchive(
            entries=(
                PromptEntry(id="x", prefix="", suffix="", descriptor_hint="d"),
                PromptEntry(id="x", prefix="", suffix="", descriptor_hint="d"),
            )
        )


def test_compose_concatenates_prefix_question_suffix():
    e = PromptEntry(id="i", prefix="P:", suffix=":S", descriptor_hint="d")
    assert e.compose("Q") == "P:Q:S"


def test_math500_archive_v1_shape():
    arc = MATH500_ARCHIVE_V1
    assert arc.k == 4
    ids = [e.id for e in arc.entries]
    assert ids == ["direct", "algebraic", "verify", "stepwise"]
    # every entry has a descriptor hint; v1 selector is metadata-only
    for e in arc.entries:
        assert e.descriptor_hint
    # memory disabled in v1 archive
    assert arc.max_retrieved_memory_entries == 0
    assert arc.max_retrieved_memory_tokens == 0
