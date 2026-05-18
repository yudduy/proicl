from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


LOCK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetLock:
    track: str
    source_repo: str
    config: str
    split: str
    row_count: int
    row_id_hashes: tuple[str, ...]
    loader_version: str
    split_id: str = "final"
    status: str = "locked"
    cache_path: str | None = None
    notes: str = ""

    @property
    def lock_id(self) -> str:
        payload = json.dumps(
            {
                "track": self.track,
                "source_repo": self.source_repo,
                "config": self.config,
                "split": self.split,
                "split_id": self.split_id,
                "row_count": self.row_count,
                "row_id_hashes": list(self.row_id_hashes),
                "loader_version": self.loader_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["row_id_hashes"] = list(self.row_id_hashes)
        payload["lock_id"] = self.lock_id
        return payload


def hash_row_id(row_id: str) -> str:
    return hashlib.sha256(str(row_id).encode("utf-8")).hexdigest()


def build_dataset_lock(
    *,
    track: str,
    source_repo: str,
    config: str,
    split: str,
    rows: Iterable[Any],
    row_id: Callable[[Any, int], str],
    loader_version: str,
    split_id: str = "final",
    cache_path: str | None = None,
    status: str = "locked",
    notes: str = "",
) -> DatasetLock:
    hashes: list[str] = []
    count = 0
    for idx, row in enumerate(rows):
        hashes.append(hash_row_id(row_id(row, idx)))
        count += 1
    return DatasetLock(
        track=track,
        source_repo=source_repo,
        config=config,
        split=split,
        row_count=count,
        row_id_hashes=tuple(hashes),
        loader_version=loader_version,
        split_id=split_id,
        cache_path=cache_path,
        status=status,
        notes=notes,
    )


def read_dataset_locks(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_dataset_locks(path: Path, locks: Iterable[DatasetLock]) -> dict[str, Any]:
    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "locks": [lock.to_jsonable() for lock in locks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def find_dataset_lock(
    locks_payload: dict[str, Any],
    *,
    track: str,
    split: str | None = None,
    split_id: str | None = None,
) -> dict[str, Any] | None:
    for lock in locks_payload.get("locks", []):
        if lock.get("track") != track:
            continue
        if split is not None and lock.get("split") != split:
            continue
        if split_id is not None and lock.get("split_id") != split_id:
            continue
        return lock
    return None


def assert_locked_dataset(
    locks_payload: dict[str, Any],
    *,
    track: str,
    split: str | None = None,
    split_id: str | None = None,
) -> dict[str, Any]:
    lock = find_dataset_lock(
        locks_payload, track=track, split=split, split_id=split_id
    )
    if lock is None:
        raise ValueError(
            f"missing dataset lock for track={track!r} split={split!r} "
            f"split_id={split_id!r}"
        )
    if lock.get("status") != "locked":
        raise ValueError(f"dataset lock for {track!r} is not locked: {lock.get('status')}")
    if int(lock.get("row_count", 0)) <= 0:
        raise ValueError(f"dataset lock for {track!r} has no rows")
    if not lock.get("row_id_hashes"):
        raise ValueError(f"dataset lock for {track!r} has no row-id hashes")
    return lock


def scan_for_gpqa_leakage(root: Path) -> list[Path]:
    """Conservative tracked-file leak check.

    It intentionally ignores ignored runtime caches under `runs/`.
    """
    forbidden = ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2")
    offenders: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or ".venv" in path.parts or ".venv-eval" in path.parts:
            continue
        if path.suffix == ".py":
            continue
        if "runs" in path.parts and path.suffix in {".jsonl", ".csv", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in forbidden):
            offenders.append(path)
    return offenders
