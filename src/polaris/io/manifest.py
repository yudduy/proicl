from __future__ import annotations

import datetime as _dt
import hashlib
import json
import socket
from pathlib import Path
from typing import Any

from polaris.core.archive import FrozenArchive
from polaris.io.artifacts import write_json


def compute_archive_hash(archive: FrozenArchive) -> str:
    """SHA-256 of the archive's canonical JSON form.

    Recorded in the manifest so a run is unambiguously bound to its frozen
    archive (proposal §"Drift-Lock Protocol" — hyperparameter drift).
    """
    payload = json.dumps(archive.to_jsonable(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_run_manifest(
    path: Path,
    *,
    model_id: str,
    model_revision: str | None = None,
    model_revision_commit: str | None = None,
    benchmark: str,
    split: tuple[int, int],
    seeds: list[int],
    condition: str,
    archive_hash: str,
    alpha_policy_id: str,
    config: dict[str, Any],
    polaris_source_hash: str,
    vendored_commits: dict[str, str],
    verifier_id: str,
    preregistration_anchor: str,
    started_at: str | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    """Write `manifest.json` per proposal §"Required run artifacts".

    `preregistration_anchor` must be a non-empty string referencing the
    `TODO.md` / `runs/progress.md` lockup entry that pre-registered this run
    (proposal §"Drift-Lock Protocol" item 4).
    """
    if not preregistration_anchor:
        raise ValueError(
            "preregistration_anchor is required (proposal §Drift-Lock item 4); "
            "no expensive run may start without a TODO.md / progress.md lockup entry"
        )

    manifest = {
        "started_at": started_at
        or _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "host": host or socket.gethostname(),
        "model": model_id,
        "model_revision": model_revision,
        "model_revision_commit": model_revision_commit,
        "benchmark": benchmark,
        "split": list(split),
        "config": dict(config),
        "seeds": list(seeds),
        "polaris_source_hash": polaris_source_hash,
        "vendored_commits": dict(vendored_commits),
        "archive_hash": archive_hash,
        "alpha_policy_id": alpha_policy_id,
        "verifier_id": verifier_id,
        "preregistration_anchor": preregistration_anchor,
        "condition": condition,
    }
    write_json(path, manifest)
    return manifest
