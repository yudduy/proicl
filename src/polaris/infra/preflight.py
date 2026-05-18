from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


RunKind = Literal[
    "modal",
    "mithril",
    "flow",
    "phase",
    "farmshare",
    "cloudrift",
    "bulk",
    "local",
]


class PreflightError(RuntimeError):
    """Raised when a paid or scale-capable run is missing launch evidence."""


@dataclass(frozen=True)
class PaidRunPreflight:
    """Minimum launch contract for paid or scale-capable generation.

    This is intentionally backend-agnostic. Modal smokes, Mithril/Flow jobs,
    phase runs, and bulk generation all need the same evidence before launch:
    where artifacts go, where cache lives, what split/model/backend/seed is
    running, how much it is expected to cost, and explicit user authorization.
    """

    run_kind: RunKind
    artifact_dir: Path | None
    cache_path: Path | None
    split: tuple[int, int] | None
    seed: int | None
    model_id: str
    backend: str
    estimated_dollar_cost: float | None
    cost_cap_dollars: float | None
    user_authorized: bool
    estimated_wall_clock_seconds: float | None = None


def validate_paid_run_preflight(spec: PaidRunPreflight) -> dict:
    """Return a JSONable report or raise `PreflightError`.

    The function does not create directories or touch remote services. It is
    safe to call before imports that initialize GPU frameworks.
    """

    failures: list[str] = []

    if spec.artifact_dir is None:
        failures.append("artifact_dir is required")
    if spec.cache_path is None:
        failures.append("cache_path is required")
    if spec.split is None:
        failures.append("split is required")
    else:
        start, end = spec.split
        if start < 0 or end <= start:
            failures.append("split must satisfy 0 <= start < end")
    if spec.seed is None:
        failures.append("seed is required")
    if not spec.model_id:
        failures.append("model_id is required")
    if not spec.backend:
        failures.append("backend is required")
    if spec.estimated_dollar_cost is None:
        failures.append("estimated_dollar_cost is required")
    elif spec.estimated_dollar_cost < 0:
        failures.append("estimated_dollar_cost must be non-negative")
    elif spec.run_kind == "farmshare" and spec.estimated_dollar_cost != 0.0:
        failures.append("farmshare estimated_dollar_cost must be 0.0")
    if spec.cost_cap_dollars is None:
        failures.append("cost_cap_dollars is required")
    elif spec.cost_cap_dollars < 0:
        failures.append("cost_cap_dollars must be non-negative")
    if (
        spec.estimated_dollar_cost is not None
        and spec.cost_cap_dollars is not None
        and spec.estimated_dollar_cost > spec.cost_cap_dollars
    ):
        failures.append(
            "estimated_dollar_cost exceeds cost_cap_dollars "
            f"({spec.estimated_dollar_cost} > {spec.cost_cap_dollars})"
        )
    if spec.estimated_wall_clock_seconds is not None and spec.estimated_wall_clock_seconds < 0:
        failures.append("estimated_wall_clock_seconds must be non-negative")
    if not spec.user_authorized:
        failures.append("user_authorized must be true")

    if failures:
        raise PreflightError("; ".join(failures))

    payload = asdict(spec)
    payload["artifact_dir"] = str(spec.artifact_dir)
    payload["cache_path"] = str(spec.cache_path)
    payload["split"] = list(spec.split or ())
    payload["passed"] = True
    return payload
