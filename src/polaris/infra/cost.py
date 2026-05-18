from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CostProjection:
    observed_queries: int
    target_queries: int
    observed_dollars: float
    observed_wall_clock_seconds: float
    projected_dollars: float
    projected_wall_clock_seconds: float
    cost_cap_dollars: float
    under_cap: bool

    def to_jsonable(self) -> dict:
        return asdict(self)


def project_cost_from_observed(
    *,
    observed_queries: int,
    target_queries: int,
    observed_dollars: float,
    observed_wall_clock_seconds: float,
    cost_cap_dollars: float,
    fixed_dollars: float = 0.0,
    fixed_wall_clock_seconds: float = 0.0,
) -> CostProjection:
    if observed_queries <= 0:
        raise ValueError("observed_queries must be positive")
    if target_queries < 0:
        raise ValueError("target_queries must be non-negative")
    if observed_dollars < 0 or fixed_dollars < 0:
        raise ValueError("dollar costs must be non-negative")
    scale = target_queries / observed_queries
    projected_dollars = fixed_dollars + observed_dollars * scale
    projected_wall = fixed_wall_clock_seconds + observed_wall_clock_seconds * scale
    return CostProjection(
        observed_queries=observed_queries,
        target_queries=target_queries,
        observed_dollars=observed_dollars,
        observed_wall_clock_seconds=observed_wall_clock_seconds,
        projected_dollars=projected_dollars,
        projected_wall_clock_seconds=projected_wall,
        cost_cap_dollars=cost_cap_dollars,
        under_cap=projected_dollars <= cost_cap_dollars,
    )


def assert_cost_under_cap(projection: CostProjection) -> None:
    if not projection.under_cap:
        raise ValueError(
            "projected final cost exceeds cap "
            f"({projection.projected_dollars:.4f} > {projection.cost_cap_dollars:.4f})"
        )
