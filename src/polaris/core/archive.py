from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class PromptEntry:
    """A single entry in a frozen prompt archive.

    `prefix` is concatenated to the question at inference time:
        full_prompt = prefix + question + suffix
    `descriptor_hint` records the trace-derived reasoning style this prompt
    targets (proposal §4). It is metadata only; not used by the v1 selector.
    """

    id: str
    prefix: str
    suffix: str
    descriptor_hint: str

    def compose(self, question: str) -> str:
        return f"{self.prefix}{question}{self.suffix}"


@dataclass(frozen=True)
class FrozenArchive:
    """Bounded, frozen archive of prompt entries.

    No online memory in v1 (proposal §"Next runnable POLARIS slice" point 1).
    Uniform allocation with stable archive_id-order remainder rule per §199.
    """

    entries: tuple[PromptEntry, ...]
    max_retrieved_memory_entries: int = 0
    max_retrieved_memory_tokens: int = 0

    def __post_init__(self) -> None:
        ids = [e.id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate prompt ids in archive: {ids}")

    @property
    def k(self) -> int:
        return len(self.entries)

    def allocate(self, total_samples: int) -> dict[str, int]:
        """Uniform allocation: floor(B / k) to each entry, remainder by archive_id order."""
        if total_samples < 0:
            raise ValueError(f"total_samples must be >= 0, got {total_samples}")
        k = self.k
        if k == 0:
            return {}
        base, remainder = divmod(total_samples, k)
        allocation: dict[str, int] = {}
        for i, entry in enumerate(self.entries):
            allocation[entry.id] = base + (1 if i < remainder else 0)
        return allocation

    def to_jsonable(self) -> list[dict]:
        return [
            {
                "id": e.id,
                "prefix": e.prefix,
                "suffix": e.suffix,
                "descriptor_hint": e.descriptor_hint,
            }
            for e in self.entries
        ]

    @classmethod
    def from_entries(cls, entries: Iterable[dict]) -> "FrozenArchive":
        return cls(
            entries=tuple(
                PromptEntry(
                    id=e["id"],
                    prefix=e["prefix"],
                    suffix=e["suffix"],
                    descriptor_hint=e["descriptor_hint"],
                )
                for e in entries
            )
        )


# Pre-registered MATH500 archive for the v1 POLARIS-MATH500 slice.
# k=4 prompts chosen for behavioral diversity across the proposal §4
# descriptor categories. Frozen before held-out evaluation.
#
# The wording reuses vendored RWS prompt scaffolding (BASE/COT in
# polaris.vendored.rws.constants) so the archive composes cleanly with the
# RWS sampler grammar (terminating \boxed{} marker is required by the
# math verifier's parse_answer).
MATH500_ARCHIVE_V1 = FrozenArchive(
    entries=(
        PromptEntry(
            id="direct",
            prefix="Solve this math problem. ",
            suffix=" Show the key calculation and put your final answer within \\boxed{}.",
            descriptor_hint="direct_computation",
        ),
        PromptEntry(
            id="algebraic",
            prefix="Solve this math problem by transforming the expression algebraically before computing. ",
            suffix=" Show each transformation step and put your final answer within \\boxed{}.",
            descriptor_hint="algebraic_transformation",
        ),
        PromptEntry(
            id="verify",
            prefix="Solve this math problem, then verify by substituting your answer back. ",
            suffix=" Show the solution and the verification, and put your final answer within \\boxed{}.",
            descriptor_hint="backward_verification",
        ),
        PromptEntry(
            id="stepwise",
            prefix="Solve this math problem step by step, naming each step. ",
            suffix=" Number each step explicitly and put your final answer within \\boxed{}.",
            descriptor_hint="stepwise_decomposition",
        ),
    )
)
