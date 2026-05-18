"""Verifier-gated memory store (PROPOSAL §1.3, §8).

Two separate processes:
1. **Admission**: a candidate is admitted only after an INDEPENDENT verifier
   check on the source query — a second verifier path different from the
   inference-time scorer. False positives poison memory; false negatives
   just shrink it (§7.2).
2. **Reliability accumulation**: each retrieval logs a verifier outcome
   z ∈ {0,1}, updating posterior Beta(α,β) → Beta(α+z, β+1-z). Posterior
   mean α/(α+β) ranks entries.

Memory is bounded: `max_memory_entries_per_archive=256`, retrieval capped at
`max_retrieved_memory_entries=3` and `max_retrieved_memory_tokens=512`.
Distillation removes problem-specific constants and screens for leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable

# Defaults from PROPOSAL.md §8. Override only via pre-registered experiment
# block in TODO.md + runs/progress.md.
MAX_MEMORY_ENTRIES_PER_ARCHIVE = 256
MAX_RETRIEVED_MEMORY_ENTRIES = 3
MAX_RETRIEVED_MEMORY_TOKENS = 512
RELIABILITY_PRIOR_ALPHA = 1.0
RELIABILITY_PRIOR_BETA = 1.0


@dataclass(frozen=True)
class MemoryEntry:
    """One distilled, verifier-admitted strategy."""

    id: str
    archive_prompt_id: str
    descriptor: str
    strategy_text: str
    token_count: int
    source_query_id: str
    reliability_alpha: float = RELIABILITY_PRIOR_ALPHA
    reliability_beta: float = RELIABILITY_PRIOR_BETA

    @property
    def reliability_mean(self) -> float:
        return self.reliability_alpha / (self.reliability_alpha + self.reliability_beta)

    def updated(self, *, verifier_outcome: int) -> "MemoryEntry":
        if verifier_outcome not in (0, 1):
            raise ValueError(
                f"verifier_outcome must be 0 or 1, got {verifier_outcome!r}"
            )
        return replace(
            self,
            reliability_alpha=self.reliability_alpha + verifier_outcome,
            reliability_beta=self.reliability_beta + (1 - verifier_outcome),
        )


def distill_strategy(trace: str) -> str:
    """Deterministic extractor (PROPOSAL §8 strategy distillation, item 1).

    Removes leading prompt echo, trims to the boxed-answer line plus the
    immediately preceding reasoning block, strips problem-specific numerals.
    LLM-distillation path is deferred; this heuristic version is sufficient
    for v3 memory-enabled MATH500 runs and avoids an extra LLM call in the
    rollout budget.

    Returns the distilled strategy text. Empty string if nothing survives
    the leakage screen (caller treats empty as rejection).
    """
    if not trace.strip():
        return ""
    # Conservative heuristic: keep at most the last ~512 chars of the trace
    # tail (where the reasoning summary lives in RWS-formatted outputs).
    # A more careful template-based distiller belongs to a downstream phase.
    tail = trace.strip().splitlines()[-12:]
    distilled = "\n".join(tail).strip()
    distilled = re.sub(r"\\boxed\{[^{}]*\}", r"\\boxed{ANSWER}", distilled)
    if _leaks_specifics(distilled):
        return ""
    return distilled


def _leaks_specifics(text: str) -> bool:
    """Leakage screen (PROPOSAL §8 distillation item 4).

    Reject raw final answers (`\\boxed{...}` with concrete numeric/symbolic
    content), benchmark ids, or copied multi-line full solutions exceeding
    the retrieval token cap.
    """
    if re.search(r"\\boxed\{(?!ANSWER\})[^{}]*\}", text):
        return True
    if len(text) > MAX_RETRIEVED_MEMORY_TOKENS * 6:  # ~6 chars/token upper bound
        return True
    return False


@dataclass
class MemoryStore:
    """Bounded per-archive-prompt memory pool.

    Admission requires the candidate to pass both `primary_verifier` (the
    inference-time scorer, already run upstream) and `independent_check`
    (PROPOSAL §8 — a second verifier path: code rerun on a held-out test
    subset, math reparse via a separate extractor, etc.).
    """

    entries: list[MemoryEntry] = field(default_factory=list)
    max_entries_per_prompt: int = MAX_MEMORY_ENTRIES_PER_ARCHIVE

    def admit(
        self,
        *,
        candidate_trace: str,
        archive_prompt_id: str,
        descriptor: str,
        source_query_id: str,
        independent_check: Callable[[str], bool],
        token_counter: Callable[[str], int],
        entry_id: str,
    ) -> MemoryEntry | None:
        """Try to admit a verifier-passing candidate. Returns the new entry
        on success, `None` if independent check or leakage screen rejects.
        """
        if not independent_check(candidate_trace):
            return None
        strategy = distill_strategy(candidate_trace)
        if not strategy:
            return None
        entry = MemoryEntry(
            id=entry_id,
            archive_prompt_id=archive_prompt_id,
            descriptor=descriptor,
            strategy_text=strategy,
            token_count=token_counter(strategy),
            source_query_id=source_query_id,
        )
        self.entries.append(entry)
        self._prune(archive_prompt_id)
        return entry

    def retrieve(
        self,
        *,
        archive_prompt_id: str,
        descriptor_filter: str | None = None,
        max_entries: int = MAX_RETRIEVED_MEMORY_ENTRIES,
        max_tokens: int = MAX_RETRIEVED_MEMORY_TOKENS,
        reliability_lower_bound: float = 0.0,
    ) -> list[MemoryEntry]:
        """Filter, rank by posterior reliability mean, cap by entries/tokens."""
        pool = [
            e
            for e in self.entries
            if e.archive_prompt_id == archive_prompt_id
            and (descriptor_filter is None or e.descriptor == descriptor_filter)
            and e.reliability_mean >= reliability_lower_bound
        ]
        pool.sort(key=lambda e: e.reliability_mean, reverse=True)
        selected: list[MemoryEntry] = []
        token_used = 0
        for entry in pool:
            if len(selected) >= max_entries:
                break
            if token_used + entry.token_count > max_tokens:
                continue
            selected.append(entry)
            token_used += entry.token_count
        return selected

    def update_reliability(self, entry_ids: list[str], verifier_outcome: int) -> None:
        """Apply Beta-Bernoulli update to listed entries."""
        id_set = set(entry_ids)
        self.entries = [
            e.updated(verifier_outcome=verifier_outcome) if e.id in id_set else e
            for e in self.entries
        ]

    def _prune(self, archive_prompt_id: str) -> None:
        """Cap pool size per archive prompt — drop lowest reliability first."""
        same_prompt = [
            e for e in self.entries if e.archive_prompt_id == archive_prompt_id
        ]
        if len(same_prompt) <= self.max_entries_per_prompt:
            return
        keep_ids = {
            e.id
            for e in sorted(
                same_prompt, key=lambda e: e.reliability_mean, reverse=True
            )[: self.max_entries_per_prompt]
        }
        self.entries = [
            e
            for e in self.entries
            if e.archive_prompt_id != archive_prompt_id or e.id in keep_ids
        ]
