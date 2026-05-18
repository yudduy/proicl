from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from polaris.vendored.ace.playbook_utils import (
    apply_curator_operations,
    update_bullet_counts,
)
from polaris.vendored.dc.utils.extractor import extract_cheatsheet


@dataclass
class DynamicCheatsheetState:
    """Minimal DC cumulative/retrieval state machine from vendored source.

    The upstream Dynamic Cheatsheet implementation stores a running cheatsheet
    and extracts new sheets from model responses with `extract_cheatsheet`.
    This class isolates that comparator state without importing provider code.
    """

    cheatsheet: str = ""

    def update_from_response(self, response: str) -> str:
        self.cheatsheet = extract_cheatsheet(response, self.cheatsheet)
        return self.cheatsheet

    def render_prompt_context(self, *, query: str, retrieved: Iterable[str] = ()) -> str:
        blocks = [self.cheatsheet.strip()] if self.cheatsheet.strip() else []
        blocks.extend(item.strip() for item in retrieved if item.strip())
        return "\n\n".join(blocks + [query])


@dataclass
class ACEPlaybookState:
    """ACE comparator playbook state extracted from official source.

    Upstream ACE evolves a playbook by tagging existing bullets as helpful or
    harmful and applying curator operations. POLARIS uses this class only as the
    baseline state machine; live LLM calls remain behind production preflight.
    """

    playbook: str
    next_global_id: int = 1

    @classmethod
    def empty(cls) -> "ACEPlaybookState":
        return cls(
            playbook=(
                "## STRATEGIES & INSIGHTS\n\n"
                "## FORMULAS & CALCULATIONS\n\n"
                "## CODE SNIPPETS & TEMPLATES\n\n"
                "## COMMON MISTAKES TO AVOID\n\n"
                "## PROBLEM-SOLVING HEURISTICS\n\n"
                "## CONTEXT CLUES & INDICATORS\n\n"
                "## OTHERS"
            )
        )

    def apply_feedback(
        self,
        *,
        bullet_tags: list[dict[str, Any]],
        curator_operations: list[dict[str, Any]],
    ) -> str:
        self.playbook = update_bullet_counts(self.playbook, bullet_tags)
        self.playbook, self.next_global_id = apply_curator_operations(
            self.playbook,
            curator_operations,
            self.next_global_id,
        )
        return self.playbook


@dataclass(frozen=True)
class PublishedComparatorLedger:
    name: str
    source: str
    training_rollouts: int
    inference_rollouts_per_query: int
    notes: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


PUBLISHED_COMPARATORS: dict[str, PublishedComparatorLedger] = {
    "published_p2o": PublishedComparatorLedger(
        name="published_p2o",
        source="proposal_reference",
        training_rollouts=0,
        inference_rollouts_per_query=1,
        notes="Populate with paper-faithful P2O rollout accounting before final analysis.",
    ),
    "published_grpo_ledger": PublishedComparatorLedger(
        name="published_grpo_ledger",
        source="proposal_reference",
        training_rollouts=0,
        inference_rollouts_per_query=1,
        notes="Populate with paper-faithful GRPO rollout accounting before final analysis.",
    ),
}
