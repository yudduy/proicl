"""Rollout accounting (PROPOSAL §5.4 — primary evaluation axis).

A rollout is ONE candidate trajectory generated from the base model. Per
PROPOSAL §5.4:

```text
RolloutTotal(POLARIS, N) = ArchiveConstructionRollouts + N * InferenceRolloutsPerQuery
RolloutTotal(GRPO,    N) = TrainingRollouts + N
```

Break-even N is the headline. Dollars/tokens/latency stay in `costs.json`
as diagnostics, not as the primary comparison axis.

The ledger is written next to `costs.json` (same `out_dir`) so manifest
parsers can produce break-even N without touching candidate JSONLs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from polaris.io.artifacts import write_json


@dataclass
class RolloutLedger:
    """Counts of base-model rollouts charged to each phase of a run."""

    archive_construction: int = 0  # GEPA mutation + dev eval
    inference: int = 0  # per-query candidates served at eval time
    verifier: int = (
        0  # extra rollouts the verifier path consumes (e.g. independent_check)
    )
    memory_admission: int = 0  # rollouts spent during offline memory build
    memory_distillation: int = 0  # rollouts spent in LLM-distillation, if any

    rollouts_by_condition: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return (
            self.archive_construction
            + self.inference
            + self.verifier
            + self.memory_admission
            + self.memory_distillation
        )

    def charge_inference(self, condition: str, n: int = 1) -> None:
        self.inference += n
        self.rollouts_by_condition[condition] = (
            self.rollouts_by_condition.get(condition, 0) + n
        )

    def write(self, path: Path) -> None:
        payload = asdict(self)
        payload["total"] = self.total
        write_json(path, payload)
