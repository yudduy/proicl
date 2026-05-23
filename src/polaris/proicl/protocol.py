from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArchiveScope(str, Enum):
    TRANSDUCTIVE_SUPPORT = "transductive_support"
    WITHIN_FAMILY = "within_family"
    CROSS_FAMILY_CURRICULUM = "cross_family_curriculum"


class MemoryProtocol(str, Enum):
    OFF = "off"
    FROZEN_DEV = "frozen_dev"
    ONLINE = "online"
    DIAGNOSTIC_PRORL_TRANSPLANT = "diagnostic_prorl_transplant"


class RepairMode(str, Enum):
    OFF = "off"
    VERIFIER_GUIDED = "verifier_guided"


class ForkSearchMode(str, Enum):
    OFF = "off"
    ENTROPY_GATED = "entropy_gated"


@dataclass(frozen=True)
class ProICLConditionSpec:
    key: str
    runtime_condition: str
    model_key: str
    archive_kind: str
    budget_role: str
    alpha_policy: str
    uses_power_sampling: bool = False
    uses_gepa_archive: bool = False
    uses_memory: bool = False
    uses_repair: bool = False
    uses_fork_search: bool = False
    slow_weight_reference: bool = False
    memory_mode: str = "off"
    memory_protocol: MemoryProtocol = MemoryProtocol.OFF
    repair_mode: RepairMode = RepairMode.OFF
    fork_search_mode: ForkSearchMode = ForkSearchMode.OFF
    preliminary: bool = False

    @property
    def is_full_proicl(self) -> bool:
        return (
            self.uses_gepa_archive
            and self.uses_power_sampling
            and self.uses_repair
            and self.uses_fork_search
            and self.uses_memory
            and self.memory_protocol == MemoryProtocol.FROZEN_DEV
        )


PAPER_ALIGNED_SUSTAINED_TRACKS: tuple[str, ...] = (
    "reasoning_gym_family_relationships",
    "reasoning_gym_graph_color_n5",
    "reasoning_gym_graph_color_n8",
    "reasoning_gym_graph_color_n10",
    "reasoning_gym_graph_color_n13",
    "reasoning_gym_graph_color_n15",
    "reasoning_gym_graph_color_n18",
    "reasoning_gym_graph_color_n20",
    "reasoning_gym_boxnet",
)

HELDOUT_ARCHIVE_TRAIN_TRACKS: tuple[str, ...] = (
    "reasoning_gym_family_relationships",
    "reasoning_gym_graph_color_n10",
    "reasoning_gym_maze",
    "reasoning_gym_palindrome_generation",
    "reasoning_gym_letter_counting",
)

HELDOUT_EVAL_TRACKS: tuple[str, ...] = (
    "reasoning_gym_boxnet",
    "reasoning_gym_acre",
    "reasoning_gym_game_of_life_halting",
    "reasoning_gym_graph_color_n12",
)


def archive_scope_id(scope: ArchiveScope, tracks: tuple[str, ...]) -> str:
    digest = "-".join(track.replace("reasoning_gym_", "rg_") for track in tracks)
    return f"proicl_{scope.value}_{digest}"


def validate_archive_scope_membership(
    *,
    archive_scope: ArchiveScope,
    train_tracks: tuple[str, ...],
    heldout_tracks: tuple[str, ...],
) -> None:
    if not train_tracks:
        raise ValueError("archive train_tracks must be non-empty")
    if archive_scope == ArchiveScope.CROSS_FAMILY_CURRICULUM:
        leakage = sorted(set(train_tracks) & set(heldout_tracks))
        if leakage:
            raise ValueError(
                "cross-family curriculum archive leaks held-out target tracks: "
                + ", ".join(leakage)
            )
    if archive_scope == ArchiveScope.TRANSDUCTIVE_SUPPORT and (
        set(train_tracks) != set(heldout_tracks)
    ):
        raise ValueError(
            "transductive support archives must use the same train and heldout tracks"
        )
