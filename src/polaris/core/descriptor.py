from __future__ import annotations

import re
from dataclasses import dataclass

# Versioned per PROPOSAL §9: the descriptor extractor is a measurement
# instrument and must be pinned in the manifest.
#
# v1 (4 labels) is the frozen extractor used by MATH500_ARCHIVE_V1 and
# the v1 MAP-Elites grid. It remains stable for replication.
# v2 (8 labels + 4 numeric features) is the PROPOSAL §9 full specification
# and gates the v2 descriptor inter-judge reliability audit.
DESCRIPTOR_EXTRACTOR_VERSION = "v1-heuristic-2026-05-12"
DESCRIPTOR_EXTRACTOR_VERSION_V2 = "v2-heuristic-2026-05-12"

DESCRIPTOR_CATEGORIES: tuple[str, ...] = (
    "direct_computation",
    "algebraic_transformation",
    "backward_verification",
    "stepwise_decomposition",
)

# PROPOSAL §9 pattern labels. Superset of v1; v1 labels remain valid as a
# subset, with `stepwise_decomposition` mapping cleanly to v2's framing.
DESCRIPTOR_CATEGORIES_V2: tuple[str, ...] = (
    "direct_computation",
    "algebraic_transformation",
    "case_analysis",
    "contradiction",
    "backward_verification",
    "induction",
    "search_enumeration",
    "mixed_other",
)

_STEPWISE_PAT = re.compile(r"\bstep\s*\d+\b", re.IGNORECASE)
_ALGEBRAIC_PAT = re.compile(r"\blet\s+[a-z]\s*=", re.IGNORECASE)
_VERIFY_PAT = re.compile(
    r"\b(verify|substitut\w+\s+back|check\s+(the\s+)?answer)\b", re.IGNORECASE
)
_CASE_PAT = re.compile(r"\bcase\s+\d+\b|\bif\s+.*?,\s+then\b", re.IGNORECASE)
_CONTRADICTION_PAT = re.compile(
    r"\b(suppose|assume)\b.*?\b(contradict|impossible|cannot\s+be)\b",
    re.IGNORECASE | re.DOTALL,
)
_INDUCTION_PAT = re.compile(
    r"\b(induction|base\s+case|inductive\s+(step|hypothesis))\b", re.IGNORECASE
)
_SEARCH_PAT = re.compile(
    r"\b(try\s+\w+\s*=|enumerate|exhaustive|brute[- ]?force)\b", re.IGNORECASE
)
_SYMBOL_PAT = re.compile(r"[A-Za-z]\w*")
_BRANCH_HINT_PAT = re.compile(
    r"\b(case|alternative|otherwise|else if)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class TraceFeatures:
    """v2 descriptor output (PROPOSAL §9): pattern label + 4 numeric features.

    `pattern_label` is one of `DESCRIPTOR_CATEGORIES_V2`. Numeric features are
    estimated by the heuristic; LLM-judge or hybrid extractors will replace
    them under the pre-registered descriptor-audit experiment.
    """

    pattern_label: str
    step_count: int
    branch_count: int
    verification_density: float
    symbol_diversity: float


def classify_trace(trace: str) -> tuple[str, float]:
    """v1 extractor — returns (DESCRIPTOR_CATEGORIES label, confidence).

    Frozen for `MATH500_ARCHIVE_V1` and v1 replication runs. Do not modify
    without bumping `DESCRIPTOR_EXTRACTOR_VERSION` and amending TODO.md.
    """
    if not trace:
        return "direct_computation", 0.0

    n_step = len(_STEPWISE_PAT.findall(trace))
    n_alg = len(_ALGEBRAIC_PAT.findall(trace))
    n_ver = len(_VERIFY_PAT.findall(trace))

    counts = {
        "stepwise_decomposition": n_step,
        "algebraic_transformation": n_alg,
        "backward_verification": n_ver,
    }
    best_label = max(counts, key=lambda k: counts[k])
    best_count = counts[best_label]
    if best_count == 0:
        return "direct_computation", 0.0
    confidence = min(1.0, best_count / 3.0)
    if confidence < 0.25:
        return "direct_computation", confidence
    return best_label, confidence


def extract_features(trace: str) -> TraceFeatures:
    """v2 extractor — 8 labels + step/branch/verification/symbol features.

    Heuristic implementation; the LLM-judge and hybrid variants land with
    the descriptor inter-judge audit (PROPOSAL §9, locked in TODO.md as
    `polaris-math500-v2-sglang` phase).
    """
    if not trace.strip():
        return TraceFeatures(
            pattern_label="direct_computation",
            step_count=0,
            branch_count=0,
            verification_density=0.0,
            symbol_diversity=0.0,
        )

    n_step = len(_STEPWISE_PAT.findall(trace))
    n_alg = len(_ALGEBRAIC_PAT.findall(trace))
    n_ver = len(_VERIFY_PAT.findall(trace))
    n_case = len(_CASE_PAT.findall(trace))
    n_contra = len(_CONTRADICTION_PAT.findall(trace))
    n_ind = len(_INDUCTION_PAT.findall(trace))
    n_search = len(_SEARCH_PAT.findall(trace))

    candidate_counts = {
        "stepwise_decomposition": n_step,  # v1-compat tag; not in v2 list but observable
        "algebraic_transformation": n_alg,
        "backward_verification": n_ver,
        "case_analysis": n_case,
        "contradiction": n_contra,
        "induction": n_ind,
        "search_enumeration": n_search,
    }
    best_label = max(candidate_counts, key=lambda k: candidate_counts[k])
    if candidate_counts[best_label] == 0:
        label = "direct_computation"
    elif best_label == "stepwise_decomposition":
        # Map v1's stepwise tag onto v2's mixed_other since v2 collapses
        # generic step-numbered traces into mixed/other (PROPOSAL §9 label list).
        label = "mixed_other"
    else:
        label = best_label

    steps = max(n_step, len(re.findall(r"\n\s*[-*]\s|\.\s+[A-Z]", trace)))
    branches = n_case + len(_BRANCH_HINT_PAT.findall(trace))
    verification_density = n_ver / max(steps, 1) if steps else (1.0 if n_ver else 0.0)
    symbols = _SYMBOL_PAT.findall(trace)
    symbol_diversity = (len(set(symbols)) / len(symbols)) if symbols else 0.0

    return TraceFeatures(
        pattern_label=label,
        step_count=steps,
        branch_count=branches,
        verification_density=float(min(1.0, verification_density)),
        symbol_diversity=float(symbol_diversity),
    )
