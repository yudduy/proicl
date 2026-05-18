from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable


def _entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def trace_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def diversity_diagnostics(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(candidates)
    answers = [str(row.get("extracted_answer", "")) for row in rows]
    descriptors = [str(row.get("descriptor", "")) for row in rows]
    traces = [str(row.get("generation", row.get("response", ""))) for row in rows]
    lengths = [
        int(row.get("generation_token_count", row.get("response_length", len(trace))))
        for row, trace in zip(rows, traces)
    ]
    return {
        "n_candidates": len(rows),
        "answer_entropy": _entropy(answers),
        "unique_answer_count": len(set(answers)),
        "descriptor_entropy": _entropy(descriptors),
        "trace_hash_count": len({trace_hash(trace) for trace in traces}),
        "response_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        },
    }
