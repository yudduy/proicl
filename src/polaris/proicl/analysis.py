from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_REQUIRED_KEYS = (
    "base",
    "mcmc_only",
    "gepa_only",
    "gepa_mcmc",
    "gepa_mcmc_memory",
    "prorl_v2_greedy",
)

_DIR_TO_ACCURACY_KEY = {
    "base_greedy": "base",
    "mcmc_only": "mcmc_only",
    "gepa_only": "gepa_only",
    "gepa_mcmc": "gepa_mcmc",
    "gepa_mcmc_memory": "gepa_mcmc_memory",
    "prorl_v2_greedy": "prorl_v2_greedy",
}


def _accuracy(source: Mapping[str, float], key: str) -> float:
    try:
        return float(source[key])
    except KeyError as exc:
        raise ValueError(f"missing ProICL accuracy: {key}") from exc


def _rf(*, base: float, recovered: float, trained: float) -> float | None:
    denom = trained - base
    if denom <= 0.0:
        return None
    return max(0.0, min(1.0, (recovered - base) / denom))


def compute_proicl_decomposition(accuracies: Mapping[str, float]) -> dict[str, Any]:
    """Compute the ProICL fast-weight decomposition from condition accuracies."""

    missing = [key for key in _REQUIRED_KEYS if key not in accuracies]
    if missing:
        raise ValueError("missing ProICL accuracies: " + ", ".join(missing))

    base = _accuracy(accuracies, "base")
    mcmc = _accuracy(accuracies, "mcmc_only")
    gepa = _accuracy(accuracies, "gepa_only")
    gepa_mcmc = _accuracy(accuracies, "gepa_mcmc")
    memory = _accuracy(accuracies, "gepa_mcmc_memory")
    prorl = _accuracy(accuracies, "prorl_v2_greedy")

    rf_denominator = prorl - base
    rf_valid = rf_denominator > 0.0
    report = {
        "A_base": base,
        "A_mcmc": mcmc,
        "A_gepa": gepa,
        "A_gepa_mcmc": gepa_mcmc,
        "A_memory": memory,
        "A_prorl_v2": prorl,
        "rf_denominator": rf_denominator,
        "rf_valid": rf_valid,
        "RF_mcmc": _rf(base=base, recovered=mcmc, trained=prorl),
        "RF_gepa": _rf(base=base, recovered=gepa, trained=prorl),
        "RF_gepa_mcmc": _rf(base=base, recovered=gepa_mcmc, trained=prorl),
        "RF_memory": _rf(base=base, recovered=memory, trained=prorl),
        "slow_weight_residual": prorl - gepa_mcmc,
        "memory_gain": memory - gepa_mcmc,
        "discovery_gain": gepa - mcmc,
        "composition_gain": gepa_mcmc - max(mcmc, gepa),
    }
    if not rf_valid:
        report["analysis_warning"] = (
            "prorl_v2_greedy accuracy did not exceed base_greedy on this slice; "
            "RF is undefined for this track/smoke slice."
        )
    return report


def _fmt(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):.6f}"


def write_proicl_decomposition(
    *,
    accuracies: Mapping[str, float],
    out_dir: Path,
) -> dict[str, Any]:
    report = compute_proicl_decomposition(accuracies)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proicl_decomposition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ProICL decomposition",
        "",
        f"- A_base: {report['A_base']:.6f}",
        f"- A_mcmc: {report['A_mcmc']:.6f}",
        f"- A_gepa: {report['A_gepa']:.6f}",
        f"- A_gepa_mcmc: {report['A_gepa_mcmc']:.6f}",
        f"- A_memory: {report['A_memory']:.6f}",
        f"- A_prorl_v2: {report['A_prorl_v2']:.6f}",
        "",
        f"- rf_denominator: {report['rf_denominator']:.6f}",
        f"- rf_valid: {report['rf_valid']}",
        f"- RF_mcmc: {_fmt(report['RF_mcmc'])}",
        f"- RF_gepa: {_fmt(report['RF_gepa'])}",
        f"- RF_gepa_mcmc: {_fmt(report['RF_gepa_mcmc'])}",
        f"- RF_memory: {_fmt(report['RF_memory'])}",
        "",
        f"- slow_weight_residual: {report['slow_weight_residual']:.6f}",
        f"- memory_gain: {report['memory_gain']:.6f}",
        f"- discovery_gain: {report['discovery_gain']:.6f}",
        f"- composition_gain: {report['composition_gain']:.6f}",
    ]
    if report.get("analysis_warning"):
        lines.extend(["", f"- warning: {report['analysis_warning']}"])
    (out_dir / "proicl_decomposition.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def collect_track_accuracies(*, root: Path, track: str) -> dict[str, float]:
    """Collect weighted condition accuracies from ProICL shard artifacts."""

    track_dir = root / "runs" / track
    totals: dict[str, tuple[float, int]] = {}
    for dirname, key in _DIR_TO_ACCURACY_KEY.items():
        condition_dir = track_dir / dirname
        if not condition_dir.exists():
            continue
        weighted = 0.0
        n_total = 0
        for metrics_path in sorted(condition_dir.glob("shard-*/metrics.json")):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            n = int(metrics.get("n_problems", 0))
            weighted += float(metrics.get("accuracy", 0.0)) * n
            n_total += n
        if n_total > 0:
            totals[key] = (weighted, n_total)
    return {key: weighted / n for key, (weighted, n) in totals.items()}


def write_proicl_decomposition_by_track(
    *,
    root: Path,
    tracks: list[str] | tuple[str, ...],
    out_dir: Path,
) -> dict[str, Any]:
    """Write per-track ProICL decomposition from canonical run artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for track in tracks:
        accuracies = collect_track_accuracies(root=root, track=track)
        report = compute_proicl_decomposition(accuracies)
        reports[track] = {"accuracies": accuracies, "decomposition": report}

    (out_dir / "proicl_decomposition.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# ProICL decomposition", ""]
    for track, payload in reports.items():
        report = payload["decomposition"]
        lines.extend(
            [
                f"## {track}",
                "",
                f"- A_base: {report['A_base']:.6f}",
                f"- A_mcmc: {report['A_mcmc']:.6f}",
                f"- A_gepa: {report['A_gepa']:.6f}",
                f"- A_gepa_mcmc: {report['A_gepa_mcmc']:.6f}",
                f"- A_memory: {report['A_memory']:.6f}",
                f"- A_prorl_v2: {report['A_prorl_v2']:.6f}",
                f"- rf_denominator: {report['rf_denominator']:.6f}",
                f"- rf_valid: {report['rf_valid']}",
                f"- RF_mcmc: {_fmt(report['RF_mcmc'])}",
                f"- RF_gepa: {_fmt(report['RF_gepa'])}",
                f"- RF_gepa_mcmc: {_fmt(report['RF_gepa_mcmc'])}",
                f"- RF_memory: {_fmt(report['RF_memory'])}",
                f"- slow_weight_residual: {report['slow_weight_residual']:.6f}",
                f"- memory_gain: {report['memory_gain']:.6f}",
                f"- discovery_gain: {report['discovery_gain']:.6f}",
                f"- composition_gain: {report['composition_gain']:.6f}",
                *(
                    [f"- warning: {report['analysis_warning']}"]
                    if report.get("analysis_warning")
                    else []
                ),
                "",
            ]
        )
    (out_dir / "proicl_decomposition.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return reports
