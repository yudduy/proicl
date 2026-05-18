"""Analyze POLARIS artifact directories without touching live backends."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/analysis/analysis.json"))
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _selected_rows(path: Path) -> list[dict]:
    selected = path / "selected.jsonl"
    if not selected.exists():
        return []
    return [
        json.loads(line)
        for line in selected.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze_artifact_dir(path: Path) -> dict:
    manifest = _read_json(path / "manifest.json")
    metrics = _read_json(path / "metrics.json")
    costs = _read_json(path / "costs.json")
    rollouts = _read_json(path / "rollouts.json") if (path / "rollouts.json").exists() else {}
    rows = _selected_rows(path)
    n = int(metrics.get("n_problems", len(rows)))
    k = sum(1 for row in rows if row.get("selected_passed"))
    if not rows:
        k = round(float(metrics.get("accuracy", 0.0)) * n)
    accuracy = k / n if n else 0.0
    return {
        "artifact_dir": str(path),
        "track": metrics.get("track") or manifest.get("config", {}).get("track"),
        "condition": metrics.get("condition") or manifest.get("condition"),
        "model_key": metrics.get("model_key") or manifest.get("config", {}).get("model_key"),
        "accuracy": accuracy,
        "ci95": _wilson_ci(k, n),
        "n": n,
        "correct": k,
        "estimated_dollar_cost": costs.get("estimated_dollar_cost", 0.0),
        "rollouts_total": rollouts.get("total", costs.get("rollout_total")),
        "false_admission_audit": "not_applicable_without_memory_events",
    }


def main() -> None:
    args = _parse_args()
    rows = [analyze_artifact_dir(path) for path in args.artifact_dirs]
    output = {
        "schema_version": 1,
        "runs": rows,
        "mcnemar": "requires paired baseline rows; not computed for unpaired inputs",
        "factorial_effects": "use polaris.stats.factorial on row-level pass/fail table",
        "break_even_n": "use polaris.stats.breakeven with rollout ledgers",
        "falsification_table": [
            {
                "claim": "POLARIS beats all production baselines",
                "status": "unresolved",
                "required_evidence": "paired clean final artifacts per track and baseline",
            }
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    falsification = args.out.with_name("falsification_table.md")
    falsification.write_text(
        "# Falsification Table\n\n"
        "| claim | status | required evidence |\n"
        "|---|---|---|\n"
        "| POLARIS beats all production baselines | unresolved | paired clean final artifacts per track and baseline |\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
