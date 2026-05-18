"""Materialize exact successful ProRL/BroRL trajectories for Phase 3.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-input", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate_rows(root: Path):
    for path in sorted(root.glob("**/candidates.jsonl")):
        manifest_path = path.parent / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        config = manifest.get("config", {})
        for row in _read_jsonl(path):
            row.setdefault("checkpoint", config.get("model_key") or manifest.get("model_id"))
            row.setdefault("task_family", config.get("track") or manifest.get("benchmark"))
            yield row


def main() -> None:
    from polaris.prorl_recovery.phase3 import materialize_phase3_trajectories

    args = _parse_args()
    if not args.phase3_input.exists():
        raise SystemExit(f"missing phase3 input: {args.phase3_input}")
    if not args.phase1_root.exists():
        raise SystemExit(f"missing phase1 root: {args.phase1_root}")
    rows = materialize_phase3_trajectories(
        phase3_rows=_read_jsonl(args.phase3_input),
        candidate_rows=_candidate_rows(args.phase1_root),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
