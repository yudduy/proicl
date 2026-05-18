"""Derive the locked Phase 3 ProRL/BroRL-only input set."""

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
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--rung7", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload if isinstance(payload, list) else payload["rows"])
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported input suffix: {path.suffix}")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if path.suffix == ".parquet":
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    raise ValueError(f"unsupported output suffix: {path.suffix}")


def main() -> None:
    from polaris.prorl_recovery.phase3 import (
        derive_phase3_input_set,
        require_phase3_inputs,
    )

    args = _parse_args()
    try:
        require_phase3_inputs(args.phase1, args.rung7)
        rows = derive_phase3_input_set(
            phase1_rows=_read_rows(args.phase1),
            rung7_rows=_read_rows(args.rung7),
        )
        _write_rows(args.out, rows)
    except Exception as exc:
        raise SystemExit(f"phase3 input derivation failed: {exc}") from exc
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
