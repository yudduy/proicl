"""HF/RWS authoritative scorer for Phase 3.1 ProRL trace logprobs."""

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-model-key", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _read_trajectories(path: Path):
    from polaris.prorl_recovery.logprob import TrajectoryForScoring

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        yield TrajectoryForScoring(
            row_id=str(row.get("row_id") or row.get("candidate_id")),
            task_family=str(row.get("task_family") or row.get("track")),
            problem_id=str(row["problem_id"]),
            prompt_text=str(row["prompt_text"]),
            response_text=str(row.get("response_text") or row.get("generation")),
        )


def main() -> None:
    from polaris.infra.serving.hf import RWSGenerator
    from polaris.prorl_recovery.logprob import score_trajectories_batched
    from polaris.registry import resolve_model

    args = _parse_args()
    model = resolve_model(args.base_model_key)
    sampler = RWSGenerator(
        model_id=model.hf_id,
        revision=model.revision,
        local_files_only=args.local_files_only,
    )
    rows = score_trajectories_batched(
        sampler=sampler,
        trajectories=_read_trajectories(args.input),
        batch_size=args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
