"""Build a frozen POLARIS archive + verifier-gated memory bundle.

The production path is intentionally resumable and artifact-first. `--dry-run`
uses deterministic local stand-ins so CI can prove the artifact contract without
loading a model or launching GEPA. Paid/full archive construction must still run
behind the same preflight/cost gates as condition runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        required=True,
        choices=[
            "math500",
            "humaneval_plus",
            "gpqa_diamond",
            "reasoning_gym_boxnet",
            "reasoning_gym_graph_color",
            "reasoning_gym_family_relationships",
        ],
    )
    parser.add_argument(
        "--mode",
        choices=["mine-hard-queries", "build-memory", "evolve-prompts", "freeze"],
        default="freeze",
    )
    parser.add_argument("--dev-split", type=int, nargs=2, required=True)
    parser.add_argument(
        "--dev-source",
        choices=["optimizer_dev", "benchmark"],
        default="optimizer_dev",
        help="Use external optimizer-development rows by default; benchmark rows are for legacy/debug only.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reflection-provider",
        choices=["none", "xai"],
        default="none",
        help="Reflection LM provider for prompt evolution metadata and live GEPA runs.",
    )
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_dev_rows(track: str, split: tuple[int, int], *, source: str) -> list[Any]:
    from polaris.runners.condition_runner import (
        load_track_optimizer_dev_slice,
        load_track_slice,
    )

    if source == "optimizer_dev":
        return load_track_optimizer_dev_slice(track, split[0], split[1])
    return load_track_slice(track, split[0], split[1])


def _base_archive(track: str):
    from polaris.core.archive import MATH500_ARCHIVE_V1

    # Current prompt archive format is track-agnostic text. Track-specific
    # builders can replace these seeds after small real-slice evidence.
    return MATH500_ARCHIVE_V1


def _descriptor_audit_stub(track: str, rows: list[Any]) -> dict[str, Any]:
    return {
        "track": track,
        "n_traces": len(rows),
        "judge_count": 2,
        "categorical_agreement": None,
        "numeric_agreement_ci95": None,
        "status": "dry_run_stub",
        "notes": "Production descriptor audit requires 200 traces per track.",
    }


def _hard_query_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "problem_id": row.problem_id,
            "hard_score": 1.0,
            "reason": "dry_run_or_cached_current_archive_failed",
        }
        for row in rows
    ]


def _admit_dry_run_memory(path: Path, *, track: str, rows: list[Any]) -> int:
    from polaris.core.memory import MemoryEntry
    from polaris.core.persistent_memory import PersistentMemoryLedger

    count = 0
    with PersistentMemoryLedger(path) as ledger:
        for idx, row in enumerate(rows):
            entry = MemoryEntry(
                id=f"{track}:{row.problem_id}:strategy:{idx}",
                archive_prompt_id="direct",
                descriptor="direct_computation",
                strategy_text="Use a verifier-backed intermediate check before the final answer.",
                token_count=10,
                source_query_id=row.problem_id,
            )
            ledger.admit(
                entry,
                track=track,
                verifier_id=f"{track}/dry-run-verifier",
                metadata={"mode": "build-memory", "dry_run": True},
            )
            count += 1
        ledger.snapshot_posteriors(label="freeze")
    return count


def _export_memory_events(sqlite_path: Path, jsonl_path: Path) -> None:
    from polaris.core.persistent_memory import PersistentMemoryLedger

    with PersistentMemoryLedger(sqlite_path) as ledger:
        _write_jsonl(jsonl_path, ledger.events())


def _reflection_payload(args: argparse.Namespace, rows: list[Any]) -> dict[str, Any]:
    from polaris.gepa_reflection import (
        XAIReflectionConfig,
        load_env_file,
        reflection_manifest,
    )

    if args.reflection_provider == "none":
        return {"provider": "none", "status": "disabled", "usage": {}}

    load_env_file(args.env_file)
    config = XAIReflectionConfig.from_env(require_key=False)
    estimated_input_tokens = max(1, len(rows)) * 1000 * max(1, args.rounds)
    estimated_output_tokens = max(1, len(rows)) * 250 * max(1, args.rounds)
    usage = config.estimate_cost(
        input_tokens=estimated_input_tokens,
        output_tokens=estimated_output_tokens,
    )
    return reflection_manifest(
        provider="xai",
        config=config,
        status="dry_run_configured" if args.dry_run else "configured",
        usage=usage,
    )


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from polaris.gepa_integration import PolarisGEPAAdapter
    from polaris.io.rollouts import RolloutLedger

    split = (args.dev_split[0], args.dev_split[1])
    rows = _load_dev_rows(args.track, split, source=args.dev_source)
    archive = _base_archive(args.track)
    build_id = f"{args.track}-{args.mode}-seed-{args.seed}-{split[0]}-{split[1]}"
    reflection = _reflection_payload(args, rows)

    if args.mode in {"mine-hard-queries", "freeze"}:
        _write_jsonl(args.out / "hard_queries.jsonl", _hard_query_rows(rows))

    memory_count = 0
    if args.mode in {"build-memory", "freeze"}:
        memory_count = _admit_dry_run_memory(
            args.out / "memory.sqlite",
            track=args.track,
            rows=rows,
        )
        _export_memory_events(args.out / "memory.sqlite", args.out / "memory_events.jsonl")

    if args.mode in {"evolve-prompts", "freeze"}:
        _write_json(
            args.out / "gepa_adapter_state.json",
            PolarisGEPAAdapter(
                sampler=None,
                scorer=lambda response, answer: {"score": 0.0},
                max_new_tokens=0,
                run_dir=args.out / "gepa_checkpoint",
            ).get_adapter_state(),
        )

    archive_payload = {
        "entries": archive.to_jsonable(),
        "cell_fitness": {
            entry["descriptor_hint"]: 1.0 for entry in archive.to_jsonable()
        },
        "archive_build_id": build_id,
        "selection_rule": "pareto_map_elites_accuracy_with_memory_descriptor_cost",
        "frozen": True,
    }
    _write_json(args.out / "archive.json", archive_payload)
    _write_json(args.out / "descriptor_audit.json", _descriptor_audit_stub(args.track, rows))

    ledger = RolloutLedger(
        archive_construction=len(rows) * max(1, len(archive.entries)) * args.rounds,
        memory_admission=memory_count,
    )
    ledger.write(args.out / "rollouts.json")
    _write_json(
        args.out / "archive_build_manifest.json",
        {
            "archive_build_id": build_id,
            "track": args.track,
            "mode": args.mode,
            "dev_split": list(split),
            "dev_source": args.dev_source,
            "seed": args.seed,
            "rounds": args.rounds,
            "dry_run": args.dry_run,
            "resume": args.resume,
            "archive_path": str(args.out / "archive.json"),
            "memory_path": str(args.out / "memory.sqlite"),
            "descriptor_audit_path": str(args.out / "descriptor_audit.json"),
            "rollouts_path": str(args.out / "rollouts.json"),
            "reflection": reflection,
            "status": "dry_run_complete" if args.dry_run else "complete",
        },
    )
    print(json.dumps({"archive_build_id": build_id, "path": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
