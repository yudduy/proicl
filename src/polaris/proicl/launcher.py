from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from polaris.prorl_recovery.protocol import BASE_MODEL_KEY, PRORL_V2_MODEL_KEY
from polaris.proicl.analysis import write_proicl_decomposition_by_track


REASONING_GYM_TRACKS: tuple[str, ...] = (
    "reasoning_gym_boxnet",
    "reasoning_gym_graph_color",
    "reasoning_gym_family_relationships",
)

COMMON_ARTIFACTS: tuple[str, ...] = (
    "metrics.json",
    "candidates.jsonl",
    "scores.jsonl",
    "selected.jsonl",
    "costs.json",
    "archive.json",
    "rollouts.json",
    "preflight.json",
    "environment.json",
    "manifest.json",
    "run_plan_cell.json",
    "audit.md",
)


@dataclass(frozen=True)
class LaunchCell:
    proicl_condition: str
    runtime_condition: str
    condition: str
    track: str
    model_key: str
    split: tuple[int, int]
    shard_id: int
    num_shards: int
    rollout_budget: int
    archive_path: str
    archive_build_id: str
    memory_build_id: str
    cache_path: str
    artifact_dir: str
    uses_gepa_archive: bool = False
    uses_memory: bool = False
    memory_store_path: str | None = None
    seed: int = 17
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def to_jsonable(self) -> dict[str, Any]:
        payload = {
            "proicl_condition": self.proicl_condition,
            "runtime_condition": self.runtime_condition,
            "condition": self.condition,
            "track": self.track,
            "model_key": self.model_key,
            "split": list(self.split),
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "rollout_budget": self.rollout_budget,
            "archive_path": self.archive_path,
            "archive_build_id": self.archive_build_id,
            "memory_build_id": self.memory_build_id,
            "cache_path": self.cache_path,
            "artifact_dir": self.artifact_dir,
            "uses_gepa_archive": self.uses_gepa_archive,
            "uses_memory": self.uses_memory,
            "memory_store_path": self.memory_store_path,
            "seed": self.seed,
            "extra_args": list(self.extra_args),
        }
        return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "ts": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}
    row.update(payload)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def source_hash(repo_root: Path) -> str:
    h = hashlib.sha256()
    roots = ["src", "scripts", "docs", "tests", "configs", "TODO.PROICL.md", "pyproject.toml"]
    for rel in roots:
        path = repo_root / rel
        if not path.exists():
            continue
        if path.is_file():
            files = [path]
        else:
            files = sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            h.update(str(file.relative_to(repo_root)).encode("utf-8"))
            h.update(b"\0")
            h.update(file.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def vendored_commit(repo_root: Path, rel: str) -> str:
    env_key = {
        "upstream/reasoning-with-sampling": "POLARIS_RWS_COMMIT",
        "upstream/gepa": "POLARIS_GEPA_COMMIT",
        "upstream/evalplus": "POLARIS_EVALPLUS_COMMIT",
        "upstream/dynamic-cheatsheet": "POLARIS_DC_COMMIT",
    }.get(rel)
    if env_key and os.environ.get(env_key):
        return str(os.environ[env_key])
    result = subprocess.run(
        ["git", "-C", str(repo_root / rel), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def build_signal_cells(
    *,
    root: Path,
    tracks: Iterable[str],
    split: tuple[int, int],
    rollout_budget: int,
    num_shards: int,
    memory_num_shards: int,
    seed: int = 17,
) -> list[LaunchCell]:
    if memory_num_shards != 1:
        raise ValueError(
            "ProICL memory cells must use memory_num_shards=1 so a track has "
            "one serialized verifier-gated curriculum store."
        )
    cells: list[LaunchCell] = []
    specs = (
        ("base_greedy", "greedy", BASE_MODEL_KEY, "direct", 1, False, False),
        ("mcmc_only", "single_prompt_power", BASE_MODEL_KEY, "direct", rollout_budget, False, False),
        ("gepa_only", "gepa_only", BASE_MODEL_KEY, "gepa", rollout_budget, True, False),
        ("gepa_mcmc", "proicl_gepa_mcmc", BASE_MODEL_KEY, "gepa", rollout_budget, True, False),
        (
            "gepa_mcmc_memory",
            "proicl_gepa_mcmc_memory",
            BASE_MODEL_KEY,
            "gepa",
            rollout_budget,
            True,
            True,
        ),
        ("prorl_v2_greedy", "greedy", PRORL_V2_MODEL_KEY, "direct", 1, False, False),
    )
    for track in tracks:
        direct_archive = root / "archives" / track / "direct.json"
        gepa_archive = root / "archives" / "proicl_cross_task_gepa" / "archive.json"
        for (
            proicl_condition,
            runtime_condition,
            model_key,
            archive_kind,
            budget,
            uses_gepa,
            uses_memory,
        ) in specs:
            shards = memory_num_shards if uses_memory else num_shards
            for shard_id in range(shards):
                archive_path = gepa_archive if archive_kind == "gepa" else direct_archive
                memory_path = (
                    root / "memory" / track / f"{proicl_condition}.sqlite"
                    if uses_memory
                    else None
                )
                extra: tuple[str, ...] = ()
                if uses_memory:
                    extra = ("--memory-mode", "distilled_strategies", "--admit-memory", "--online-memory")
                cells.append(
                    LaunchCell(
                        proicl_condition=proicl_condition,
                        runtime_condition=runtime_condition,
                        condition=runtime_condition,
                        track=track,
                        model_key=model_key,
                        split=split,
                        shard_id=shard_id,
                        num_shards=shards,
                        rollout_budget=budget,
                        archive_path=str(archive_path),
                        archive_build_id="proicl-cross-task-gepa" if uses_gepa else "none",
                        memory_build_id=f"proicl-memory-{track}" if uses_memory else "none",
                        cache_path=str(
                            root
                            / "trajectory_cache"
                            / f"{track}-{proicl_condition}-shard-{shard_id}.sqlite"
                        ),
                        artifact_dir=str(
                            root / "runs" / track / proicl_condition / f"shard-{shard_id}"
                        ),
                        uses_gepa_archive=uses_gepa,
                        uses_memory=uses_memory,
                        memory_store_path=str(memory_path) if memory_path else None,
                        seed=seed,
                        extra_args=extra,
                    )
                )
    return cells


def required_artifacts(cell: LaunchCell) -> tuple[str, ...]:
    extra: list[str] = []
    if cell.uses_gepa_archive or cell.uses_memory:
        extra.append("archive_build_manifest.json")
    if cell.uses_memory:
        extra.extend(["memory.sqlite", "memory_events.jsonl"])
    return COMMON_ARTIFACTS + tuple(extra)


def cell_complete(cell: LaunchCell) -> bool:
    out = Path(cell.artifact_dir)
    if not out.exists():
        return False
    for rel in required_artifacts(cell):
        path = out / rel
        if not path.exists() or path.stat().st_size == 0:
            return False
    try:
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(metrics.get("n_problems", 0)) > 0


def run_condition_command(
    *,
    repo_root: Path,
    cell: LaunchCell,
    backend: str,
    local_files_only: bool,
    polaris_source_hash: str,
    cost_cap_dollars: float,
    estimated_dollar_cost: float,
    estimated_wall_clock_seconds: float,
    run_kind: str,
    run_stage: str,
    max_new_tokens: int,
    mcmc_steps: int | None = None,
    mcmc_block_num: int | None = None,
    vllm_dtype: str = "float32",
    vllm_model_impl: str = "transformers",
    vllm_gpu_memory_utilization: float = 0.85,
    vllm_max_model_len: int | None = None,
    vllm_scoring_mode: str = "forced_decode_v0",
    vllm_parity_artifact: str | None = None,
) -> list[str]:
    if (
        cell.proicl_condition
        in {"mcmc_only", "gepa_only", "gepa_mcmc", "gepa_mcmc_memory"}
        and cell.model_key != BASE_MODEL_KEY
    ):
        raise ValueError(
            "ProICL frozen-base conditions must use the base model, not "
            f"{cell.model_key!r}"
        )
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_condition.py"),
        "--track",
        cell.track,
        "--model-key",
        cell.model_key,
        "--condition",
        cell.condition,
        "--archive",
        cell.archive_path,
        "--split",
        str(cell.split[0]),
        str(cell.split[1]),
        "--shard-id",
        str(cell.shard_id),
        "--num-shards",
        str(cell.num_shards),
        "--seed",
        str(cell.seed),
        "--backend",
        backend,
        "--samples-per-problem",
        str(cell.rollout_budget),
        "--sampling-temperature",
        "1.0",
        "--max-new-tokens",
        str(max_new_tokens),
        "--polaris-source-hash",
        polaris_source_hash,
        "--preregistration-anchor",
        "TODO.PROICL.md#proicl-fast-weight-recovery-audit",
        "--out",
        cell.artifact_dir,
        "--trajectory-cache",
        cell.cache_path,
        "--archive-build-id",
        cell.archive_build_id,
        "--memory-build-id",
        cell.memory_build_id,
        "--run-stage",
        run_stage,
        "--run-kind",
        run_kind,
        "--estimated-dollar-cost",
        str(estimated_dollar_cost),
        "--estimated-wall-clock-seconds",
        str(estimated_wall_clock_seconds),
        "--cost-cap-dollars",
        str(cost_cap_dollars),
        "--user-authorized-paid-run",
        "--rws-commit",
        vendored_commit(repo_root, "upstream/reasoning-with-sampling"),
        "--gepa-commit",
        vendored_commit(repo_root, "upstream/gepa"),
        "--evalplus-commit",
        vendored_commit(repo_root, "upstream/evalplus"),
        "--dc-commit",
        vendored_commit(repo_root, "upstream/dynamic-cheatsheet"),
    ]
    if mcmc_steps is not None:
        cmd.extend(["--mcmc-steps", str(mcmc_steps)])
    if mcmc_block_num is not None:
        cmd.extend(["--mcmc-block-num", str(mcmc_block_num)])
    if local_files_only:
        cmd.append("--local-files-only")
    if backend == "vllm":
        cmd.extend(
            [
                "--vllm-dtype",
                vllm_dtype,
                "--vllm-model-impl",
                vllm_model_impl,
                "--vllm-gpu-memory-utilization",
                str(vllm_gpu_memory_utilization),
                "--vllm-scoring-mode",
                vllm_scoring_mode,
            ]
        )
        if vllm_max_model_len is not None:
            cmd.extend(["--vllm-max-model-len", str(vllm_max_model_len)])
        if vllm_parity_artifact is not None:
            cmd.extend(["--vllm-parity-artifact", vllm_parity_artifact])
    if cell.memory_store_path is not None:
        cmd.extend(["--memory-store", cell.memory_store_path])
    cmd.extend(cell.extra_args)
    return cmd


def run_cells(
    *,
    repo_root: Path,
    cells: list[LaunchCell],
    gpus: list[str],
    events_path: Path,
    backend: str,
    local_files_only: bool,
    cost_cap_dollars: float,
    estimated_dollar_cost: float,
    estimated_wall_clock_seconds: float,
    run_kind: str,
    run_stage: str,
    max_new_tokens: int,
    mcmc_steps: int | None = None,
    mcmc_block_num: int | None = None,
    vllm_dtype: str = "float32",
    vllm_model_impl: str = "transformers",
    vllm_gpu_memory_utilization: float = 0.85,
    vllm_max_model_len: int | None = None,
    vllm_scoring_mode: str = "forced_decode_v0",
    vllm_parity_artifact: str | None = None,
    stop_on_failure: bool = True,
) -> None:
    if not gpus:
        raise ValueError("at least one GPU slot is required")
    polaris_hash = source_hash(repo_root)
    queue = [cell for cell in cells if not cell_complete(cell)]
    skipped = len(cells) - len(queue)
    append_event(events_path, "queue_start", cells=len(cells), pending=len(queue), skipped=skipped)
    active: list[tuple[subprocess.Popen, LaunchCell, str]] = []
    available_gpus = list(gpus)
    failures: list[dict[str, Any]] = []
    while queue or active:
        while queue and available_gpus:
            cell = queue.pop(0)
            gpu = available_gpus.pop(0)
            out = Path(cell.artifact_dir)
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "proicl_launch_cell.json", cell.to_jsonable())
            cmd = run_condition_command(
                repo_root=repo_root,
                cell=cell,
                backend=backend,
                local_files_only=local_files_only,
                polaris_source_hash=polaris_hash,
                cost_cap_dollars=cost_cap_dollars,
                estimated_dollar_cost=estimated_dollar_cost,
                estimated_wall_clock_seconds=estimated_wall_clock_seconds,
                run_kind=run_kind,
                run_stage=run_stage,
                max_new_tokens=max_new_tokens,
                mcmc_steps=mcmc_steps,
                mcmc_block_num=mcmc_block_num,
                vllm_dtype=vllm_dtype,
                vllm_model_impl=vllm_model_impl,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                vllm_max_model_len=vllm_max_model_len,
                vllm_scoring_mode=vllm_scoring_mode,
                vllm_parity_artifact=vllm_parity_artifact,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            append_event(
                events_path,
                "cell_start",
                gpu=gpu,
                track=cell.track,
                condition=cell.proicl_condition,
                shard=cell.shard_id,
            )
            stdout = (out / "stdout.json").open("w", encoding="utf-8")
            stderr = (out / "stderr.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=stdout, stderr=stderr)
            stdout.close()
            stderr.close()
            active.append((proc, cell, gpu))
        time.sleep(2)
        still_active: list[tuple[subprocess.Popen, LaunchCell, str]] = []
        for proc, cell, gpu in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, cell, gpu))
                continue
            if rc == 0 and cell_complete(cell):
                available_gpus.append(gpu)
                append_event(
                    events_path,
                    "cell_done",
                    gpu=gpu,
                    track=cell.track,
                    condition=cell.proicl_condition,
                    shard=cell.shard_id,
                )
            else:
                available_gpus.append(gpu)
                failure = {
                    "track": cell.track,
                    "condition": cell.proicl_condition,
                    "shard": cell.shard_id,
                    "returncode": rc,
                    "artifact_dir": cell.artifact_dir,
                }
                failures.append(failure)
                append_event(events_path, "cell_failed", **failure)
                if stop_on_failure:
                    for live, _, _ in still_active:
                        live.terminate()
                    raise RuntimeError(f"ProICL cell failed: {failure}")
        active = still_active
    append_event(events_path, "queue_done", failures=len(failures))
    if failures:
        raise RuntimeError(f"ProICL queue had failures: {failures}")


def aggregate(root: Path, tracks: Iterable[str], out_dir: Path) -> dict[str, Any]:
    return write_proicl_decomposition_by_track(
        root=root,
        tracks=tuple(tracks),
        out_dir=out_dir,
    )
