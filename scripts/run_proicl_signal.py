"""Run the ProICL 20-problem/B=8 overnight signal experiment.

This is the executable ProICL harness: it builds prompt archives, runs the
six operational conditions, skips completed cells on resume, and aggregates
the decomposition. It defaults to the HF vendored-RWS backend for MCMC
faithfulness; vLLM is available only as an explicitly selected calibrated path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/proicl_overnight_signal"))
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=[
            "reasoning_gym_boxnet",
            "reasoning_gym_graph_color",
            "reasoning_gym_family_relationships",
        ],
    )
    parser.add_argument("--eval-split", type=int, nargs=2, default=(20, 40))
    parser.add_argument("--gepa-dev-split", type=int, nargs=2, default=(0, 6))
    parser.add_argument("--rollout-budget", type=int, default=8)
    parser.add_argument("--archive-size", type=int, default=8)
    parser.add_argument("--max-metric-calls", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--mcmc-steps", type=int, default=None)
    parser.add_argument("--mcmc-block-num", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--memory-num-shards", type=int, default=1)
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--vllm-dtype", default="float32")
    parser.add_argument("--vllm-model-impl", default="transformers")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument(
        "--vllm-scoring-mode",
        choices=["forced_decode_v0", "native_segment"],
        default="forced_decode_v0",
    )
    parser.add_argument("--vllm-parity-artifact", type=Path, default=None)
    parser.add_argument("--run-kind", default="flow")
    parser.add_argument("--run-stage", default="small_real_slice")
    parser.add_argument("--cost-cap-dollars", type=float, default=4.0)
    parser.add_argument("--estimated-dollar-cost-per-cell", type=float, default=0.25)
    parser.add_argument("--estimated-wall-clock-seconds-per-cell", type=float, default=7200)
    parser.add_argument("--xai-reflection-cap-dollars", type=float, default=2.0)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--force-gepa", action="store_true")
    parser.add_argument(
        "--cell-stride",
        type=int,
        default=1,
        help="Run only every Nth full-run cell. Used to split one ProICL run across Flow hosts.",
    )
    parser.add_argument(
        "--cell-offset",
        type=int,
        default=0,
        help="Offset for --cell-stride partitioning.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Do not aggregate after this worker finishes. Use for partition workers.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, env: dict[str, str] | None = None, stdout: Path | None = None, stderr: Path | None = None) -> None:
    stdout_f = stdout.open("w", encoding="utf-8") if stdout else None
    stderr_f = stderr.open("w", encoding="utf-8") if stderr else None
    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            check=True,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    finally:
        if stdout_f:
            stdout_f.close()
        if stderr_f:
            stderr_f.close()


def _setup_env(args: argparse.Namespace) -> dict[str, str]:
    from polaris.gepa_reflection import load_env_file
    from polaris.proicl.launcher import vendored_commit

    load_env_file(args.env_file)
    env = os.environ.copy()
    configure_flow_cache_env(env, run_kind=args.run_kind)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["XAI_REFLECTION_INITIAL_CAP_DOLLARS"] = str(args.xai_reflection_cap_dollars)
    env["XAI_REFLECTION_HARD_CAP_DOLLARS"] = str(args.xai_reflection_cap_dollars)
    env.setdefault("POLARIS_RWS_COMMIT", vendored_commit(REPO_ROOT, "upstream/reasoning-with-sampling"))
    env.setdefault("POLARIS_GEPA_COMMIT", vendored_commit(REPO_ROOT, "upstream/gepa"))
    env.setdefault("POLARIS_EVALPLUS_COMMIT", vendored_commit(REPO_ROOT, "upstream/evalplus"))
    env.setdefault("POLARIS_DC_COMMIT", vendored_commit(REPO_ROOT, "upstream/dynamic-cheatsheet"))
    return env


def configure_flow_cache_env(
    env: dict[str, str],
    *,
    run_kind: str,
    mnt_local: Path = Path("/mnt/local"),
    cache_root: Path = Path("/mnt/local/proicl-cache"),
) -> None:
    """Keep Flow runs away from root-owned /workspace cache directories."""
    if run_kind != "flow":
        return
    if not mnt_local.exists():
        return
    paths = {
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "PIP_CACHE_DIR": cache_root / "pip",
        "TORCH_HOME": cache_root / "torch",
        "CUDA_CACHE_PATH": cache_root / "cuda",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)


def _visible_gpus(args: argparse.Namespace) -> list[str]:
    if args.gpus:
        return [str(g) for g in args.gpus]
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            capture_output=True,
            check=True,
        )
        count = sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception:
        pass
    return ["0"]


def _prefetch_models(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.skip_prefetch:
        return
    from huggingface_hub import snapshot_download
    from polaris.config import MODEL_REGISTRY

    out = args.root / "prefetch"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in ("deepseek-r1-distill-qwen-1.5b", "nemotron-prorl-v2"):
        spec = MODEL_REGISTRY[key]
        path = snapshot_download(
            repo_id=spec["hf_id"],
            revision=spec.get("revision"),
            local_files_only=False,
        )
        rows.append({"model_key": key, "hf_id": spec["hf_id"], "revision": spec.get("revision"), "path": path})
    (out / "models.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _archive_is_live(
    path: Path,
    *,
    archive_size: int,
    tracks: list[str],
    dev_split: tuple[int, int],
    max_metric_calls: int,
) -> bool:
    manifest = path / "archive_build_manifest.json"
    archive = path / "archive.json"
    if not manifest.exists() or not archive.exists():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("tracks") != list(tracks):
        return False
    if payload.get("dev_split") != list(dev_split):
        return False
    if int(payload.get("gepa", {}).get("max_metric_calls", -1)) != int(max_metric_calls):
        return False
    if payload.get("gepa", {}).get("dry_run", True):
        return False
    if payload.get("reflection", {}).get("provider") != "xai":
        return False
    archive_payload = json.loads(archive.read_text(encoding="utf-8"))
    return len(archive_payload.get("entries", [])) == archive_size


def _build_direct_archives(root: Path, tracks: list[str]) -> None:
    _run(
        [
            sys.executable,
            "scripts/run_proicl.py",
            "write-direct-archives",
            "--root",
            str(root),
            "--tracks",
            *tracks,
        ]
    )


def _build_gepa_archive(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    root: Path,
    tracks: list[str],
    dev_split: tuple[int, int],
    archive_size: int,
    max_metric_calls: int,
    sampler_max_new_tokens: int,
    reflection_max_new_tokens: int,
    cuda_visible_devices: str | None = None,
    force: bool = False,
) -> None:
    out = root / "archives" / "proicl_cross_task_gepa"
    if (
        _archive_is_live(
            out,
            archive_size=archive_size,
            tracks=tracks,
            dev_split=dev_split,
            max_metric_calls=max_metric_calls,
        )
        and not force
    ):
        return
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _run(
        _gepa_archive_command(
            args=args,
            out=out,
            tracks=tracks,
            dev_split=dev_split,
            archive_size=archive_size,
            max_metric_calls=max_metric_calls,
            sampler_max_new_tokens=sampler_max_new_tokens,
            reflection_max_new_tokens=reflection_max_new_tokens,
            cuda_visible_devices=cuda_visible_devices,
        ),
        env=env,
        stdout=log_dir / f"build_gepa_k{archive_size}.stdout.json",
        stderr=log_dir / f"build_gepa_k{archive_size}.stderr.log",
    )


def _gepa_archive_command(
    *,
    args: argparse.Namespace,
    out: Path,
    tracks: list[str],
    dev_split: tuple[int, int],
    archive_size: int,
    max_metric_calls: int,
    sampler_max_new_tokens: int,
    reflection_max_new_tokens: int,
    cuda_visible_devices: str | None,
) -> list[str]:
    cmd = [
            sys.executable,
            "scripts/run_proicl.py",
            "build-archive",
            "--out",
            str(out),
            "--tracks",
            *tracks,
            "--dev-split",
            str(dev_split[0]),
            str(dev_split[1]),
            "--archive-size",
            str(archive_size),
            "--max-metric-calls",
            str(max_metric_calls),
            "--reflection-provider",
            "xai",
            "--env-file",
            str(args.env_file),
            "--live-gepa",
            "--sampler-max-new-tokens",
            str(sampler_max_new_tokens),
            "--reflection-max-new-tokens",
            str(reflection_max_new_tokens),
        ]
    if args.backend == "vllm":
        cmd.extend(
            [
                "--sampler-backend",
                "vllm",
                "--vllm-dtype",
                args.vllm_dtype,
                "--vllm-model-impl",
                args.vllm_model_impl,
                "--vllm-gpu-memory-utilization",
                str(args.vllm_gpu_memory_utilization),
                "--vllm-scoring-mode",
                args.vllm_scoring_mode,
            ]
        )
        if args.vllm_max_model_len is not None:
            cmd.extend(["--vllm-max-model-len", str(args.vllm_max_model_len)])
        if args.vllm_parity_artifact is not None:
            cmd.extend(["--vllm-parity-artifact", str(args.vllm_parity_artifact)])
    if cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", cuda_visible_devices])
    if args.local_files_only:
        cmd.append("--local-files-only")
    return cmd


def _start_gepa_archive(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    root: Path,
    tracks: list[str],
    dev_split: tuple[int, int],
    archive_size: int,
    max_metric_calls: int,
    sampler_max_new_tokens: int,
    reflection_max_new_tokens: int,
    cuda_visible_devices: str,
    force: bool = False,
) -> subprocess.Popen | None:
    out = root / "archives" / "proicl_cross_task_gepa"
    if (
        _archive_is_live(
            out,
            archive_size=archive_size,
            tracks=tracks,
            dev_split=dev_split,
            max_metric_calls=max_metric_calls,
        )
        and not force
    ):
        return None
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / f"build_gepa_k{archive_size}.stdout.json").open("w", encoding="utf-8")
    stderr = (log_dir / f"build_gepa_k{archive_size}.stderr.log").open("w", encoding="utf-8")
    cmd = _gepa_archive_command(
        args=args,
        out=out,
        tracks=tracks,
        dev_split=dev_split,
        archive_size=archive_size,
        max_metric_calls=max_metric_calls,
        sampler_max_new_tokens=sampler_max_new_tokens,
        reflection_max_new_tokens=reflection_max_new_tokens,
        cuda_visible_devices=cuda_visible_devices,
    )
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    return proc


def _run_cells_for_root(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    root: Path,
    tracks: list[str],
    split: tuple[int, int],
    rollout_budget: int,
    num_shards: int,
    memory_num_shards: int,
    gpus: list[str],
    run_stage: str,
) -> None:
    from polaris.proicl.launcher import (
        aggregate,
        append_event,
        build_signal_cells,
        run_cells,
        write_json,
    )

    events = root / "events.jsonl"
    cells = build_signal_cells(
        root=root,
        tracks=tracks,
        split=split,
        rollout_budget=rollout_budget,
        num_shards=num_shards,
        memory_num_shards=memory_num_shards,
    )
    write_json(root / "proicl_signal_plan.json", [cell.to_jsonable() for cell in cells])
    append_event(events, "run_cells_start", root=str(root), tracks=tracks, split=list(split))
    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        run_cells(
            repo_root=REPO_ROOT,
            cells=cells,
            gpus=gpus,
            events_path=events,
            backend=args.backend,
            local_files_only=args.local_files_only,
            cost_cap_dollars=args.cost_cap_dollars,
            estimated_dollar_cost=args.estimated_dollar_cost_per_cell,
            estimated_wall_clock_seconds=args.estimated_wall_clock_seconds_per_cell,
            run_kind=args.run_kind,
            run_stage=run_stage,
            max_new_tokens=args.max_new_tokens,
            mcmc_steps=args.mcmc_steps,
            mcmc_block_num=args.mcmc_block_num,
            vllm_dtype=args.vllm_dtype,
            vllm_model_impl=args.vllm_model_impl,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_scoring_mode=args.vllm_scoring_mode,
            vllm_parity_artifact=str(args.vllm_parity_artifact)
            if args.vllm_parity_artifact is not None
            else None,
        )
        report = aggregate(root=root, tracks=tracks, out_dir=root / "analysis")
        write_json(root / "analysis" / "aggregate_stdout.json", report)
        append_event(events, "aggregate_done", out=str(root / "analysis"))
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def _run_cell_list(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    cells: list[Any],
    gpus: list[str],
    events: Path,
    root: Path,
    tracks: list[str],
    run_stage: str,
    aggregate_after: bool = False,
) -> None:
    from polaris.proicl.launcher import aggregate, run_cells, write_json

    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        run_cells(
            repo_root=REPO_ROOT,
            cells=cells,
            gpus=gpus,
            events_path=events,
            backend=args.backend,
            local_files_only=args.local_files_only,
            cost_cap_dollars=args.cost_cap_dollars,
            estimated_dollar_cost=args.estimated_dollar_cost_per_cell,
            estimated_wall_clock_seconds=args.estimated_wall_clock_seconds_per_cell,
            run_kind=args.run_kind,
            run_stage=run_stage,
            max_new_tokens=args.max_new_tokens,
            mcmc_steps=args.mcmc_steps,
            mcmc_block_num=args.mcmc_block_num,
            vllm_dtype=args.vllm_dtype,
            vllm_model_impl=args.vllm_model_impl,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_scoring_mode=args.vllm_scoring_mode,
            vllm_parity_artifact=str(args.vllm_parity_artifact)
            if args.vllm_parity_artifact is not None
            else None,
        )
        if aggregate_after:
            report = aggregate(root=root, tracks=tracks, out_dir=root / "analysis")
            write_json(root / "analysis" / "aggregate_stdout.json", report)
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def _partition_cells(cells: list[Any], *, stride: int, offset: int) -> list[Any]:
    if stride < 1:
        raise ValueError("--cell-stride must be >= 1")
    if offset < 0 or offset >= stride:
        raise ValueError("--cell-offset must satisfy 0 <= offset < --cell-stride")
    if stride == 1:
        return cells
    return [cell for idx, cell in enumerate(cells) if idx % stride == offset]


def main() -> None:
    args = _parse_args()
    if args.backend == "vllm":
        from polaris.infra.vllm_calibration import (
            CalibrationArtifactError,
            validate_vllm_calibration_artifact,
        )
        from polaris.registry import resolve_model

        try:
            validate_vllm_calibration_artifact(
                args.vllm_parity_artifact,
                expected_model_id=resolve_model("deepseek-r1-distill-qwen-1.5b").hf_id,
            )
        except CalibrationArtifactError as exc:
            raise SystemExit(f"vLLM calibration gate failed: {exc}") from exc
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    env = _setup_env(args)
    gpus = _visible_gpus(args)
    (args.root / "logs").mkdir(parents=True, exist_ok=True)

    _prefetch_models(args, env)

    if not args.skip_smoke:
        smoke_root = args.root / "smoke"
        _build_direct_archives(smoke_root, ["reasoning_gym_boxnet"])
        _build_gepa_archive(
            args=args,
            env=env,
            root=smoke_root,
            tracks=args.tracks,
            dev_split=(0, 1),
            archive_size=2,
            max_metric_calls=4,
            sampler_max_new_tokens=128,
            reflection_max_new_tokens=256,
            cuda_visible_devices=gpus[0] if gpus else None,
            force=args.force_gepa,
        )
        _run_cells_for_root(
            args=args,
            env=env,
            root=smoke_root,
            tracks=["reasoning_gym_boxnet"],
            split=(args.eval_split[0], args.eval_split[0] + 1),
            rollout_budget=2,
            num_shards=1,
            memory_num_shards=1,
            gpus=gpus[: min(len(gpus), 6)],
            run_stage="smoke",
        )
        if args.smoke_only:
            return

    full_root = args.root / "full"
    _build_direct_archives(full_root, args.tracks)
    from polaris.proicl.launcher import append_event, build_signal_cells, write_json

    events = full_root / "events.jsonl"
    all_cells = build_signal_cells(
        root=full_root,
        tracks=args.tracks,
        split=tuple(args.eval_split),
        rollout_budget=args.rollout_budget,
        num_shards=args.num_shards,
        memory_num_shards=args.memory_num_shards,
    )
    partitioned_all_cells = _partition_cells(
        all_cells,
        stride=args.cell_stride,
        offset=args.cell_offset,
    )
    write_json(full_root / "proicl_signal_plan.json", [cell.to_jsonable() for cell in all_cells])
    write_json(
        full_root / f"proicl_signal_plan.worker-{args.cell_offset}-of-{args.cell_stride}.json",
        [cell.to_jsonable() for cell in partitioned_all_cells],
    )
    append_event(events, "run_cells_start", root=str(full_root), tracks=args.tracks, split=list(args.eval_split))
    if args.cell_stride > 1:
        append_event(
            events,
            "cell_partition",
            stride=args.cell_stride,
            offset=args.cell_offset,
            total_cells=len(all_cells),
            worker_cells=len(partitioned_all_cells),
        )

    gepa_gpu = gpus[0]
    worker_gpus = gpus[1:] if len(gpus) > 1 else gpus
    gepa_proc = _start_gepa_archive(
        args=args,
        env=env,
        root=full_root,
        tracks=args.tracks,
        dev_split=tuple(args.gepa_dev_split),
        archive_size=args.archive_size,
        max_metric_calls=args.max_metric_calls,
        sampler_max_new_tokens=args.max_new_tokens,
        reflection_max_new_tokens=512,
        cuda_visible_devices=gepa_gpu,
        force=args.force_gepa,
    )
    direct_cells = [cell for cell in partitioned_all_cells if not cell.uses_gepa_archive and not cell.uses_memory]
    dependent_cells = [cell for cell in partitioned_all_cells if cell.uses_gepa_archive or cell.uses_memory]
    if gepa_proc is not None and len(gpus) > 1:
        append_event(events, "overlap_direct_cells_start", gepa_gpu=gepa_gpu, worker_gpus=worker_gpus)
        _run_cell_list(
            args=args,
            env=env,
            cells=direct_cells,
            gpus=worker_gpus,
            events=events,
            root=full_root,
            tracks=args.tracks,
            run_stage=args.run_stage,
        )
    rc = gepa_proc.wait() if gepa_proc is not None else 0
    if rc != 0:
        append_event(events, "gepa_archive_failed", returncode=rc)
        raise RuntimeError(f"GEPA archive build failed with return code {rc}")
    append_event(events, "gepa_archive_ready", gpu=gepa_gpu)
    remaining_cells = (
        dependent_cells
        if gepa_proc is not None and len(gpus) > 1
        else partitioned_all_cells
    )
    _run_cell_list(
        args=args,
        env=env,
        cells=remaining_cells,
        gpus=gpus,
        events=events,
        root=full_root,
        tracks=args.tracks,
        run_stage=args.run_stage,
        aggregate_after=not args.skip_aggregate and args.cell_stride == 1,
    )
    if args.skip_aggregate or args.cell_stride > 1:
        append_event(events, "aggregate_skipped", reason="partition_worker")
    else:
        append_event(events, "aggregate_done", out=str(full_root / "analysis"))


if __name__ == "__main__":
    main()
