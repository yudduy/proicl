"""Compare SPS against block-MCMC power sampling on a small MATH500 slice."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-key", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--split", type=int, nargs=2, default=(0, 20))
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--mcmc-steps", type=int, default=10)
    parser.add_argument("--sps-block-size", type=int, default=192)
    parser.add_argument("--sps-top-k", type=int, default=8)
    parser.add_argument("--sps-candidate-pool-size", type=int, default=8)
    parser.add_argument("--sps-rollouts-per-candidate", type=int, default=8)
    parser.add_argument("--sps-rollout-horizon", type=int, default=128)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--vllm-parity-artifact", type=Path, required=True)
    parser.add_argument("--vllm-dtype", default="float32")
    parser.add_argument("--vllm-model-impl", default="transformers")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--vllm-scoring-mode", default="forced_decode_v0")
    parser.add_argument("--run-kind", default="local")
    parser.add_argument("--cost-cap-dollars", type=float, default=0.0)
    parser.add_argument("--estimated-dollar-cost-per-cell", type=float, default=0.0)
    parser.add_argument("--estimated-wall-clock-seconds-per-cell", type=float, default=7200)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=True)


def _write_math_archive(path: Path) -> None:
    from polaris.prorl_recovery.orchestration import write_archive

    write_archive(path, kind="rws_math_direct")


def _run_cell(
    *,
    args: argparse.Namespace,
    archive: Path,
    out: Path,
    power_sampler: str,
    block_num: int,
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_condition.py",
        "--track",
        "math500",
        "--model-key",
        args.model_key,
        "--condition",
        "single_prompt_power",
        "--archive",
        str(archive),
        "--split",
        str(args.split[0]),
        str(args.split[1]),
        "--seed",
        "17",
        "--polaris-source-hash",
        "sps-calibration",
        "--preregistration-anchor",
        "TODO.md#sps-transition-amendment",
        "--out",
        str(out),
        "--backend",
        args.backend,
        "--samples-per-problem",
        str(args.samples_per_problem),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--mcmc-steps",
        str(args.mcmc_steps),
        "--mcmc-block-num",
        str(block_num),
        "--power-sampler",
        power_sampler,
        "--sps-top-k",
        str(args.sps_top_k),
        "--sps-candidate-pool-size",
        str(args.sps_candidate_pool_size),
        "--sps-rollouts-per-candidate",
        str(args.sps_rollouts_per_candidate),
        "--sps-rollout-horizon",
        str(args.sps_rollout_horizon),
        "--vllm-dtype",
        args.vllm_dtype,
        "--vllm-model-impl",
        args.vllm_model_impl,
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--vllm-scoring-mode",
        args.vllm_scoring_mode,
        "--vllm-parity-artifact",
        str(args.vllm_parity_artifact),
        "--trajectory-cache",
        str(out / "trajectory_cache.sqlite"),
        "--run-stage",
        "small_real_slice",
        "--run-kind",
        args.run_kind,
        "--estimated-dollar-cost",
        str(args.estimated_dollar_cost_per_cell),
        "--estimated-wall-clock-seconds",
        str(args.estimated_wall_clock_seconds_per_cell),
        "--cost-cap-dollars",
        str(args.cost_cap_dollars),
        "--user-authorized-paid-run",
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    _run(cmd)


def _read_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "archive" / "rws_math_direct.json"
    _write_math_archive(archive)
    block_num = max(1, math.ceil(args.max_new_tokens / args.sps_block_size))

    mcmc_out = args.out / "mcmc"
    sps_out = args.out / "sps"
    _run_cell(
        args=args,
        archive=archive,
        out=mcmc_out,
        power_sampler="mcmc",
        block_num=block_num,
    )
    _run_cell(
        args=args,
        archive=archive,
        out=sps_out,
        power_sampler="sps",
        block_num=block_num,
    )

    mcmc = _read_metrics(mcmc_out)
    sps = _read_metrics(sps_out)
    diff = abs(float(mcmc["accuracy"]) - float(sps["accuracy"]))
    summary = {
        "passed": diff <= args.tolerance,
        "tolerance": args.tolerance,
        "accuracy_abs_diff": diff,
        "mcmc_accuracy": mcmc["accuracy"],
        "sps_accuracy": sps["accuracy"],
        "n_problems": mcmc["n_problems"],
        "max_new_tokens": args.max_new_tokens,
        "sps_block_size": args.sps_block_size,
        "block_num": block_num,
        "sps_top_k": args.sps_top_k,
        "sps_candidate_pool_size": args.sps_candidate_pool_size,
        "sps_rollouts_per_candidate": args.sps_rollouts_per_candidate,
        "sps_rollout_horizon": args.sps_rollout_horizon,
        "artifacts": {
            "mcmc": str(mcmc_out),
            "sps": str(sps_out),
            "archive": str(archive),
        },
    }
    (args.out / "sps_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("SPS calibration failed")


if __name__ == "__main__":
    main()
