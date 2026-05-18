"""Compatibility shim for the generic POLARIS condition runner."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "greedy",
            "bon_temp1",
            "single_prompt_power",
            "single_best_prompt",
            "full_archive_fixed",
            "full_archive_mixed",
            "full_archive_decaying",
            "bon_temp1_archive",
        ],
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--test-slice", type=int, nargs=2, default=[0, 75], metavar=("START", "END")
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--polaris-source-hash", required=True)
    parser.add_argument("--preregistration-anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--trajectory-cache", type=Path, default=None)
    parser.add_argument(
        "--run-kind",
        choices=[
            "modal",
            "mithril",
            "flow",
            "phase",
            "farmshare",
            "cloudrift",
            "bulk",
            "local",
        ],
        default="phase",
    )
    parser.add_argument("--estimated-dollar-cost", type=float, default=None)
    parser.add_argument("--estimated-wall-clock-seconds", type=float, default=None)
    parser.add_argument("--cost-cap-dollars", type=float, default=None)
    parser.add_argument("--user-authorized-paid-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
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
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rws-commit", default="")
    parser.add_argument("--evalplus-commit", default="")
    parser.add_argument("--gepa-commit", default="")
    parser.add_argument("--dc-commit", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    argv = [
        str(REPO_ROOT / "scripts" / "run_condition.py"),
        "--track",
        "math500",
        "--model-key",
        "qwen2.5-math-7b",
        "--condition",
        args.condition,
        "--archive",
        str(args.archive),
        "--split",
        str(args.test_slice[0]),
        str(args.test_slice[1]),
        "--seed",
        str(args.seed),
        "--polaris-source-hash",
        args.polaris_source_hash,
        "--preregistration-anchor",
        args.preregistration_anchor,
        "--out",
        str(args.out),
        "--backend",
        args.backend,
        "--run-kind",
        args.run_kind,
        "--run-stage",
        "small_real_slice",
    ]
    if args.model_revision is not None:
        argv.extend(["--model-revision", args.model_revision])
    if args.trajectory_cache is not None:
        argv.extend(["--trajectory-cache", str(args.trajectory_cache)])
    if args.estimated_dollar_cost is not None:
        argv.extend(["--estimated-dollar-cost", str(args.estimated_dollar_cost)])
    if args.estimated_wall_clock_seconds is not None:
        argv.extend(
            ["--estimated-wall-clock-seconds", str(args.estimated_wall_clock_seconds)]
        )
    if args.cost_cap_dollars is not None:
        argv.extend(["--cost-cap-dollars", str(args.cost_cap_dollars)])
    if args.user_authorized_paid_run:
        argv.append("--user-authorized-paid-run")
    if args.preflight_only:
        argv.append("--preflight-only")
    if args.local_files_only:
        argv.append("--local-files-only")
    for flag, value in (
        ("--vllm-dtype", args.vllm_dtype),
        ("--vllm-model-impl", args.vllm_model_impl),
        ("--vllm-gpu-memory-utilization", args.vllm_gpu_memory_utilization),
        ("--vllm-scoring-mode", args.vllm_scoring_mode),
        ("--rws-commit", args.rws_commit),
        ("--evalplus-commit", args.evalplus_commit),
        ("--gepa-commit", args.gepa_commit),
        ("--dc-commit", args.dc_commit),
    ):
        argv.extend([flag, str(value)])
    if args.vllm_max_model_len is not None:
        argv.extend(["--vllm-max-model-len", str(args.vllm_max_model_len)])
    if args.vllm_parity_artifact is not None:
        argv.extend(["--vllm-parity-artifact", str(args.vllm_parity_artifact)])

    sys.argv = argv
    runpy.run_path(str(REPO_ROOT / "scripts" / "run_condition.py"), run_name="__main__")


if __name__ == "__main__":
    main()
