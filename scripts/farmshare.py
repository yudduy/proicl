"""FarmShare operations for the ProRL recovery audit.

Default mode prints the exact command or script. Use `--execute` only when the
launch fields and user authorization are already satisfied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _ssh(host: str, remote_cmd: str, *, execute: bool) -> None:
    cmd = ["ssh", host, remote_cmd]
    if execute:
        _run(cmd)
    else:
        print(" ".join(cmd))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="farmshare")
    sub = parser.add_subparsers(dest="action", required=True)

    probe = sub.add_parser("probe", help="print or run FarmShare GPU/QoS probes")
    probe.add_argument("--no-gpu", action="store_true")
    probe.add_argument("--execute", action="store_true")

    env = sub.add_parser("env", help="print or run micromamba env setup commands")
    env.add_argument("--remote-root", default="/scratch/users/$USER/polaris")
    env.add_argument("--execute", action="store_true")

    model = sub.add_parser("model", help="render a one-GPU model import smoke script")
    model.add_argument("--model-key", default="deepseek-r1-distill-qwen-1.5b")
    model.add_argument("--out", type=Path, default=None)

    submit = sub.add_parser("submit", help="render or submit a four-shard Slurm array")
    submit.add_argument("--job-name", default="polaris-rf")
    submit.add_argument("--remote-root", default="/scratch/users/$USER/polaris")
    submit.add_argument("--repo-dir", default="/scratch/users/$USER/polaris/repo")
    submit.add_argument("--num-shards", type=int, default=4)
    submit.add_argument("--array-tasks", type=int, default=None)
    submit.add_argument("--max-concurrent", type=int, default=None)
    submit.add_argument("--time-limit", default="02:00:00")
    submit.add_argument("--command", required=True)
    submit.add_argument("--script-out", type=Path, default=None)
    submit.add_argument("--execute", action="store_true")

    monitor = sub.add_parser("monitor", help="print or run squeue/sacct status")
    monitor.add_argument("--job-id", default=None)
    monitor.add_argument("--execute", action="store_true")

    sync = sub.add_parser("sync", help="print or run rsync for repo/artifacts")
    sync.add_argument("--direction", choices=["up", "down"], required=True)
    sync.add_argument("--remote-root", default="/scratch/users/$USER/polaris")
    sync.add_argument("--local-artifacts", type=Path, default=REPO_ROOT / "runs" / "farmshare")
    sync.add_argument("--execute", action="store_true")

    audit = sub.add_parser("audit", help="audit a local synced FarmShare artifact dir")
    audit.add_argument("--path", type=Path, required=True)
    return parser.parse_args()


def _cmd_probe(args: argparse.Namespace) -> None:
    from polaris.infra.farmshare import probe_commands

    remote = " && ".join(probe_commands(include_gpu=not args.no_gpu))
    _ssh(args.host, remote, execute=args.execute)


def _cmd_env(args: argparse.Namespace) -> None:
    from polaris.infra.farmshare import env_commands

    remote = " && ".join(env_commands(args.remote_root))
    _ssh(args.host, remote, execute=args.execute)


def _cmd_model(args: argparse.Namespace) -> None:
    from polaris.registry import resolve_model

    model = resolve_model(args.model_key)
    payload = {
        "model_key": model.key,
        "hf_id": model.hf_id,
        "revision": model.revision,
        "revision_commit": model.revision_commit,
    }
    script = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "export POLARIS_REMOTE_ROOT=/scratch/users/$USER/polaris\n"
        "export HF_HOME=/scratch/users/$USER/.cache/huggingface\n"
        "export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub\n"
        "export TRANSFORMERS_CACHE=$HF_HOME\n"
        "mkdir -p \"$HF_HOME\" \"$HUGGINGFACE_HUB_CACHE\" \"$TRANSFORMERS_CACHE\"\n"
        "python - <<'PY'\n"
        "from transformers import AutoConfig, AutoTokenizer\n"
        f"model_id = {model.hf_id!r}\n"
        f"revision = {model.revision!r}\n"
        "cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)\n"
        "tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)\n"
        "print({'model_type': cfg.model_type, 'vocab_size': len(tok)})\n"
        "PY\n"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(script, encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(script)


def _cmd_submit(args: argparse.Namespace) -> None:
    from polaris.infra.farmshare import SlurmArraySpec, render_slurm_array

    script = render_slurm_array(
        SlurmArraySpec(
            job_name=args.job_name,
            command=args.command,
            remote_root=args.remote_root,
            repo_dir=args.repo_dir,
            num_shards=args.num_shards,
            array_tasks=args.array_tasks,
            max_concurrent=args.max_concurrent,
            time_limit=args.time_limit,
        )
    )
    if args.script_out is not None:
        args.script_out.parent.mkdir(parents=True, exist_ok=True)
        args.script_out.write_text(script, encoding="utf-8")
    if not args.execute:
        print(script)
        return
    remote_script = f"{args.remote_root}/slurm/{args.job_name}.sbatch"
    subprocess.run(
        ["ssh", args.host, f"cat > {remote_script} && sbatch {remote_script}"],
        input=script,
        text=True,
        check=True,
    )


def _cmd_monitor(args: argparse.Namespace) -> None:
    cmd = "squeue -u $USER -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'"
    if args.job_id:
        cmd += f" && sacct -j {args.job_id} --format=JobID,JobName,State,Elapsed,ExitCode,NodeList"
    _ssh(args.host, cmd, execute=args.execute)


def _cmd_sync(args: argparse.Namespace) -> None:
    from polaris.infra.farmshare import rsync_to_farmshare_command

    if args.direction == "up":
        cmd = rsync_to_farmshare_command(
            host=args.host,
            local_repo=REPO_ROOT,
            remote_repo=f"{args.remote_root}/repo",
        )
    else:
        args.local_artifacts.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-az",
            f"{args.host}:{args.remote_root}/runs/prorl_recovery/",
            f"{args.local_artifacts}/",
        ]
    if args.execute:
        _run(cmd)
    else:
        print(" ".join(str(x) for x in cmd))


def _cmd_audit(args: argparse.Namespace) -> None:
    required = [
        "manifest.json",
        "archive.json",
        "candidates.jsonl",
        "scores.jsonl",
        "selected.jsonl",
        "metrics.json",
        "costs.json",
        "rollouts.json",
        "preflight.json",
        "environment.json",
        "run_plan_cell.json",
        "audit.md",
    ]
    missing = [name for name in required if not (args.path / name).exists()]
    payload = {"path": str(args.path), "passed": not missing, "missing": missing}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(1)


def main() -> None:
    args = _parse_args()
    dispatch = {
        "probe": _cmd_probe,
        "env": _cmd_env,
        "model": _cmd_model,
        "submit": _cmd_submit,
        "monitor": _cmd_monitor,
        "sync": _cmd_sync,
        "audit": _cmd_audit,
    }
    dispatch[args.action](args)


if __name__ == "__main__":
    main()
