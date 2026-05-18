"""Stage and launch the ProICL signal run on an allocated Flow host.

This script intentionally runs outside Flow. It waits until a bid exposes SSH,
rsyncs the current checkout plus the staged ProICL archive artifacts, installs
the repo on the remote host, and starts the resumable ProICL runner.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/polaris"
REMOTE_VENV = "/mnt/local/venvs/proicl"
REMOTE_SERVICE = "proicl-signal-root.service"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bid_id")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=[
            "reasoning_gym_boxnet",
            "reasoning_gym_graph_color",
            "reasoning_gym_family_relationships",
        ],
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--cell-stride", type=int, default=1)
    parser.add_argument("--cell-offset", type=int, default=0)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--root", default="runs/proicl_overnight_signal")
    parser.add_argument(
        "--staged-root",
        type=Path,
        default=REPO_ROOT / "runs" / "remote_mirrors" / "proicl_overnight_signal",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        check=check,
        capture_output=capture,
    )


def _flow_ssh_json(bid_id: str) -> dict[str, Any]:
    result = _run(["flow", "ssh", "--json", bid_id], capture=True)
    return json.loads(result.stdout)


def _wait_for_host(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = _run(["flow", "status", args.bid_id, "--json"], capture=True, check=False)
        if status.returncode == 0:
            print(status.stdout.strip(), flush=True)
        try:
            last = _flow_ssh_json(args.bid_id)
        except Exception as exc:  # pragma: no cover - defensive CLI wrapper
            print(f"flow ssh probe failed: {exc}", flush=True)
            last = {}
        host = last.get("host")
        if host and host != "None":
            target, opts = _ssh_parts(last)
            probe = subprocess.run(
                ["ssh", *opts, target, "true"],
                cwd=REPO_ROOT,
                text=True,
                check=False,
                capture_output=True,
            )
            if probe.returncode == 0:
                return last
            print(
                f"host {host} is not SSH-ready yet: rc={probe.returncode}",
                flush=True,
            )
        time.sleep(args.poll_seconds)
    raise TimeoutError(f"Flow bid {args.bid_id} did not expose SSH within {args.timeout_seconds}s")


def _ssh_parts(info: dict[str, Any]) -> tuple[str, list[str]]:
    user = info["user"]
    host = info["host"]
    port = str(info.get("port", 22))
    key = info["key_path"]
    opts = [
        "-p",
        port,
        "-i",
        key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "Compression=yes",
    ]
    return f"{user}@{host}", opts


def _ssh(info: dict[str, Any], remote_cmd: str) -> None:
    target, opts = _ssh_parts(info)
    _run(["ssh", *opts, target, remote_cmd])


def _rsync(info: dict[str, Any], src: str, dst: str, *, extra: list[str] | None = None) -> None:
    target, opts = _ssh_parts(info)
    ssh_cmd = "ssh " + " ".join(opts)
    cmd = ["rsync", "-az", "--delete", "-e", ssh_cmd]
    if extra:
        cmd.extend(extra)
    cmd.extend([src, f"{target}:{dst}"])
    _run(cmd)


def _sync_repo(info: dict[str, Any], staged_root: Path) -> None:
    _ssh(
        info,
        f"sudo mkdir -p {REMOTE_REPO} {REMOTE_REPO}/runs && "
        f"sudo chown -R $(id -u):$(id -g) {REMOTE_REPO}",
    )
    excludes = [
        "--exclude",
        ".git/",
        "--exclude",
        ".venv/",
        "--exclude",
        ".venv-eval/",
        "--exclude",
        ".pytest_cache/",
        "--exclude",
        ".mypy_cache/",
        "--exclude",
        "runs/",
        "--exclude",
        "__pycache__/",
    ]
    _rsync(info, str(REPO_ROOT) + "/", REMOTE_REPO + "/", extra=excludes)
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        _rsync(info, str(env_file), REMOTE_REPO + "/.env")
    if not staged_root.exists():
        raise FileNotFoundError(f"missing staged artifacts: {staged_root}")
    _rsync(info, str(staged_root) + "/", REMOTE_REPO + "/runs/proicl_overnight_signal/")


def _install(info: dict[str, Any]) -> None:
    _ssh(
        info,
        "set -euo pipefail; "
        "sudo apt-get update; "
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev build-essential; "
        f"cd {REMOTE_REPO}; "
        f"python3 -m venv {REMOTE_VENV}; "
        f". {REMOTE_VENV}/bin/activate; "
        "python -m pip install --upgrade pip; "
        "python -m pip install -e '.[gepa_reflection]' reasoning-gym",
    )


def _launch(info: dict[str, Any], args: argparse.Namespace) -> None:
    gpu_args = " ".join(str(gpu) for gpu in args.gpus)
    track_args = " ".join(str(track) for track in args.tracks)
    partition_args = ""
    if args.cell_stride != 1 or args.cell_offset != 0:
        partition_args += f"  --cell-stride {args.cell_stride} \\\n"
        partition_args += f"  --cell-offset {args.cell_offset} \\\n"
    if args.skip_aggregate:
        partition_args += "  --skip-aggregate \\\n"
    run_script = f"{REMOTE_REPO}/{args.root}/logs/run_proicl_signal_root.sh"
    remote_cmd = f"""
set -euo pipefail
cd {REMOTE_REPO}
if sudo systemctl is-active --quiet {REMOTE_SERVICE}; then
  sudo systemctl show -p MainPID --value {REMOTE_SERVICE} > {args.root}/runner.pid
  echo "runner already active: $(cat {args.root}/runner.pid)"
  exit 0
fi
mkdir -p {args.root}/logs
sudo mkdir -p /mnt/local/proicl-cache
sudo chown -R "$(id -u):$(id -g)" /mnt/local/proicl-cache
cat > {run_script} <<'PROICL_RUNNER'
#!/usr/bin/env bash
set -euo pipefail
cd {REMOTE_REPO}
. {REMOTE_VENV}/bin/activate
export HF_HOME=/mnt/local/proicl-cache/huggingface
export HF_HUB_CACHE=/mnt/local/proicl-cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/mnt/local/proicl-cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/local/proicl-cache/huggingface/transformers
export XDG_CACHE_HOME=/mnt/local/proicl-cache/xdg
export PIP_CACHE_DIR=/mnt/local/proicl-cache/pip
export TORCH_HOME=/mnt/local/proicl-cache/torch
export CUDA_CACHE_PATH=/mnt/local/proicl-cache/cuda
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export POLARIS_RWS_COMMIT=720a8e9d084c87a630595e316f5260f1d7c3446c
export POLARIS_GEPA_COMMIT=ce51b50cd196b539c25fae99ad0e0255c23004a4
export POLARIS_EVALPLUS_COMMIT=26d6d00bb1fd0fa37f39c99d5290da67891d1c5e
export POLARIS_DC_COMMIT=5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
python scripts/run_proicl_signal.py \\
  --root {args.root} \\
  --tracks {track_args} \\
  --eval-split 20 40 \\
  --gepa-dev-split 0 6 \\
  --rollout-budget 8 \\
  --archive-size 8 \\
  --max-metric-calls 64 \\
  --max-new-tokens 512 \\
  --num-shards {args.num_shards} \\
  --memory-num-shards 1 \\
  --gpus {gpu_args} \\
  --backend hf \\
  --run-kind flow \\
  --run-stage small_real_slice \\
  --cost-cap-dollars 4 \\
  --estimated-dollar-cost-per-cell 0.25 \\
  --estimated-wall-clock-seconds-per-cell 7200 \\
  --xai-reflection-cap-dollars 2 \\
{partition_args}  \\
  --skip-smoke \\
  > {args.root}/logs/runner.stdout.log \\
  2> {args.root}/logs/runner.stderr.log
PROICL_RUNNER
chmod +x {run_script}
sudo systemctl reset-failed {REMOTE_SERVICE} >/dev/null 2>&1 || true
sudo systemd-run \\
  --unit={REMOTE_SERVICE.removesuffix(".service")} \\
  --working-directory={REMOTE_REPO} \\
  /bin/bash {run_script}
sleep 2
sudo systemctl show -p MainPID --value {REMOTE_SERVICE} > {args.root}/runner.pid
echo "PROICL_RUNNER_STARTED $(cat {args.root}/runner.pid)"
sudo systemctl status --no-pager {REMOTE_SERVICE} | sed -n '1,18p'
"""
    _ssh(info, remote_cmd)


def main() -> None:
    args = _parse_args()
    info = _wait_for_host(args)
    _ssh(info, "hostname; date -u; nvidia-smi -L")
    _sync_repo(info, args.staged_root)
    if not args.skip_install:
        _install(info)
    _launch(info, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
