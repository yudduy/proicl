"""Launch the paper-aligned ProICL preliminary run on an allocated 8xA100 Flow host.

This is a local controller. It does not create, cancel, or update Flow bids.
It waits for an already-provisioned task to expose SSH, syncs this checkout to
the host, runs the exact smoke from the ProICL preliminary plan, and starts the
main preliminary run as a systemd unit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/polaris"
REMOTE_VENV = "/mnt/local/venvs/proicl"
REMOTE_SERVICE = "proicl-a100x8-prelim.service"
TRACKS = (
    "reasoning_gym_family_relationships",
    "reasoning_gym_graph_color_n5",
    "reasoning_gym_graph_color_n8",
    "reasoning_gym_graph_color_n10",
    "reasoning_gym_graph_color_n13",
    "reasoning_gym_graph_color_n15",
    "reasoning_gym_graph_color_n18",
    "reasoning_gym_graph_color_n20",
    "reasoning_gym_boxnet",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Flow task/bid name or id, e.g. proicl-a100x8-cheap-20260521")
    parser.add_argument("--timeout-seconds", type=int, default=6 * 3600)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--local-mirror-root", type=Path, default=REPO_ROOT / "runs" / "remote_mirrors")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--force-relaunch", action="store_true")
    return parser.parse_args()


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    input_text: str | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        check=check,
        capture_output=capture,
        input=input_text,
        timeout=timeout,
    )


def _flow_ssh_json(task: str) -> dict[str, Any] | None:
    result = _run(["flow", "ssh", "--json", task], capture=True, check=False, timeout=45)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


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
        "GSSAPIAuthentication=no",
        "-o",
        "Compression=yes",
    ]
    return f"{user}@{host}", opts


def _ssh(info: dict[str, Any], remote_cmd: str, *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    target, opts = _ssh_parts(info)
    return _run(["ssh", *opts, target, remote_cmd], timeout=timeout)


def _ssh_script(
    info: dict[str, Any],
    script: str,
    *,
    timeout: int | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    target, opts = _ssh_parts(info)
    return _run(
        ["ssh", *opts, target, "bash -s"],
        input_text=script,
        timeout=timeout,
        capture=capture,
    )


def _ssh_ready(info: dict[str, Any]) -> bool:
    host = info.get("host")
    if not host or host == "None":
        return False
    target, opts = _ssh_parts(info)
    result = _run(["ssh", *opts, target, "true"], check=False, capture=True, timeout=20)
    return result.returncode == 0


def _wait_for_host(task: str, *, timeout_seconds: int, poll_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = _run(["flow", "status", task, "--json"], capture=True, check=False, timeout=60)
        if status.stdout.strip():
            print(status.stdout.strip(), flush=True)
        info = _flow_ssh_json(task)
        if info and _ssh_ready(info):
            return info
        print(f"waiting for SSH host for {task}; sleeping {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)
    raise TimeoutError(f"{task} did not expose SSH within {timeout_seconds}s")


def _rsync_repo(info: dict[str, Any]) -> None:
    _ssh(
        info,
        f"sudo mkdir -p {REMOTE_REPO} {REMOTE_REPO}/runs && "
        f"sudo chown -R $(id -u):$(id -g) {REMOTE_REPO}",
        timeout=60,
    )
    target, opts = _ssh_parts(info)
    ssh_cmd = "ssh " + " ".join(shlex.quote(x) for x in opts)
    excludes = [
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=.venv-eval/",
        "--exclude=.pytest_cache/",
        "--exclude=.mypy_cache/",
        "--exclude=runs/",
        "--exclude=__pycache__/",
    ]
    _run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            ssh_cmd,
            *excludes,
            str(REPO_ROOT) + "/",
            f"{target}:{REMOTE_REPO}/",
        ],
        timeout=20 * 60,
    )
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        _run(
            [
                "rsync",
                "-az",
                "-e",
                ssh_cmd,
                str(env_file),
                f"{target}:{REMOTE_REPO}/.env",
            ],
            timeout=120,
        )


def _remote_common_env() -> str:
    return f"""
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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
"""


def _install(info: dict[str, Any]) -> None:
    script = f"""
set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-dev build-essential rsync
cd {REMOTE_REPO}
python3 -m venv {REMOTE_VENV}
. {REMOTE_VENV}/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e ".[gepa_reflection]" reasoning-gym==0.1.25
"""
    _ssh_script(info, script, timeout=45 * 60)


def _verify(info: dict[str, Any], *, skip_tests: bool) -> None:
    test_cmd = "" if skip_tests else "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/unit/test_proicl.py tests/unit/test_prorl_recovery.py"
    script = f"""
set -euo pipefail
cd {REMOTE_REPO}
. {REMOTE_VENV}/bin/activate
{_remote_common_env()}
hostname
date -u
nvidia-smi -L
gpu_count="$(nvidia-smi -L | grep -c '^GPU ')"
test "$gpu_count" = "8"
bash scripts/check_protocol_sync.sh
{test_cmd}
"""
    _ssh_script(info, script, timeout=20 * 60)


def _run_smoke(info: dict[str, Any]) -> str:
    script = f"""
set -euo pipefail
cd {REMOTE_REPO}
. {REMOTE_VENV}/bin/activate
{_remote_common_env()}
RUN_ROOT=/workspace/polaris/runs/proicl_a100x8_smoke_$(date -u +%Y%m%dT%H%M%SZ)
python scripts/run_proicl_signal.py \\
  --root "$RUN_ROOT" \\
  --tracks reasoning_gym_family_relationships reasoning_gym_graph_color_n10 reasoning_gym_boxnet \\
  --eval-split 0 2 \\
  --gepa-dev-split 0 1 \\
  --rollout-budget 2 \\
  --archive-size 2 \\
  --max-metric-calls 4 \\
  --max-new-tokens 256 \\
  --mcmc-steps 2 \\
  --mcmc-block-num 2 \\
  --num-shards 2 \\
  --memory-num-shards 1 \\
  --gpus 0 1 \\
  --backend hf \\
  --run-kind flow \\
  --run-stage smoke \\
  --cost-cap-dollars 2 \\
  --estimated-dollar-cost-per-cell 0.05 \\
  --estimated-wall-clock-seconds-per-cell 900 \\
  --reflection-provider local-hf \\
  --reflection-model-id Qwen/Qwen2.5-7B-Instruct \\
  --smoke-only
echo "SMOKE_RUN_ROOT=$RUN_ROOT"
"""
    result = _ssh_script(info, script, timeout=3 * 3600, capture=True)
    smoke_root = ""
    for line in result.stdout.splitlines() if result.stdout else []:
        if line.startswith("SMOKE_RUN_ROOT="):
            smoke_root = line.split("=", 1)[1]
    return smoke_root


def _launch_main(info: dict[str, Any], *, force_relaunch: bool) -> str:
    active_check = "false" if force_relaunch else f"sudo systemctl is-active --quiet {REMOTE_SERVICE}"
    tracks = " ".join(TRACKS)
    script = f"""
set -euo pipefail
cd {REMOTE_REPO}
if {active_check}; then
  existing="$(sudo systemctl show -p MainPID --value {REMOTE_SERVICE} || true)"
  echo "MAIN_ALREADY_RUNNING pid=$existing"
  find /workspace/polaris/runs -maxdepth 1 -type d -name 'proicl_a100x8_prelim_*' | sort | tail -1 | sed 's#^#MAIN_RUN_ROOT=#'
  exit 0
fi
RUN_ROOT=/workspace/polaris/runs/proicl_a100x8_prelim_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT/logs"
cat > "$RUN_ROOT/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="__RUN_ROOT__"
cd /workspace/polaris
. /mnt/local/venvs/proicl/bin/activate
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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
python scripts/run_proicl_signal.py \\
  --root "$RUN_ROOT" \\
  --tracks {tracks} \\
  --eval-split 0 50 \\
  --gepa-dev-split 0 50 \\
  --rollout-budget 32 \\
  --archive-size 16 \\
  --max-metric-calls 256 \\
  --max-new-tokens 512 \\
  --mcmc-steps 4 \\
  --mcmc-block-num 4 \\
  --num-shards 4 \\
  --memory-num-shards 1 \\
  --gpus 0 1 2 3 4 5 6 7 \\
  --backend hf \\
  --run-kind flow \\
  --run-stage small_real_slice \\
  --cost-cap-dollars 10 \\
  --estimated-dollar-cost-per-cell 0.50 \\
  --estimated-wall-clock-seconds-per-cell 7200 \\
  --reflection-provider local-hf \\
  --reflection-model-id Qwen/Qwen2.5-7B-Instruct \\
  --skip-smoke \\
  > "$RUN_ROOT/logs/runner.stdout.log" \\
  2> "$RUN_ROOT/logs/runner.stderr.log"
SH
sed -i "s#__RUN_ROOT__#$RUN_ROOT#g" "$RUN_ROOT/run.sh"
chmod +x "$RUN_ROOT/run.sh"
sudo systemctl reset-failed {REMOTE_SERVICE} >/dev/null 2>&1 || true
sudo systemd-run \\
  --unit={REMOTE_SERVICE.removesuffix(".service")} \\
  --working-directory={REMOTE_REPO} \\
  /bin/bash "$RUN_ROOT/run.sh"
sleep 2
sudo systemctl show -p MainPID --value {REMOTE_SERVICE} > "$RUN_ROOT/runner.pid"
sudo systemctl status --no-pager {REMOTE_SERVICE} | sed -n '1,30p'
echo "MAIN_RUN_ROOT=$RUN_ROOT"
"""
    result = _ssh_script(info, script, timeout=120, capture=True)
    main_root = ""
    for line in result.stdout.splitlines() if result.stdout else []:
        if line.startswith("MAIN_RUN_ROOT="):
            main_root = line.split("=", 1)[1]
    if not main_root:
        raise RuntimeError("main run launched but MAIN_RUN_ROOT was not reported")
    return main_root


def _sync_artifacts(info: dict[str, Any], remote_run_root: str, local_mirror_root: Path) -> Path:
    target, opts = _ssh_parts(info)
    ssh_cmd = "ssh " + " ".join(shlex.quote(x) for x in opts)
    local_mirror_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            "-e",
            ssh_cmd,
            f"{target}:{remote_run_root}",
            str(local_mirror_root) + "/",
        ],
        timeout=20 * 60,
    )
    return local_mirror_root / Path(remote_run_root).name


def _status(info: dict[str, Any], main_root: str) -> None:
    script = f"""
set +e
echo "=== gpu ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=== service ==="
sudo systemctl status --no-pager {REMOTE_SERVICE} | sed -n '1,30p'
echo "=== events ==="
tail -n 40 {main_root}/full/events.jsonl 2>/dev/null || true
echo "=== metrics ==="
find {main_root}/full/runs -name metrics.json 2>/dev/null | wc -l
echo "=== logs ==="
tail -n 40 {main_root}/logs/runner.stderr.log 2>/dev/null || true
"""
    _ssh_script(info, script, timeout=120)


def main() -> None:
    args = _parse_args()
    info = _wait_for_host(
        args.task,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    _ssh(info, "hostname; date -u; nvidia-smi -L", timeout=60)
    _rsync_repo(info)
    if not args.skip_install:
        _install(info)
    _verify(info, skip_tests=args.skip_tests)
    smoke_root = "skipped"
    if not args.skip_smoke:
        smoke_root = _run_smoke(info)
    main_root = _launch_main(info, force_relaunch=args.force_relaunch)
    _status(info, main_root)
    local_path = _sync_artifacts(info, main_root, args.local_mirror_root)
    print(
        json.dumps(
            {
                "task": args.task,
                "host": info.get("host"),
                "smoke_root": smoke_root,
                "main_root": main_root,
                "local_mirror": str(local_path),
                "service": REMOTE_SERVICE,
                "expected_metrics": 189,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
