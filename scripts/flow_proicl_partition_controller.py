"""Launch and mirror partitioned ProICL workers on Flow bids.

This controller stays local. It waits for existing Flow bids to expose SSH,
starts a partition worker on each host, mirrors artifacts back without deleting
other workers' outputs, and runs the local aggregate once all expected cells are
present.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/polaris"
REMOTE_RUN_ROOT = "/workspace/polaris/runs/proicl_overnight_signal/"
DEFAULT_LOCAL_ROOT = REPO_ROOT / "runs" / "remote_mirrors" / "proicl_overnight_signal"


@dataclass
class Worker:
    bid: str
    offset: int
    launched: bool = False
    launch_returncode: int | None = None
    last_host: str | None = None
    last_sync_utc: str | None = None
    last_error: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bid", action="append", required=True, help="BID:OFFSET")
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--expected-metrics", type=int, default=360)
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--timeout-seconds", type=int, default=12 * 3600)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--state", type=Path, default=REPO_ROOT / "runs" / "proicl_overnight_signal" / "partition_controller_state.json")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "runs" / "proicl_overnight_signal" / "partition_controller_logs")
    return parser.parse_args()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: list[str], *, timeout: int | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        timeout=timeout,
        check=False,
        capture_output=capture,
        env=os.environ.copy(),
    )


def _flow_ssh_json(bid: str) -> dict[str, Any] | None:
    result = _run(["flow", "ssh", "--json", bid], timeout=30, capture=True)
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
        "ConnectTimeout=8",
        "-o",
        "ServerAliveInterval=8",
        "-o",
        "ServerAliveCountMax=1",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "Compression=yes",
    ]
    return f"{user}@{host}", opts


def _ssh_ready(info: dict[str, Any]) -> bool:
    host = info.get("host")
    if not host or host == "None":
        return False
    target, opts = _ssh_parts(info)
    result = _run(["ssh", *opts, target, "true"], timeout=20, capture=True)
    return result.returncode == 0


def _launch(worker: Worker, stride: int, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = log_dir / f"{worker.bid}.launch.stdout.log"
    stderr = log_dir / f"{worker.bid}.launch.stderr.log"
    cmd = [
        sys.executable,
        "scripts/flow_launch_proicl_signal.py",
        worker.bid,
        "--gpus",
        "0",
        "0",
        "0",
        "0",
        "--num-shards",
        "8",
        "--cell-stride",
        str(stride),
        "--cell-offset",
        str(worker.offset),
        "--skip-aggregate",
        "--timeout-seconds",
        "180",
        "--poll-seconds",
        "15",
    ]
    with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
        result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=out, stderr=err, check=False)
    worker.launch_returncode = result.returncode
    worker.launched = result.returncode == 0
    if result.returncode != 0:
        worker.last_error = f"launch_returncode={result.returncode}"


def _sync(info: dict[str, Any], local_root: Path) -> bool:
    local_root.mkdir(parents=True, exist_ok=True)
    target, opts = _ssh_parts(info)
    ssh_cmd = "ssh " + " ".join(opts)
    result = _run(
        [
            "rsync",
            "-az",
            "-e",
            ssh_cmd,
            f"{target}:{REMOTE_RUN_ROOT}",
            str(local_root) + "/",
        ],
        timeout=180,
        capture=True,
    )
    return result.returncode == 0


def _metrics_count(local_root: Path) -> int:
    runs_root = local_root / "full" / "runs"
    if not runs_root.exists():
        return 0
    return sum(1 for _ in runs_root.rglob("metrics.json"))


def _aggregate(local_root: Path) -> bool:
    code = (
        "from pathlib import Path; "
        "from polaris.proicl.launcher import aggregate; "
        "root=Path('runs/remote_mirrors/proicl_overnight_signal/full'); "
        "aggregate(root=root, tracks=('reasoning_gym_boxnet','reasoning_gym_graph_color','reasoning_gym_family_relationships'), out_dir=root/'analysis')"
    )
    result = _run(
        ["./.venv-eval/bin/python", "-c", code],
        timeout=120,
        capture=True,
    )
    return result.returncode == 0


def _write_state(path: Path, workers: list[Worker], metrics: int, done: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": _utc(),
        "metrics_count": metrics,
        "done": done,
        "workers": [asdict(worker) for worker in workers],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    workers = []
    for raw in args.bid:
        bid, offset = raw.split(":", 1)
        workers.append(Worker(bid=bid, offset=int(offset)))
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        for worker in workers:
            info = _flow_ssh_json(worker.bid)
            if not info:
                continue
            worker.last_host = info.get("host")
            if not _ssh_ready(info):
                continue
            if not worker.launched:
                _launch(worker, args.stride, args.log_dir)
            if worker.launched and _sync(info, args.local_root):
                worker.last_sync_utc = _utc()
        metrics = _metrics_count(args.local_root)
        done = metrics >= args.expected_metrics
        if done:
            done = _aggregate(args.local_root)
        _write_state(args.state, workers, metrics, done)
        if done:
            return
        time.sleep(args.poll_seconds)
    metrics = _metrics_count(args.local_root)
    _write_state(args.state, workers, metrics, False)
    raise SystemExit(124)


if __name__ == "__main__":
    main()
