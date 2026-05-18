#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${POLARIS_RUN_ROOT:-/workspace/polaris/runs/prorl_recovery}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RUN_ROOT}/flow_a100x4_probe/${STAMP}"
export OUT_DIR

mkdir -p "${OUT_DIR}" "${HF_HUB_CACHE}"

python - <<'PY' > "${OUT_DIR}/environment.json"
import json
import os
import platform
import shutil
import subprocess
import sys

def run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }

payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "cwd": os.getcwd(),
    "hf_home": os.environ.get("HF_HOME"),
    "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
    "polaris_run_root": os.environ.get("POLARIS_RUN_ROOT"),
    "secret_env_present": {
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        "HUGGINGFACE_HUB_TOKEN": bool(os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "XAI_API_KEY": bool(os.environ.get("XAI_API_KEY")),
    },
    "executables": {
        "nvidia-smi": shutil.which("nvidia-smi"),
        "python": shutil.which("python"),
    },
    "nvidia_smi_l": run(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else None,
    "nvidia_smi_query": run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader",
    ]) if shutil.which("nvidia-smi") else None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

python - <<'PY' > "${OUT_DIR}/preflight_imports.json"
import importlib.util
import json

modules = ["torch", "transformers", "huggingface_hub", "vllm", "litellm", "datasets"]
payload = {
    "imports": {module: bool(importlib.util.find_spec(module)) for module in modules}
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

python - <<'PY' > "${OUT_DIR}/gpu_check.json"
import json
import subprocess

proc = subprocess.run(
    ["nvidia-smi", "-L"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
gpu_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")]
payload = {
    "passed": proc.returncode == 0 and len(gpu_lines) == 4,
    "expected_gpus": 4,
    "observed_gpus": len(gpu_lines),
    "returncode": proc.returncode,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["passed"]:
    raise SystemExit(1)
PY

python - <<'PY' > "${OUT_DIR}/manifest.json"
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"]) if "OUT_DIR" in os.environ else None
payload = {
    "run_kind": "flow",
    "profile": "flow_a100_weekend",
    "purpose": "4x A100 provisioning and integrity probe only; no bulk experiment",
    "artifact_contract": [
        "environment.json",
        "preflight_imports.json",
        "gpu_check.json",
        "manifest.json",
        "audit.md",
    ],
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

cat > "${OUT_DIR}/audit.md" <<'EOF'
# Flow A100x4 Probe Audit

This probe validates that a Flow 4x A100 instance came up with four visible
GPUs, a writable Hugging Face cache path, and import visibility for the runtime
modules needed by the POLARIS ProRL recovery audit.

It does not run Phase 0, Phase 1, Phase 2, GEPA, or Phase 3.
EOF

echo "POLARIS_FLOW_A100X4_PROBE_OK ${OUT_DIR}"
