#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the ProICL held-out experiment.

Default run:
  bash scripts/run_experiment.sh
  bash scripts/run_experiment.sh [auto|l40|a100|h100|generic] [options]

Compatibility aliases:
  bash scripts/run_experiment_l40.sh
  bash scripts/run_experiment_a100.sh
  bash scripts/run_experiment_h100.sh

Useful overrides:
  EVAL_END=30 ROLLOUT_BUDGET=2 bash scripts/run_experiment.sh h100
  DRY_RUN=1 bash scripts/run_experiment.sh l40
  ml python/3.12.1 && bash scripts/run_experiment.sh   # Sherlock

Environment knobs:
  RUN_ROOT                 Output series directory. Default: runs/experiment
  RUN_TAG                  Short tag inside the standardized run id. Default: heldout
  TRACKS                   Eval tracks. Default: boxnet acre game_of_life_halting graph_color_n12
  ARCHIVE_TRAIN_TRACKS     GEPA training tracks. Default: five in-distribution Reasoning Gym tasks
  ARCHIVE_HELDOUT_TRACKS   Held-out tracks for cross-family provenance. Default: TRACKS
  CONDITIONS               Default: base_greedy sps_only gepa_sps_fixed prorl_v2_greedy
  EVAL_START/EVAL_END      Held-out eval slice. Default: 20/70
  GEPA_DEV_START/END       GEPA dev slice. Default: 0/50
  ROLLOUT_BUDGET           Samples per non-greedy condition. Default: 8
  NUM_SHARDS               Per-condition shards. Default: detected GPU count, at least 1
  MAX_PARALLEL_CELLS       Concurrent vLLM cell workers. Default: host-memory-aware.
  SMOKE_MAX_PARALLEL_CELLS Developer smoke workers when SMOKE_ONLY=1.
  OVERLAP_GEPA_AND_CELLS=0/1 Override GEPA/direct-cell overlap. Default: auto.
  GPUS                     Optional comma-separated GPU ids; overrides auto-detection.
  GPU_MEMORY_MIB           Optional comma-separated GPU memory override for dry runs/tests.
  HOST_MEMORY_MIB          Optional host memory override for dry runs/tests.
  MAX_NEW_TOKENS           Generation cap. Default: 1024
  SPS_BLOCK_SIZE           Target SPS block size. Default: 192
  SPS_BLOCK_NUM            Override computed SPS block count.
  SPS_TOP_K                Candidate blocks kept per SPS step. Default: 8
  SPS_CANDIDATE_POOL_SIZE  Candidate blocks sampled per SPS step. Default: 8
  SPS_ROLLOUTS_PER_CANDIDATE Lookahead rollouts per candidate. Default: 8
  SPS_ROLLOUT_HORIZON      Lookahead horizon tokens. Default: 128
  SPS_CHAIN_BATCH_SIZE     Archive-prompt SPS chains per vLLM batch. Default: GPU-aware.
  GPU_PROFILE              auto, l40, a100, h100, or generic. Default: auto.
                           Can also be supplied as the first positional arg.
  PROFILE_MAX_PARALLEL_CELLS Profile-level cap for concurrent cell workers.
  HOST_MEMORY_PER_CELL_MIB Host RAM budget per concurrent vLLM cell worker.
  HOST_MEMORY_RESERVE_MIB  Host RAM reserve before assigning cell workers.
  VLLM_DTYPE               vLLM dtype. Default: bfloat16 for GPU profiles.
  VLLM_ATTENTION_BACKEND   Optional vLLM attention backend. L40/A100/H100 default: FLASH_ATTN.
  VLLM_PREFIX_CACHING      1/0. Default: 1.
  SPS_VLLM_BATCH_SIZE      Internal vLLM request microbatch for SPS.
  CALIBRATION_DTYPE        vLLM/HF parity calibration dtype. Default: float32.
  RUN_BACKEND_PREFLIGHT=1 Opt into production-shaped SPS/vLLM preflight.
  PREFLIGHT_ONLY=1        Run only the optional backend preflight, then exit.
  VLLM_PARITY_ARTIFACT     Existing calibration_summary.json. If unset, calibration is run.
  SKIP_INSTALL=1           Reuse the current environment.
  INSTALL_PROFILE          standard or full. Default: standard.
  REFLECTION_PROVIDER      xai or local-hf. Default: local-hf.
  WANDB_PROJECT            W&B project when WANDB_API_KEY is set. Default: proicl.
  SKIP_CALIBRATION=1       Require VLLM_PARITY_ARTIFACT instead of running calibration.
  SKIP_SPS_MATH500_CALIBRATION=0 Run the slow SPS-vs-MCMC MATH500 gate. Default: skipped.
  PYTHON                   Explicit Python 3.11/3.12 interpreter. Overrides VENV auto-detection.
  SMOKE_ONLY=1             Developer-only: run only the harness smoke.
  INCLUDE_CANDIDATES=1     Include candidates.jsonl in the final bundle.
  PROICL_DISABLE_TQDM=0    Use tqdm progress bars instead of plain progress lines.

Options:
  --gpu-profile PROFILE    auto, l40, a100, h100, or generic.
  --resume [TARGET]        Resume auto, latest, a UTC timestamp, a run id, or a run directory.
                           Default: auto, which resumes the latest incomplete matching run.
  --resume-latest          Alias for --resume latest.
  --fresh                  Force a new run.
  --progress-interval SEC  Active-cell progress update interval. Default: 60.
                           Use "off" or --quiet-progress to disable.
EOF
}

PROFILE_ARG=""
RESUME_ARG="${RESUME:-auto}"
FRESH_RUN=0
HEARTBEAT_ARG="${PROICL_CELL_HEARTBEAT_SECONDS:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    auto|l40|a100|h100|generic)
      if [[ -n "$PROFILE_ARG" ]]; then
        echo "GPU profile was provided more than once: $PROFILE_ARG and $1" >&2
        exit 2
      fi
      PROFILE_ARG="$1"
      shift
      ;;
    --gpu-profile=*)
      if [[ -n "$PROFILE_ARG" ]]; then
        echo "GPU profile was provided more than once: $PROFILE_ARG and ${1#--gpu-profile=}" >&2
        exit 2
      fi
      PROFILE_ARG="${1#--gpu-profile=}"
      if [[ -z "$PROFILE_ARG" ]]; then
        echo "--gpu-profile requires one of auto, l40, a100, h100, generic" >&2
        exit 2
      fi
      shift
      ;;
    --gpu-profile)
      if [[ -z "${2:-}" ]]; then
        echo "--gpu-profile requires one of auto, l40, a100, h100, generic" >&2
        exit 2
      fi
      if [[ -n "$PROFILE_ARG" ]]; then
        echo "GPU profile was provided more than once: $PROFILE_ARG and $2" >&2
        exit 2
      fi
      PROFILE_ARG="$2"
      shift 2
      ;;
    --resume=*)
      RESUME_ARG="${1#--resume=}"
      if [[ -z "$RESUME_ARG" ]]; then
        RESUME_ARG="latest"
      fi
      FRESH_RUN=0
      shift
      ;;
    --resume)
      if [[ -n "${2:-}" && "${2:-}" != --* ]]; then
        RESUME_ARG="$2"
        shift 2
      else
        RESUME_ARG="latest"
        shift
      fi
      FRESH_RUN=0
      ;;
    --resume-latest)
      RESUME_ARG="latest"
      FRESH_RUN=0
      shift
      ;;
    --fresh)
      RESUME_ARG=""
      RUN_TIMESTAMP=""
      FRESH_RUN=1
      shift
      ;;
    --progress-interval=*|--heartbeat=*)
      HEARTBEAT_ARG="${1#--heartbeat=}"
      HEARTBEAT_ARG="${HEARTBEAT_ARG#--progress-interval=}"
      shift
      ;;
    --progress-interval|--heartbeat)
      if [[ -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "$1 requires seconds or off" >&2
        exit 2
      fi
      HEARTBEAT_ARG="$2"
      shift 2
      ;;
    --quiet-progress|--no-heartbeat)
      HEARTBEAT_ARG="off"
      shift
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      echo "Usage: bash scripts/run_experiment.sh [auto|l40|a100|h100|generic] [--fresh]" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
INSTALL_PROFILE="${INSTALL_PROFILE:-standard}"
SKIP_CALIBRATION="${SKIP_CALIBRATION:-0}"
SKIP_SPS_MATH500_CALIBRATION="${SKIP_SPS_MATH500_CALIBRATION:-1}"
SKIP_BACKEND_PREFLIGHT="${SKIP_BACKEND_PREFLIGHT:-1}"
if [[ "${RUN_BACKEND_PREFLIGHT:-0}" == "1" ]]; then
  SKIP_BACKEND_PREFLIGHT=0
fi
SMOKE_ONLY="${SMOKE_ONLY:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
INCLUDE_CANDIDATES="${INCLUDE_CANDIDATES:-0}"
export PROICL_DISABLE_TQDM="${PROICL_DISABLE_TQDM:-1}"

RUN_ROOT="${RUN_ROOT:-runs/experiment}"
RUN_TAG="${RUN_TAG:-heldout}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-}"
TRACKS="${TRACKS:-reasoning_gym_boxnet reasoning_gym_acre reasoning_gym_game_of_life_halting reasoning_gym_graph_color_n12}"
ARCHIVE_TRAIN_TRACKS="${ARCHIVE_TRAIN_TRACKS:-reasoning_gym_family_relationships reasoning_gym_graph_color_n10 reasoning_gym_maze reasoning_gym_palindrome_generation reasoning_gym_letter_counting}"
ARCHIVE_HELDOUT_TRACKS="${ARCHIVE_HELDOUT_TRACKS:-$TRACKS}"
CONDITIONS="${CONDITIONS:-base_greedy sps_only gepa_sps_fixed prorl_v2_greedy}"

EVAL_START="${EVAL_START:-20}"
EVAL_END="${EVAL_END:-70}"
GEPA_DEV_START="${GEPA_DEV_START:-0}"
GEPA_DEV_END="${GEPA_DEV_END:-50}"
ROLLOUT_BUDGET="${ROLLOUT_BUDGET:-8}"
ARCHIVE_SIZE="${ARCHIVE_SIZE:-8}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-1500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SPS_BLOCK_SIZE="${SPS_BLOCK_SIZE:-192}"
if [[ -z "${SPS_BLOCK_NUM:-}" ]]; then
  SPS_BLOCK_NUM=$(( (MAX_NEW_TOKENS + SPS_BLOCK_SIZE - 1) / SPS_BLOCK_SIZE ))
fi
SPS_TOP_K="${SPS_TOP_K:-8}"
SPS_CANDIDATE_POOL_SIZE="${SPS_CANDIDATE_POOL_SIZE:-8}"
SPS_ROLLOUTS_PER_CANDIDATE="${SPS_ROLLOUTS_PER_CANDIDATE:-8}"
SPS_ROLLOUT_HORIZON="${SPS_ROLLOUT_HORIZON:-128}"

RUN_KIND="${RUN_KIND:-local}"
RUN_STAGE="${RUN_STAGE:-small_real_slice}"
REFLECTION_PROVIDER="${REFLECTION_PROVIDER:-local-hf}"
REFLECTION_MODEL_ID="${REFLECTION_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MATH_CALIB_START="${MATH_CALIB_START:-0}"
MATH_CALIB_END="${MATH_CALIB_END:-20}"
MATH_CALIB_MAX_NEW_TOKENS="${MATH_CALIB_MAX_NEW_TOKENS:-3072}"
SPS_CALIBRATION_TOLERANCE="${SPS_CALIBRATION_TOLERANCE:-0.02}"

VENV="${VENV:-.venv-eval}"
VENV_PY="$REPO_ROOT/$VENV/bin/python"
NEED_CREATE_VENV=0
python_is_supported() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
version = sys.version_info[:2]
raise SystemExit(0 if (3, 11) <= version < (3, 13) else 1)
PY
}
select_supported_python() {
  local candidate candidate_path
  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      candidate_path="$(command -v "$candidate")"
      if python_is_supported "$candidate_path"; then
        printf '%s\n' "$candidate_path"
        return 0
      fi
    fi
  done
  return 1
}
try_load_sherlock_python_module() {
  if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/lmod/lmod/init/bash
    fi
  fi
  if command -v module >/dev/null 2>&1; then
    module load python/3.12.1 >/dev/null 2>&1 && return 0
  fi
  return 1
}
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$VENV_PY" ]]; then
  PY="$VENV_PY"
else
  NEED_CREATE_VENV=1
  PY="$(select_supported_python || true)"
  if [[ -z "$PY" ]]; then
    try_load_sherlock_python_module || true
    PY="$(select_supported_python || true)"
  fi
  if [[ -z "$PY" ]]; then
    echo "ProICL requires Python 3.11 or 3.12 because vLLM 0.9.2 does not publish Python 3.13+ wheels." >&2
    echo "On Sherlock, run: ml python/3.12.1" >&2
    echo "Then re-run this script, or set PYTHON to a supported interpreter." >&2
    exit 1
  fi
fi
if ! python_is_supported "$PY"; then
  echo "ProICL requires Python 3.11 or 3.12 because vLLM 0.9.2 does not publish Python 3.13+ wheels; selected interpreter failed: $PY" >&2
  echo "On Sherlock, run: ml python/3.12.1" >&2
  echo "Then re-run this script, or set PYTHON to a supported interpreter." >&2
  exit 1
fi

resolve_resume_timestamp() {
  local target="$1"
  "$PY" - "$RUN_ROOT" "$target" "$RUN_TAG" "$RUN_STAGE" "$TRACKS" "$CONDITIONS" "$SMOKE_ONLY" <<'PY'
import json
import re
import sys
from pathlib import Path

base = Path(sys.argv[1])
target = sys.argv[2] or "latest"
run_tag = sys.argv[3] or None
run_stage = sys.argv[4] or None
tracks = sys.argv[5].split()
conditions = sys.argv[6].split()
smoke_only = sys.argv[7] == "1"
stamp_re = re.compile(r"(\d{8}T\d{6}Z)")


def stamp_from_text(value: str) -> str | None:
    match = stamp_re.search(value)
    return match.group(1) if match else None


def stamp_from_path(value: str) -> str | None:
    path = Path(value)
    if path.name == "full":
        path = path.parent
    return stamp_from_text(path.name) or stamp_from_text(value)


def run_index_candidates() -> list[tuple[str, float, str]]:
    candidates: list[tuple[str, float, str]] = []
    if not base.exists():
        return candidates
    for run_index in base.glob("proicl_*/run_index.json"):
        try:
            payload = json.loads(run_index.read_text(encoding="utf-8"))
        except Exception:
            continue
        identity = payload.get("identity", {})
        if run_tag and identity.get("tag") != run_tag:
            continue
        if run_stage and identity.get("run_stage") != run_stage:
            continue
        if tracks and payload.get("tracks") != tracks:
            continue
        if conditions and payload.get("conditions") != conditions:
            continue
        stamp = identity.get("timestamp") or stamp_from_text(identity.get("run_id", ""))
        if not stamp:
            continue
        try:
            mtime = run_index.parent.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((stamp, mtime, str(run_index.parent)))
    return candidates


def is_packaged(run_path: str) -> bool:
    path = Path(run_path)
    bundle_roots = [path / ("smoke" if smoke_only else "full"), path]
    return any((root / "results_bundle.tar.gz").exists() for root in bundle_roots)


def directory_candidates() -> list[tuple[str, float, str]]:
    candidates: list[tuple[str, float, str]] = []
    if not base.exists():
        return candidates
    tag_token = f"_{run_tag}_" if run_tag else None
    for path in base.glob("proicl_*"):
        if not path.is_dir():
            continue
        if tag_token and tag_token not in path.name:
            continue
        if run_stage and f"proicl_{run_stage.replace('_', '-')}" not in path.name:
            continue
        stamp = stamp_from_text(path.name)
        if not stamp:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((stamp, mtime, str(path)))
    return candidates


if target in {"", "auto", "latest"}:
    candidates = run_index_candidates() or directory_candidates()
    if target in {"", "auto"}:
        candidates = [item for item in candidates if not is_packaged(item[2])]
        if not candidates:
            print("")
            raise SystemExit(0)
    if not candidates:
        raise SystemExit(f"No matching ProICL run found under {base} for --resume latest")
    print(max(candidates, key=lambda item: (item[0], item[1]))[0])
    raise SystemExit(0)

stamp = stamp_from_path(target)
if stamp:
    print(stamp)
    raise SystemExit(0)

candidate_path = base / target
stamp = stamp_from_path(str(candidate_path))
if stamp:
    print(stamp)
    raise SystemExit(0)

raise SystemExit(
    "--resume target must be latest, a UTC timestamp, a standardized run id, "
    f"or a run directory; got {target!r}"
)
PY
}

if [[ -n "${HEARTBEAT_ARG:-}" ]]; then
  case "$HEARTBEAT_ARG" in
    off|none|false|False|0)
      export PROICL_CELL_HEARTBEAT_SECONDS=0
      ;;
    *)
      if ! awk 'BEGIN { exit !(ARGV[1] ~ /^[0-9]+([.][0-9]+)?$/) }' "$HEARTBEAT_ARG"; then
        echo "--progress-interval requires seconds or off; got $HEARTBEAT_ARG" >&2
        exit 2
      fi
      export PROICL_CELL_HEARTBEAT_SECONDS="$HEARTBEAT_ARG"
      ;;
  esac
else
  export PROICL_CELL_HEARTBEAT_SECONDS=60
fi
RESUME_SOURCE="fresh"
if [[ "$FRESH_RUN" == "1" ]]; then
  RUN_TIMESTAMP=""
elif [[ -n "${RUN_TIMESTAMP:-}" && "$RESUME_ARG" == "auto" ]]; then
  RESUME_SOURCE="legacy-env"
elif [[ -n "$RESUME_ARG" ]]; then
  RUN_TIMESTAMP="$(resolve_resume_timestamp "$RESUME_ARG")"
  if [[ -n "$RUN_TIMESTAMP" ]]; then
    RESUME_SOURCE="$RESUME_ARG"
  fi
elif [[ -n "${RUN_TIMESTAMP:-}" ]]; then
  RESUME_SOURCE="legacy-env"
fi

normalize_gpu_csv() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | tr ' \t\n;' ',')"
  raw="$(printf '%s' "$raw" | sed -E 's/(^|,)gpu:/\1/g; s/,+/,/g; s/^,//; s/,$//')"
  if [[ "$raw" == "NoDevFiles" || "$raw" == "none" || "$raw" == "N/A" || "$raw" == "void" ]]; then
    raw=""
  fi
  printf '%s\n' "$raw"
}

gpu_count_to_csv() {
  local count="$1"
  if [[ "$count" =~ ^[0-9]+$ && "$count" -gt 0 ]]; then
    seq 0 $((count - 1)) | paste -sd ',' -
  fi
}

slurm_gpu_list_to_csv() {
  local raw="$1"
  raw="$(normalize_gpu_csv "$raw")"
  echo "$raw"
}

slurm_gpu_count_to_csv() {
  local raw="$1"
  raw="$(normalize_gpu_csv "$raw")"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    gpu_count_to_csv "$raw"
    return
  fi
  echo "$raw"
}

detect_nvidia_smi_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
      | paste -sd ',' - || true
    return
  fi
  echo ""
}

map_gpu_csv_to_numeric_indices() {
  local raw="$1"
  raw="$(normalize_gpu_csv "$raw")"
  if [[ -z "$raw" ]]; then
    echo ""
    return
  fi
  if [[ "$raw" == "all" ]]; then
    detect_nvidia_smi_gpus
    return
  fi
  if [[ "$raw" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "$raw"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "$raw"
    return
  fi
  local mapping
  mapping="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$mapping" ]]; then
    echo "$raw"
    return
  fi
  local out=()
  local unresolved=0
  IFS=',' read -r -a requested <<<"$raw"
  for item in "${requested[@]}"; do
    item="$(printf '%s' "$item" | xargs)"
    if [[ -z "$item" ]]; then
      continue
    fi
    if [[ "$item" =~ ^[0-9]+$ ]]; then
      out+=("$item")
      continue
    fi
    local match
    match="$(
      awk -F',' -v target="$item" '
        {
          idx=$1; uuid=$2
          gsub(/^ +| +$/, "", idx)
          gsub(/^ +| +$/, "", uuid)
          if (uuid == target) { print idx; exit }
        }
      ' <<<"$mapping"
    )"
    if [[ -n "$match" ]]; then
      out+=("$match")
    else
      unresolved=1
    fi
  done
  if [[ "$unresolved" == "0" && "${#out[@]}" -gt 0 ]]; then
    (IFS=','; echo "${out[*]}")
    return
  fi
  local visible_count
  visible_count="$(detect_nvidia_smi_gpus)"
  if [[ -n "$visible_count" ]]; then
    echo "$visible_count"
  else
    echo "$raw"
  fi
}

detect_assigned_gpus() {
  local source="default"
  local raw=""
  if [[ -n "${GPUS:-}" ]]; then
    source="GPUS"
    raw="$GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    source="CUDA_VISIBLE_DEVICES"
    raw="$CUDA_VISIBLE_DEVICES"
  elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    source="SLURM_STEP_GPUS"
    raw="$(slurm_gpu_list_to_csv "$SLURM_STEP_GPUS")"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    source="SLURM_JOB_GPUS"
    raw="$(slurm_gpu_list_to_csv "$SLURM_JOB_GPUS")"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    source="SLURM_GPUS_ON_NODE"
    raw="$(slurm_gpu_count_to_csv "$SLURM_GPUS_ON_NODE")"
  elif [[ -n "${SLURM_GPUS:-}" ]]; then
    source="SLURM_GPUS"
    raw="$(slurm_gpu_count_to_csv "$SLURM_GPUS")"
  elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES:-}" != "all" ]]; then
    source="NVIDIA_VISIBLE_DEVICES"
    raw="$NVIDIA_VISIBLE_DEVICES"
  else
    source="nvidia-smi"
    raw="$(detect_nvidia_smi_gpus)"
  fi
  raw="$(normalize_gpu_csv "$raw")"
  raw="$(map_gpu_csv_to_numeric_indices "$raw")"
  if [[ -z "$raw" ]]; then
    source="fallback"
    raw="0"
  fi
  printf '%s\t%s\n' "$source" "$raw"
}

count_gpu_csv() {
  awk -F',' '
    { n = 0; for (i = 1; i <= NF; i++) { if ($i != "") n++ } print n }
  ' <<<"$1"
}

read -r GPU_DETECTION_SOURCE ASSIGNED_GPUS < <(detect_assigned_gpus)
export CUDA_VISIBLE_DEVICES="$ASSIGNED_GPUS"

detect_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    count_gpu_csv "$CUDA_VISIBLE_DEVICES"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | grep -c '^GPU ' || true
    return
  fi
  echo 1
}

detect_gpu_profile() {
  local raw_names
  raw_names="${GPU_NAMES:-}"
  if [[ -z "$raw_names" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    raw_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  fi
  case "$(tr '[:upper:]' '[:lower:]' <<<"$raw_names")" in
    *h100*) echo "h100" ;;
    *a100*) echo "a100" ;;
    *l40*) echo "l40" ;;
    *) echo "generic" ;;
  esac
}

detect_gpu_names() {
  if [[ -n "${GPU_NAMES:-}" ]]; then
    tr '\n' ';' <<<"$GPU_NAMES" | sed 's/;*$//'
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
      | paste -sd ';' - || true
    return
  fi
  echo "unknown"
}

detect_min_gpu_memory_mib() {
  if [[ -n "${GPU_MEMORY_MIB:-}" ]]; then
    awk -F',' '
      { for (i = 1; i <= NF; i++) { gsub(/[^0-9]/, "", $i); if ($i + 0 > 0 && (min == 0 || $i + 0 < min)) min = $i + 0 } }
      END { print min + 0 }
    ' <<<"$GPU_MEMORY_MIB"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null \
      | awk -v selected="${CUDA_VISIBLE_DEVICES:-}" -F',' '
          BEGIN {
            n = split(selected, parts, ",")
            numeric_selected = 0
            for (i = 1; i <= n; i++) {
              gsub(/^ +| +$/, "", parts[i])
              if (parts[i] ~ /^[0-9]+$/) {
                wanted[parts[i]] = 1
                numeric_selected = 1
              }
            }
          }
          {
            idx = $1
            mem = $2
            gsub(/^ +| +$/, "", idx)
            gsub(/[^0-9]/, "", mem)
            if (numeric_selected && !(idx in wanted)) next
            if (mem + 0 > 0 && (min == 0 || mem + 0 < min)) min = mem + 0
          }
          END { print min + 0 }
        '
    return
  fi
  echo 0
}

detect_min_gpu_compute_cap_x10() {
  if [[ -n "${GPU_COMPUTE_CAP:-}" ]]; then
    awk -F',' '
      {
        for (i = 1; i <= NF; i++) {
          value = $i
          gsub(/^ +| +$/, "", value)
          if (value ~ /^[0-9]+[.][0-9]+$/) {
            split(value, parts, ".")
            cap = parts[1] * 10 + parts[2]
          } else {
            gsub(/[^0-9]/, "", value)
            cap = value + 0
          }
          if (cap > 0 && (min == 0 || cap < min)) min = cap
        }
      }
      END { print min + 0 }
    ' <<<"$GPU_COMPUTE_CAP"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local raw_caps
    raw_caps="$(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$raw_caps" ]]; then
      awk -v selected="${CUDA_VISIBLE_DEVICES:-}" -F',' '
          BEGIN {
            n = split(selected, parts, ",")
            numeric_selected = 0
            for (i = 1; i <= n; i++) {
              gsub(/^ +| +$/, "", parts[i])
              if (parts[i] ~ /^[0-9]+$/) {
                wanted[parts[i]] = 1
                numeric_selected = 1
              }
            }
          }
          {
            idx = $1
            cap_raw = $2
            gsub(/^ +| +$/, "", idx)
            gsub(/^ +| +$/, "", cap_raw)
            if (numeric_selected && !(idx in wanted)) next
            if (cap_raw ~ /^[0-9]+[.][0-9]+$/) {
              split(cap_raw, parts2, ".")
              cap = parts2[1] * 10 + parts2[2]
            } else {
              gsub(/[^0-9]/, "", cap_raw)
              cap = cap_raw + 0
            }
            if (cap > 0 && (min == 0 || cap < min)) min = cap
          }
          END { print min + 0 }
        ' <<<"$raw_caps"
      return
    fi
  fi
  case "$(tr '[:upper:]' '[:lower:]' <<<"${GPU_NAMES_DETECTED:-${GPU_NAMES:-}}")" in
    *"titan rtx"*|*"rtx 20"*|*"2080"*|*"2070"*|*"2060"*|*"v100"*|*"t4"*)
      echo 75
      ;;
    *)
      echo 0
      ;;
  esac
}

_mib_from_bytes() {
  awk '{ printf "%d\n", $1 / 1024 / 1024 }'
}

detect_host_memory_mib() {
  if [[ -n "${HOST_MEMORY_MIB:-}" ]]; then
    awk '{ gsub(/[^0-9]/, "", $0); print $0 + 0 }' <<<"$HOST_MEMORY_MIB"
    return
  fi
  if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE:-0}" != "0" ]]; then
    awk '{ gsub(/[^0-9]/, "", $0); print $0 + 0 }' <<<"$SLURM_MEM_PER_NODE"
    return
  fi
  if [[ -n "${SLURM_MEM_PER_GPU:-}" && "${SLURM_MEM_PER_GPU:-0}" != "0" ]]; then
    awk -v g="$GPU_COUNT" '{ gsub(/[^0-9]/, "", $0); print ($0 + 0) * g }' <<<"$SLURM_MEM_PER_GPU"
    return
  fi
  if [[ -n "${SLURM_MEM_PER_CPU:-}" && "${SLURM_MEM_PER_CPU:-0}" != "0" ]]; then
    local cpus
    cpus="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
    awk -v c="$cpus" '{ gsub(/[^0-9]/, "", $0); print ($0 + 0) * c }' <<<"$SLURM_MEM_PER_CPU"
    return
  fi
  if [[ -f /sys/fs/cgroup/memory.max ]]; then
    local raw
    raw="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"
    if [[ "$raw" =~ ^[0-9]+$ && "$raw" -gt 0 && "$raw" -lt 9000000000000000000 ]]; then
      _mib_from_bytes <<<"$raw"
      return
    fi
  fi
  if [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    local raw
    raw="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)"
    if [[ "$raw" =~ ^[0-9]+$ && "$raw" -gt 0 && "$raw" -lt 9000000000000000000 ]]; then
      _mib_from_bytes <<<"$raw"
      return
    fi
  fi
  if [[ -f /proc/meminfo ]]; then
    awk '/MemTotal:/ { printf "%d\n", $2 / 1024 }' /proc/meminfo
    return
  fi
  echo 0
}

default_parallel_cells() {
  local default="$GPU_COUNT"
  local reserve="${HOST_MEMORY_RESERVE_MIB:-16384}"
  local per_cell="${HOST_MEMORY_PER_CELL_MIB:-32768}"
  local profile_cap="${PROFILE_MAX_PARALLEL_CELLS:-0}"
  if [[ "$HOST_MEMORY_MIB_DETECTED" -gt "$reserve" ]]; then
    local by_host=$(( (HOST_MEMORY_MIB_DETECTED - reserve) / per_cell ))
    if [[ "$by_host" -lt 1 ]]; then
      by_host=1
    fi
    if [[ "$by_host" -lt "$default" ]]; then
      default="$by_host"
    fi
  elif [[ "$GPU_MIN_MEMORY_MIB" -gt 0 && "$GPU_MIN_MEMORY_MIB" -lt 60000 ]]; then
    default=1
  fi
  if [[ "$default" -lt 1 ]]; then
    default=1
  fi
  if [[ "$default" -gt "$GPU_COUNT" ]]; then
    default="$GPU_COUNT"
  fi
  if [[ "$profile_cap" -gt 0 && "$default" -gt "$profile_cap" ]]; then
    default="$profile_cap"
  fi
  echo "$default"
}

GPU_COUNT="$(detect_gpu_count)"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi
GPU_NAMES_DETECTED="$(detect_gpu_names)"
GPU_MIN_MEMORY_MIB="$(detect_min_gpu_memory_mib)"
GPU_MIN_COMPUTE_CAP_X10="$(detect_min_gpu_compute_cap_x10)"
HOST_MEMORY_MIB_DETECTED="$(detect_host_memory_mib)"
NUM_SHARDS="${NUM_SHARDS:-$GPU_COUNT}"
if [[ "$EVAL_START" -lt 0 || "$EVAL_END" -le "$EVAL_START" ]]; then
  echo "Eval split must satisfy 0 <= EVAL_START < EVAL_END; got $EVAL_START/$EVAL_END" >&2
  exit 1
fi
EVAL_PROBLEM_COUNT=$((EVAL_END - EVAL_START))
if [[ "$NUM_SHARDS" -gt "$EVAL_PROBLEM_COUNT" ]]; then
  NUM_SHARDS="$EVAL_PROBLEM_COUNT"
fi
GPU_PROFILE="${PROFILE_ARG:-${GPU_PROFILE:-auto}}"
if [[ "$GPU_PROFILE" == "auto" ]]; then
  GPU_PROFILE="$(detect_gpu_profile)"
fi
case "$GPU_PROFILE" in
  l40)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.78"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.45"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="10800"
    DEFAULT_SPS_CHAIN_BATCH_SIZE="1"
    DEFAULT_SPS_VLLM_BATCH_SIZE="32"
    DEFAULT_VLLM_DTYPE="bfloat16"
    DEFAULT_VLLM_ATTENTION_BACKEND="FLASH_ATTN"
    DEFAULT_VLLM_PREFIX_CACHING="1"
    DEFAULT_PROFILE_MAX_PARALLEL_CELLS="2"
    DEFAULT_HOST_MEMORY_RESERVE_MIB="32768"
    DEFAULT_HOST_MEMORY_PER_CELL_MIB="49152"
    ;;
  h100)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.88"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.60"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="3600"
    DEFAULT_SPS_CHAIN_BATCH_SIZE="2"
    DEFAULT_SPS_VLLM_BATCH_SIZE="64"
    DEFAULT_VLLM_DTYPE="bfloat16"
    DEFAULT_VLLM_ATTENTION_BACKEND="FLASH_ATTN"
    DEFAULT_VLLM_PREFIX_CACHING="1"
    DEFAULT_PROFILE_MAX_PARALLEL_CELLS="6"
    DEFAULT_HOST_MEMORY_RESERVE_MIB="32768"
    DEFAULT_HOST_MEMORY_PER_CELL_MIB="40960"
    ;;
  a100)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.85"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.55"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="7200"
    DEFAULT_SPS_CHAIN_BATCH_SIZE="2"
    DEFAULT_SPS_VLLM_BATCH_SIZE="32"
    DEFAULT_VLLM_DTYPE="bfloat16"
    DEFAULT_VLLM_ATTENTION_BACKEND="FLASH_ATTN"
    DEFAULT_VLLM_PREFIX_CACHING="1"
    DEFAULT_PROFILE_MAX_PARALLEL_CELLS="4"
    DEFAULT_HOST_MEMORY_RESERVE_MIB="32768"
    DEFAULT_HOST_MEMORY_PER_CELL_MIB="40960"
    ;;
  generic)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.80"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.50"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="7200"
    DEFAULT_SPS_CHAIN_BATCH_SIZE="1"
    DEFAULT_SPS_VLLM_BATCH_SIZE="16"
    DEFAULT_VLLM_DTYPE="bfloat16"
    DEFAULT_VLLM_ATTENTION_BACKEND=""
    DEFAULT_VLLM_PREFIX_CACHING="1"
    DEFAULT_PROFILE_MAX_PARALLEL_CELLS="1"
    DEFAULT_HOST_MEMORY_RESERVE_MIB="32768"
    DEFAULT_HOST_MEMORY_PER_CELL_MIB="49152"
    ;;
  *)
    echo "GPU_PROFILE must be one of auto, l40, a100, h100, generic; got $GPU_PROFILE" >&2
    exit 1
    ;;
esac

if [[ "$GPU_MIN_MEMORY_MIB" -gt 0 && "$GPU_MIN_MEMORY_MIB" -lt 60000 ]]; then
  DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.80"
  DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.50"
  DEFAULT_SPS_CHAIN_BATCH_SIZE="1"
  DEFAULT_SPS_VLLM_BATCH_SIZE="${DEFAULT_SPS_VLLM_BATCH_SIZE:-16}"
fi
if [[ "$GPU_MIN_COMPUTE_CAP_X10" -gt 0 && "$GPU_MIN_COMPUTE_CAP_X10" -lt 80 ]]; then
  DEFAULT_VLLM_DTYPE="float16"
fi

ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL="${ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL:-$DEFAULT_WALL_CLOCK_SECONDS_PER_CELL}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}"
CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="${CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION}"
SPS_CHAIN_BATCH_SIZE="${SPS_CHAIN_BATCH_SIZE:-$DEFAULT_SPS_CHAIN_BATCH_SIZE}"
SPS_VLLM_BATCH_SIZE="${SPS_VLLM_BATCH_SIZE:-$DEFAULT_SPS_VLLM_BATCH_SIZE}"
VLLM_DTYPE="${VLLM_DTYPE:-$DEFAULT_VLLM_DTYPE}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-$DEFAULT_VLLM_ATTENTION_BACKEND}"
VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-$DEFAULT_VLLM_PREFIX_CACHING}"
CALIBRATION_DTYPE="${CALIBRATION_DTYPE:-float32}"
if [[ "$GPU_MIN_MEMORY_MIB" -gt 0 && "$GPU_MIN_MEMORY_MIB" -lt 60000 && "${ALLOW_LOW_VRAM_HIGH_UTIL:-0}" != "1" ]]; then
  VLLM_GPU_MEMORY_UTILIZATION="0.80"
  CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.50"
  SPS_CHAIN_BATCH_SIZE="1"
fi
if [[ "$GPU_MIN_COMPUTE_CAP_X10" -gt 0 && "$GPU_MIN_COMPUTE_CAP_X10" -lt 80 && "$VLLM_DTYPE" == "bfloat16" ]]; then
  if [[ "${ALLOW_UNSUPPORTED_BF16:-0}" == "1" ]]; then
    echo "Warning: keeping VLLM_DTYPE=bfloat16 on compute capability ${GPU_MIN_COMPUTE_CAP_X10}; vLLM may reject this GPU." >&2
  else
    echo "Detected GPU compute capability ${GPU_MIN_COMPUTE_CAP_X10}; switching vLLM dtype from bfloat16 to float16." >&2
    VLLM_DTYPE="float16"
  fi
fi
PROFILE_MAX_PARALLEL_CELLS="${PROFILE_MAX_PARALLEL_CELLS:-$DEFAULT_PROFILE_MAX_PARALLEL_CELLS}"
HOST_MEMORY_RESERVE_MIB="${HOST_MEMORY_RESERVE_MIB:-$DEFAULT_HOST_MEMORY_RESERVE_MIB}"
HOST_MEMORY_PER_CELL_MIB="${HOST_MEMORY_PER_CELL_MIB:-$DEFAULT_HOST_MEMORY_PER_CELL_MIB}"
MAX_PARALLEL_CELLS="${MAX_PARALLEL_CELLS:-$(default_parallel_cells)}"
if [[ "$MAX_PARALLEL_CELLS" -lt 1 ]]; then
  MAX_PARALLEL_CELLS=1
fi
if [[ "$MAX_PARALLEL_CELLS" -gt "$GPU_COUNT" ]]; then
  MAX_PARALLEL_CELLS="$GPU_COUNT"
fi
SMOKE_MAX_PARALLEL_CELLS="${SMOKE_MAX_PARALLEL_CELLS:-1}"
if [[ "$SMOKE_MAX_PARALLEL_CELLS" -lt 1 ]]; then
  SMOKE_MAX_PARALLEL_CELLS=1
fi
if [[ "$SMOKE_MAX_PARALLEL_CELLS" -gt "$MAX_PARALLEL_CELLS" ]]; then
  SMOKE_MAX_PARALLEL_CELLS="$MAX_PARALLEL_CELLS"
fi
if [[ -z "${OVERLAP_GEPA_AND_CELLS:-}" ]]; then
  if [[ "$GPU_COUNT" -gt 1 && "$MAX_PARALLEL_CELLS" -gt 1 ]]; then
    OVERLAP_GEPA_AND_CELLS=1
  else
    OVERLAP_GEPA_AND_CELLS=0
  fi
fi
if [[ "$GPU_COUNT" -le 1 ]]; then
  OVERLAP_GEPA_AND_CELLS=0
fi
PARALLELISM_STRATEGY="full experiment only; host-memory-aware cell workers; GEPA/direct-cell overlap enabled on multi-GPU"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-24000}"
if [[ "$GPU_MIN_MEMORY_MIB" -gt 0 && "$GPU_MIN_MEMORY_MIB" -lt "$MIN_GPU_MEMORY_MIB" ]]; then
  echo "Detected minimum GPU memory ${GPU_MIN_MEMORY_MIB}MiB, below required ${MIN_GPU_MEMORY_MIB}MiB." >&2
  echo "Use a larger GPU profile or lower the experiment settings before launching." >&2
  exit 1
fi

CACHE_ROOT="${PROICL_CACHE_ROOT:-$REPO_ROOT/.cache/proicl}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export PROICL_SPS_CHAIN_BATCH_SIZE="$SPS_CHAIN_BATCH_SIZE"
export PROICL_SPS_VLLM_BATCH_SIZE="$SPS_VLLM_BATCH_SIZE"
export SPS_VLLM_BATCH_SIZE="$SPS_VLLM_BATCH_SIZE"
if [[ "$VLLM_PREFIX_CACHING" != "0" && "$VLLM_PREFIX_CACHING" != "1" ]]; then
  echo "VLLM_PREFIX_CACHING must be 0 or 1; got $VLLM_PREFIX_CACHING" >&2
  exit 1
fi
if [[ -n "$VLLM_ATTENTION_BACKEND" ]]; then
  export VLLM_ATTENTION_BACKEND
else
  unset VLLM_ATTENTION_BACKEND
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$HF_HOME" "$PIP_CACHE_DIR"
fi

write_launch_config() {
  local out="$1"
  "$PY" - "$out" <<'PY'
import json
import os
import sys

out = sys.argv[1]
payload = {
    "schema": "proicl_launch_config.v1",
    "gpu_profile": os.environ["PROICL_GPU_PROFILE"],
    "gpu_detection_source": os.environ["PROICL_GPU_DETECTION_SOURCE"],
    "gpu_names": os.environ.get("PROICL_GPU_NAMES", "unknown"),
    "gpu_min_memory_mib": int(os.environ.get("PROICL_GPU_MIN_MEMORY_MIB", "0")),
    "gpu_min_compute_cap_x10": int(os.environ.get("PROICL_GPU_MIN_COMPUTE_CAP_X10", "0")),
    "host_memory_mib": int(os.environ.get("PROICL_HOST_MEMORY_MIB", "0")),
    "gpu_count": int(os.environ["PROICL_GPU_COUNT"]),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "num_shards": int(os.environ["PROICL_NUM_SHARDS"]),
    "max_parallel_cells": int(os.environ["PROICL_MAX_PARALLEL_CELLS"]),
    "smoke_max_parallel_cells": int(os.environ["PROICL_SMOKE_MAX_PARALLEL_CELLS"]),
    "overlap_gepa_and_cells": os.environ["PROICL_OVERLAP_GEPA_AND_CELLS"] == "1",
    "profile_max_parallel_cells": int(os.environ["PROICL_PROFILE_MAX_PARALLEL_CELLS"]),
    "host_memory_reserve_mib": int(os.environ["PROICL_HOST_MEMORY_RESERVE_MIB"]),
    "host_memory_per_cell_mib": int(os.environ["PROICL_HOST_MEMORY_PER_CELL_MIB"]),
    "parallelism_strategy": os.environ["PROICL_PARALLELISM_STRATEGY"],
    "vllm_gpu_memory_utilization": float(os.environ["PROICL_VLLM_GPU_MEMORY_UTILIZATION"]),
    "vllm_dtype": os.environ["PROICL_VLLM_DTYPE"],
    "vllm_attention_backend": os.environ.get("PROICL_VLLM_ATTENTION_BACKEND") or None,
    "vllm_prefix_caching": os.environ["PROICL_VLLM_PREFIX_CACHING"] == "1",
    "calibration_dtype": os.environ["PROICL_CALIBRATION_DTYPE"],
    "calibration_vllm_gpu_memory_utilization": float(
        os.environ["PROICL_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION"]
    ),
    "estimated_wall_clock_seconds_per_cell": int(
        os.environ["PROICL_ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"]
    ),
    "run_root": os.environ["PROICL_RUN_ROOT"],
    "run_tag": os.environ["PROICL_RUN_TAG"],
    "run_timestamp": os.environ.get("PROICL_RUN_TIMESTAMP") or None,
    "resume_source": os.environ.get("PROICL_RESUME_SOURCE") or "fresh",
    "progress_interval_seconds": float(os.environ.get("PROICL_CELL_HEARTBEAT_SECONDS", "60")),
    "cell_heartbeat_seconds": float(os.environ.get("PROICL_CELL_HEARTBEAT_SECONDS", "60")),
    "reflection_provider": os.environ["PROICL_REFLECTION_PROVIDER"],
    "tracks": os.environ["PROICL_TRACKS"].split(),
    "conditions": os.environ["PROICL_CONDITIONS"].split(),
    "eval_split": [
        int(os.environ["PROICL_EVAL_START"]),
        int(os.environ["PROICL_EVAL_END"]),
    ],
    "gepa_dev_split": [
        int(os.environ["PROICL_GEPA_DEV_START"]),
        int(os.environ["PROICL_GEPA_DEV_END"]),
    ],
    "rollout_budget": int(os.environ["PROICL_ROLLOUT_BUDGET"]),
    "max_new_tokens": int(os.environ["PROICL_MAX_NEW_TOKENS"]),
    "sps": {
        "block_num": int(os.environ["PROICL_SPS_BLOCK_NUM"]),
        "top_k": int(os.environ["PROICL_SPS_TOP_K"]),
        "candidate_pool_size": int(os.environ["PROICL_SPS_CANDIDATE_POOL_SIZE"]),
        "rollouts_per_candidate": int(os.environ["PROICL_SPS_ROLLOUTS_PER_CANDIDATE"]),
        "rollout_horizon": int(os.environ["PROICL_SPS_ROLLOUT_HORIZON"]),
        "chain_batch_size": int(os.environ["PROICL_SPS_CHAIN_BATCH_SIZE"]),
        "vllm_batch_size": int(os.environ["PROICL_SPS_VLLM_BATCH_SIZE"]),
    },
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

write_resource_probe() {
  local out="$1"
  "$PY" - "$out" <<'PY'
import json
import os
import sys

out = sys.argv[1]
slurm_keys = [
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_STEP_ID",
    "SLURM_STEP_GPUS",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_GPU",
    "SLURM_MEM_PER_CPU",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_NODELIST",
]
payload = {
    "schema": "proicl_resource_probe.v1",
    "gpu_detection_source": os.environ.get("PROICL_GPU_DETECTION_SOURCE"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_count": int(os.environ.get("PROICL_GPU_COUNT", "0")),
    "gpu_names": os.environ.get("PROICL_GPU_NAMES", "unknown"),
    "gpu_min_memory_mib": int(os.environ.get("PROICL_GPU_MIN_MEMORY_MIB", "0")),
    "gpu_min_compute_cap_x10": int(os.environ.get("PROICL_GPU_MIN_COMPUTE_CAP_X10", "0")),
    "host_memory_mib": int(os.environ.get("PROICL_HOST_MEMORY_MIB", "0")),
    "max_parallel_cells": int(os.environ.get("PROICL_MAX_PARALLEL_CELLS", "0")),
    "smoke_max_parallel_cells": int(os.environ.get("PROICL_SMOKE_MAX_PARALLEL_CELLS", "0")),
    "profile_max_parallel_cells": int(os.environ.get("PROICL_PROFILE_MAX_PARALLEL_CELLS", "0")),
    "host_memory_reserve_mib": int(os.environ.get("PROICL_HOST_MEMORY_RESERVE_MIB", "0")),
    "host_memory_per_cell_mib": int(os.environ.get("PROICL_HOST_MEMORY_PER_CELL_MIB", "0")),
    "vllm_dtype": os.environ.get("PROICL_VLLM_DTYPE"),
    "vllm_attention_backend": os.environ.get("PROICL_VLLM_ATTENTION_BACKEND"),
    "vllm_prefix_caching": os.environ.get("PROICL_VLLM_PREFIX_CACHING") == "1",
    "calibration_dtype": os.environ.get("PROICL_CALIBRATION_DTYPE"),
    "sps_vllm_batch_size": int(os.environ.get("PROICL_SPS_VLLM_BATCH_SIZE", "0")),
    "progress_interval_seconds": float(os.environ.get("PROICL_CELL_HEARTBEAT_SECONDS", "60")),
    "cell_heartbeat_seconds": float(os.environ.get("PROICL_CELL_HEARTBEAT_SECONDS", "60")),
    "slurm": {key: os.environ.get(key) for key in slurm_keys if os.environ.get(key) is not None},
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

print_launch_summary() {
  echo "ProICL launch profile:"
  echo "  gpu_profile=$GPU_PROFILE gpu_count=$GPU_COUNT cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all}"
  echo "  python=$PY"
  echo "  gpu_detection_source=$GPU_DETECTION_SOURCE"
  echo "  gpu_names=$GPU_NAMES_DETECTED"
  echo "  gpu_min_memory_mib=$GPU_MIN_MEMORY_MIB"
  echo "  gpu_min_compute_cap_x10=$GPU_MIN_COMPUTE_CAP_X10"
  echo "  host_memory_mib=$HOST_MEMORY_MIB_DETECTED"
  echo "  num_shards=$NUM_SHARDS strategy=$PARALLELISM_STRATEGY"
  echo "  max_parallel_cells=$MAX_PARALLEL_CELLS smoke_max_parallel_cells=$SMOKE_MAX_PARALLEL_CELLS overlap_gepa_and_cells=$OVERLAP_GEPA_AND_CELLS"
  echo "  profile_max_parallel_cells=$PROFILE_MAX_PARALLEL_CELLS host_memory_reserve_mib=$HOST_MEMORY_RESERVE_MIB host_memory_per_cell_mib=$HOST_MEMORY_PER_CELL_MIB"
  echo "  vllm_gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION calibration=$CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION"
  echo "  sps_chain_batch_size=$SPS_CHAIN_BATCH_SIZE"
  echo "  sps_vllm_batch_size=$SPS_VLLM_BATCH_SIZE"
  echo "  vllm_dtype=$VLLM_DTYPE vllm_attention_backend=${VLLM_ATTENTION_BACKEND:-auto} vllm_prefix_caching=$VLLM_PREFIX_CACHING"
  echo "  calibration_dtype=$CALIBRATION_DTYPE"
  echo "  resume_source=$RESUME_SOURCE run_timestamp=${RUN_TIMESTAMP:-new}"
  echo "  progress_interval_seconds=${PROICL_CELL_HEARTBEAT_SECONDS:-60}"
  echo "  reflection_provider=$REFLECTION_PROVIDER"
  echo "  estimated_wall_clock_seconds_per_cell=$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"
}

export PROICL_GPU_PROFILE="$GPU_PROFILE"
export PROICL_GPU_DETECTION_SOURCE="$GPU_DETECTION_SOURCE"
export PROICL_GPU_NAMES="$GPU_NAMES_DETECTED"
export PROICL_GPU_MIN_MEMORY_MIB="$GPU_MIN_MEMORY_MIB"
export PROICL_GPU_MIN_COMPUTE_CAP_X10="$GPU_MIN_COMPUTE_CAP_X10"
export PROICL_HOST_MEMORY_MIB="$HOST_MEMORY_MIB_DETECTED"
export PROICL_GPU_COUNT="$GPU_COUNT"
export PROICL_NUM_SHARDS="$NUM_SHARDS"
export PROICL_MAX_PARALLEL_CELLS="$MAX_PARALLEL_CELLS"
export PROICL_SMOKE_MAX_PARALLEL_CELLS="$SMOKE_MAX_PARALLEL_CELLS"
export PROICL_OVERLAP_GEPA_AND_CELLS="$OVERLAP_GEPA_AND_CELLS"
export PROICL_PROFILE_MAX_PARALLEL_CELLS="$PROFILE_MAX_PARALLEL_CELLS"
export PROICL_HOST_MEMORY_RESERVE_MIB="$HOST_MEMORY_RESERVE_MIB"
export PROICL_HOST_MEMORY_PER_CELL_MIB="$HOST_MEMORY_PER_CELL_MIB"
export PROICL_PARALLELISM_STRATEGY="$PARALLELISM_STRATEGY"
export PROICL_VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION"
export PROICL_VLLM_DTYPE="$VLLM_DTYPE"
export PROICL_VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND"
export PROICL_VLLM_PREFIX_CACHING="$VLLM_PREFIX_CACHING"
export PROICL_CALIBRATION_DTYPE="$CALIBRATION_DTYPE"
export PROICL_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="$CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION"
export PROICL_ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL="$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"
export PROICL_RUN_ROOT="$RUN_ROOT"
export PROICL_RUN_TAG="$RUN_TAG"
export PROICL_RUN_TIMESTAMP="$RUN_TIMESTAMP"
export PROICL_RESUME_SOURCE="$RESUME_SOURCE"
export PROICL_REFLECTION_PROVIDER="$REFLECTION_PROVIDER"
export PROICL_TRACKS="$TRACKS"
export PROICL_CONDITIONS="$CONDITIONS"
export PROICL_EVAL_START="$EVAL_START"
export PROICL_EVAL_END="$EVAL_END"
export PROICL_GEPA_DEV_START="$GEPA_DEV_START"
export PROICL_GEPA_DEV_END="$GEPA_DEV_END"
export PROICL_ROLLOUT_BUDGET="$ROLLOUT_BUDGET"
export PROICL_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export PROICL_SPS_BLOCK_NUM="$SPS_BLOCK_NUM"
export PROICL_SPS_TOP_K="$SPS_TOP_K"
export PROICL_SPS_CANDIDATE_POOL_SIZE="$SPS_CANDIDATE_POOL_SIZE"
export PROICL_SPS_ROLLOUTS_PER_CANDIDATE="$SPS_ROLLOUTS_PER_CANDIDATE"
export PROICL_SPS_ROLLOUT_HORIZON="$SPS_ROLLOUT_HORIZON"
export PROICL_SPS_CHAIN_BATCH_SIZE="$SPS_CHAIN_BATCH_SIZE"
export PROICL_SPS_VLLM_BATCH_SIZE="$SPS_VLLM_BATCH_SIZE"

print_launch_summary
if [[ "$DRY_RUN" != "1" ]]; then
  write_launch_config "$RUN_ROOT/launch_config.json"
  write_resource_probe "$RUN_ROOT/resource_probe.json"
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

section() {
  echo
  echo "==> $*"
}

if [[ "$DRY_RUN" != "1" && ! -f pyproject.toml ]]; then
  echo "Run this script from the ProICL repository root." >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" && "$SKIP_INSTALL" != "1" ]]; then
  section "Installing ProICL dependencies ($INSTALL_PROFILE profile)"
  if [[ "$PY" == "python" ]]; then
    NEED_CREATE_VENV=1
  fi
  if [[ "$NEED_CREATE_VENV" == "1" ]]; then
    run_cmd "$PY" -m venv "$VENV"
    PY="$VENV_PY"
  fi
  run_cmd "$PY" -m pip install -U pip wheel setuptools
  install_xformers_polyfill() {
    # xformers 0.0.29.post2 only ships manylinux_2_28 wheels; CentOS 7 has glibc 2.17.
    # Workaround: download the wheel, retag to manylinux_2_17, install, then
    # patch its .so files with polyfill-glibc so they resolve glibc symbols >=2.28.
    local xf_dir="$REPO_ROOT/.cache-xformers-wheel"
    local polyfill_bin="/share/software/user/open/polyfill-glibc/0.1/bin/polyfill-glibc"
    local site_pkgs
    site_pkgs="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    mkdir -p "$xf_dir"
    run_cmd "$PY" -m pip download --no-deps --dest "$xf_dir" \
      "xformers==0.0.29.post2" \
      --python-version 3.12 --platform manylinux_2_28_x86_64 --only-binary=:all:
    local src_whl="$xf_dir/xformers-0.0.29.post2-cp312-cp312-manylinux_2_28_x86_64.whl"
    local dst_whl="$xf_dir/xformers-0.0.29.post2-cp312-cp312-manylinux_2_17_x86_64.whl"
    if [[ -f "$src_whl" ]]; then
      mv -f "$src_whl" "$dst_whl"
    fi
    run_cmd "$PY" -m pip install --force-reinstall --no-deps "$dst_whl"
    if [[ -x "$polyfill_bin" ]]; then
      find "$site_pkgs/xformers" -name '*.so' -print0 \
        | xargs -0 -r -n1 "$polyfill_bin" --target-glibc=2.17 || true
    else
      echo "WARNING: polyfill-glibc not found at $polyfill_bin; xformers may fail at runtime on glibc<2.28" >&2
    fi
  }

  case "$INSTALL_PROFILE" in
    standard|light)
      run_cmd "$PY" -m pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
      install_xformers_polyfill
      run_cmd "$PY" -m pip install -r requirements.txt
      if [[ "$REFLECTION_PROVIDER" == "xai" ]]; then
        run_cmd "$PY" -m pip install -e ".[gepa_reflection]"
      fi
      ;;
    full)
      run_cmd "$PY" -m pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
      install_xformers_polyfill
      run_cmd "$PY" -m pip install -e ".[code,dc,gepa_reflection]"
      run_cmd "$PY" -m pip install "vllm==0.8.5.post1"
      ;;
    *)
      echo "INSTALL_PROFILE must be standard or full; got $INSTALL_PROFILE" >&2
      exit 1
      ;;
  esac
fi

if [[ "$DRY_RUN" != "1" && "$REFLECTION_PROVIDER" == "xai" ]]; then
  if ! "$PY" -c "import litellm" >/dev/null 2>&1; then
    echo "REFLECTION_PROVIDER=xai requires litellm. Run: $PY -m pip install -e '.[gepa_reflection]'" >&2
    exit 1
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  section "Initializing optional W&B tracking"
  run_cmd "$PY" scripts/wandb_run.py start \
    --run-root "$RUN_ROOT" \
    --config "$RUN_ROOT/launch_config.json" \
    --resource-probe "$RUN_ROOT/resource_probe.json" \
    --env-file "$REPO_ROOT/.env"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  section "Checking CUDA GPU host"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; this run needs a CUDA GPU host." >&2
    exit 1
  fi
fi

CALIB_DTYPE_ID="$(printf '%s' "$CALIBRATION_DTYPE" | tr -c '[:alnum:]_' '_')"
CALIB_DIR="$RUN_ROOT/calibration/deepseek_r1_distill_qwen_1p5b_vllm_${CALIB_DTYPE_ID}"
CALIB_ARTIFACT="${VLLM_PARITY_ARTIFACT:-$CALIB_DIR/calibration_summary.json}"
if [[ "$SKIP_CALIBRATION" == "1" && -z "${VLLM_PARITY_ARTIFACT:-}" ]]; then
  echo "SKIP_CALIBRATION=1 requires VLLM_PARITY_ARTIFACT=/path/to/calibration_summary.json" >&2
  exit 1
fi

calibration_artifact_passes() {
  local artifact="$1"
  [[ -f "$artifact" ]] || return 1
  "$PY" - "$artifact" <<'PY'
import sys
from pathlib import Path

from polaris.infra.vllm_calibration import (
    CalibrationArtifactError,
    validate_vllm_calibration_artifact,
)

try:
    validate_vllm_calibration_artifact(Path(sys.argv[1]))
except CalibrationArtifactError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
}

if [[ "$DRY_RUN" != "1" && -z "${VLLM_PARITY_ARTIFACT:-}" && -f "$CALIB_ARTIFACT" ]]; then
  if ! calibration_artifact_passes "$CALIB_ARTIFACT" >/dev/null 2>&1; then
    echo "Existing vLLM calibration artifact failed validation; rebuilding $CALIB_DIR" >&2
    rm -rf "$CALIB_DIR"
  fi
fi

if [[ -z "${VLLM_PARITY_ARTIFACT:-}" && ( "$DRY_RUN" == "1" || ! -f "$CALIB_ARTIFACT" ) ]]; then
  section "Running vLLM/HF calibration"
  CALIB_CMD=("$PY" scripts/vllm_hf_calibration.py \
    --model-key deepseek-r1-distill-qwen-1.5b \
    --out "$CALIB_DIR" \
    --temperature 0.25 \
    --segment-lens 1 2 8 32 128 \
    --hf-dtype "$CALIBRATION_DTYPE" \
    --vllm-dtype "$CALIBRATION_DTYPE" \
    --vllm-model-impl transformers \
    --vllm-gpu-memory-utilization "$CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-4096}")
  if [[ "$VLLM_PREFIX_CACHING" == "0" ]]; then
    CALIB_CMD+=(--no-vllm-prefix-caching)
  fi
  CALIB_ENV=(env)
  if [[ "$CALIBRATION_DTYPE" == "float32" ]]; then
    CALIB_ENV+=(-u VLLM_ATTENTION_BACKEND)
  fi
  _calibration_visible_default="${CUDA_VISIBLE_DEVICES:-}"
  _calibration_visible_default="${_calibration_visible_default%%,*}"
  CALIBRATION_CUDA_VISIBLE_DEVICES="${CALIBRATION_CUDA_VISIBLE_DEVICES:-$_calibration_visible_default}"
  if [[ -n "${CALIBRATION_CUDA_VISIBLE_DEVICES:-}" ]]; then
    CALIB_ENV+=("CUDA_VISIBLE_DEVICES=$CALIBRATION_CUDA_VISIBLE_DEVICES")
  fi
  run_cmd "${CALIB_ENV[@]}" "${CALIB_CMD[@]}"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if ! calibration_artifact_passes "$CALIB_ARTIFACT"; then
    echo "vLLM calibration artifact did not pass: $CALIB_ARTIFACT" >&2
    exit 1
  fi
fi

SPS_CALIB_DIR="$RUN_ROOT/calibration/sps_vs_mcmc_math500"
if [[ "$SKIP_SPS_MATH500_CALIBRATION" != "1" && "$SMOKE_ONLY" != "1" ]]; then
  section "Running SPS-vs-MCMC calibration"
  run_cmd "$PY" scripts/calibrate_sps.py \
    --out "$SPS_CALIB_DIR" \
    --model-key deepseek-r1-distill-qwen-1.5b \
    --split "$MATH_CALIB_START" "$MATH_CALIB_END" \
    --max-new-tokens "$MATH_CALIB_MAX_NEW_TOKENS" \
    --sps-block-size "$SPS_BLOCK_SIZE" \
    --sps-top-k "$SPS_TOP_K" \
    --sps-candidate-pool-size "$SPS_CANDIDATE_POOL_SIZE" \
    --sps-rollouts-per-candidate "$SPS_ROLLOUTS_PER_CANDIDATE" \
    --sps-rollout-horizon "$SPS_ROLLOUT_HORIZON" \
    --vllm-parity-artifact "$CALIB_ARTIFACT" \
    --estimated-wall-clock-seconds-per-cell "$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL" \
    --tolerance "$SPS_CALIBRATION_TOLERANCE"
  run_cmd "$PY" scripts/check_sps_calibration.py "$SPS_CALIB_DIR" \
    --tolerance "$SPS_CALIBRATION_TOLERANCE"
fi

BACKEND_PREFLIGHT_DIR="$RUN_ROOT/backend_preflight"
RUNTIME_PROFILE_PATH="$RUN_ROOT/runtime_profile.json"
if [[ "$DRY_RUN" != "1" && "$SKIP_BACKEND_PREFLIGHT" != "1" ]]; then
  section "Running production-shaped SPS/vLLM backend preflight"
  BACKEND_PREFLIGHT_CMD=(
    "$PY" scripts/backend_preflight.py
    --out-dir "$BACKEND_PREFLIGHT_DIR"
    --runtime-profile "$RUNTIME_PROFILE_PATH"
    --track reasoning_gym_boxnet
    --split "$EVAL_START" "$EVAL_END"
    --model-key deepseek-r1-distill-qwen-1.5b
    --vllm-parity-artifact "$CALIB_ARTIFACT"
    --vllm-dtype "$VLLM_DTYPE"
    --vllm-model-impl transformers
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-4096}"
    --vllm-prefix-caching "$VLLM_PREFIX_CACHING"
    --sps-vllm-batch-size "$SPS_VLLM_BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --sps-block-num "$SPS_BLOCK_NUM"
    --sps-top-k "$SPS_TOP_K"
    --sps-candidate-pool-size "$SPS_CANDIDATE_POOL_SIZE"
    --sps-rollouts-per-candidate "$SPS_ROLLOUTS_PER_CANDIDATE"
    --sps-rollout-horizon "$SPS_ROLLOUT_HORIZON"
    --gpu "${CUDA_VISIBLE_DEVICES%%,*}"
    --gpu-profile "$GPU_PROFILE"
    --gpu-names "$GPU_NAMES_DETECTED"
    --gpu-min-memory-mib "$GPU_MIN_MEMORY_MIB"
    --run-kind "$RUN_KIND"
    --estimated-wall-clock-seconds "$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"
    --allow-fallbacks
  )
  if [[ -n "$VLLM_ATTENTION_BACKEND" ]]; then
    BACKEND_PREFLIGHT_CMD+=(--vllm-attention-backend "$VLLM_ATTENTION_BACKEND")
  fi
  if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
    BACKEND_PREFLIGHT_CMD+=(--local-files-only)
  fi
  run_cmd "${BACKEND_PREFLIGHT_CMD[@]}"
  if [[ -f "$RUNTIME_PROFILE_PATH" ]]; then
    eval "$("$PY" - "$RUNTIME_PROFILE_PATH" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
profile = json.loads(open(path, encoding="utf-8").read())
values = {
    "VLLM_DTYPE": profile["vllm_dtype"],
    "VLLM_PREFIX_CACHING": "1" if profile["vllm_prefix_caching"] else "0",
    "PROICL_VLLM_DTYPE": profile["vllm_dtype"],
    "PROICL_VLLM_PREFIX_CACHING": "1" if profile["vllm_prefix_caching"] else "0",
    "PROICL_SPS_VLLM_BATCH_SIZE": str(profile["sps_vllm_batch_size"]),
    "SPS_VLLM_BATCH_SIZE": str(profile["sps_vllm_batch_size"]),
}
backend = profile.get("vllm_attention_backend")
if backend:
    values["VLLM_ATTENTION_BACKEND"] = str(backend)
    values["PROICL_VLLM_ATTENTION_BACKEND"] = str(backend)
else:
    print("unset VLLM_ATTENTION_BACKEND")
    values["PROICL_VLLM_ATTENTION_BACKEND"] = ""
for key, value in values.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
    VLLM_DTYPE="$PROICL_VLLM_DTYPE"
    VLLM_PREFIX_CACHING="$PROICL_VLLM_PREFIX_CACHING"
    VLLM_ATTENTION_BACKEND="$PROICL_VLLM_ATTENTION_BACKEND"
    SPS_VLLM_BATCH_SIZE="$PROICL_SPS_VLLM_BATCH_SIZE"
    export PROICL_SPS_VLLM_BATCH_SIZE="$SPS_VLLM_BATCH_SIZE"
    write_launch_config "$RUN_ROOT/launch_config.json"
    write_resource_probe "$RUN_ROOT/resource_probe.json"
  fi
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  if [[ "$SKIP_BACKEND_PREFLIGHT" == "1" ]]; then
    echo "PREFLIGHT_ONLY=1 requires RUN_BACKEND_PREFLIGHT=1." >&2
    exit 1
  fi
  exit 0
fi

read -r -a TRACK_ARGS <<<"$TRACKS"
read -r -a TRAIN_ARGS <<<"$ARCHIVE_TRAIN_TRACKS"
read -r -a HELDOUT_ARGS <<<"$ARCHIVE_HELDOUT_TRACKS"
read -r -a CONDITION_ARGS <<<"$CONDITIONS"

MAIN_CMD=(
  "$PY" scripts/run_proicl_signal.py
  --backend vllm
  --power-sampler sps
  --vllm-parity-artifact "$CALIB_ARTIFACT"
  --standard-run-root
  --root "$RUN_ROOT"
  --run-tag "$RUN_TAG"
  --run-kind "$RUN_KIND"
  --run-stage "$RUN_STAGE"
  --tracks "${TRACK_ARGS[@]}"
  --archive-scope cross_family_curriculum
  --archive-train-tracks "${TRAIN_ARGS[@]}"
  --archive-heldout-tracks "${HELDOUT_ARGS[@]}"
  --conditions "${CONDITION_ARGS[@]}"
  --eval-split "$EVAL_START" "$EVAL_END"
  --gepa-dev-split "$GEPA_DEV_START" "$GEPA_DEV_END"
  --rollout-budget "$ROLLOUT_BUDGET"
  --archive-size "$ARCHIVE_SIZE"
  --max-metric-calls "$MAX_METRIC_CALLS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --sps-block-num "$SPS_BLOCK_NUM"
  --sps-top-k "$SPS_TOP_K"
  --sps-candidate-pool-size "$SPS_CANDIDATE_POOL_SIZE"
  --sps-rollouts-per-candidate "$SPS_ROLLOUTS_PER_CANDIDATE"
  --sps-rollout-horizon "$SPS_ROLLOUT_HORIZON"
  --num-shards "$NUM_SHARDS"
  --max-parallel-cells "$MAX_PARALLEL_CELLS"
  --smoke-max-parallel-cells "$SMOKE_MAX_PARALLEL_CELLS"
  --memory-num-shards 1
  --estimated-wall-clock-seconds-per-cell "$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"
  --reflection-provider "$REFLECTION_PROVIDER"
  --reflection-model-id "$REFLECTION_MODEL_ID"
  --vllm-dtype "$VLLM_DTYPE"
)

if [[ "$SMOKE_ONLY" == "1" ]]; then
  MAIN_CMD+=(--smoke-only)
else
  MAIN_CMD+=(--skip-smoke)
fi
if [[ "${SKIP_PREFETCH:-0}" == "1" ]]; then
  MAIN_CMD+=(--skip-prefetch)
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  MAIN_CMD+=(--local-files-only)
fi
if [[ "$OVERLAP_GEPA_AND_CELLS" == "1" ]]; then
  MAIN_CMD+=(--overlap-gepa-and-cells)
fi
if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  MAIN_CMD+=(--vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi
if [[ "$VLLM_PREFIX_CACHING" == "0" ]]; then
  MAIN_CMD+=(--no-vllm-prefix-caching)
fi
if [[ -n "$RUN_TIMESTAMP" ]]; then
  MAIN_CMD+=(--run-timestamp "$RUN_TIMESTAMP")
fi
if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  MAIN_CMD+=(--vllm-max-model-len "$VLLM_MAX_MODEL_LEN")
fi

section "Running ProICL held-out experiment"
RUN_MARKER="$RUN_ROOT/.launch_marker"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_ROOT"
  : > "$RUN_MARKER"
fi
run_cmd "${MAIN_CMD[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

RUN_DIR="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'proicl_*' -newer "$RUN_MARKER" -print | sort | tail -1)"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'proicl_*' -print | sort | tail -1)"
fi
if [[ -z "$RUN_DIR" ]]; then
  echo "Could not identify standardized run directory under $RUN_ROOT" >&2
  exit 1
fi
cp "$RUN_ROOT/launch_config.json" "$RUN_DIR/launch_config.json"
cp "$RUN_ROOT/resource_probe.json" "$RUN_DIR/resource_probe.json"
if [[ -f "$RUN_ROOT/wandb.json" ]]; then
  cp "$RUN_ROOT/wandb.json" "$RUN_DIR/wandb.json"
fi

PACKAGE_ROOT="$RUN_DIR"
if [[ "$SMOKE_ONLY" == "1" ]]; then
  PACKAGE_ROOT="$RUN_DIR/smoke"
else
  run_cmd "$PY" scripts/audit_proicl_artifacts.py \
    --plan "$RUN_DIR/full/proicl_signal_plan.json" \
    --out-dir "$RUN_DIR/full/artifact_audit" \
    --require-passed
fi

PACKAGE_CMD=("$PY" scripts/package_results.py --run-root "$PACKAGE_ROOT")
if [[ "$INCLUDE_CANDIDATES" == "1" ]]; then
  PACKAGE_CMD+=(--include-candidates)
fi
run_cmd "${PACKAGE_CMD[@]}"

echo
echo "Run directory: $RUN_DIR"
BUNDLE_PATH="$PACKAGE_ROOT/results_bundle.tar.gz"
BUNDLE_ABS="$(cd "$(dirname "$BUNDLE_PATH")" && pwd)/$(basename "$BUNDLE_PATH")"
if [[ -f "$RUN_DIR/wandb.json" ]]; then
  run_cmd "$PY" scripts/wandb_run.py finish \
    --run-root "$RUN_DIR" \
    --bundle "$BUNDLE_ABS" \
    --summary "$RUN_DIR/full/analysis/aggregate_stdout.json" \
    --events "$RUN_DIR/full/events.jsonl" \
    --env-file "$REPO_ROOT/.env"
fi
echo "Result bundle: $BUNDLE_ABS"
