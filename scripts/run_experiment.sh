#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the ProICL held-out experiment.

Default run:
  bash scripts/run_experiment.sh

Useful overrides:
  EVAL_END=30 ROLLOUT_BUDGET=2 bash scripts/run_experiment.sh
  DRY_RUN=1 bash scripts/run_experiment.sh

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
  GPUS                     Optional comma-separated GPU ids; sets CUDA_VISIBLE_DEVICES.
  GPU_MEMORY_MIB           Optional comma-separated GPU memory override for dry runs/tests.
  MAX_NEW_TOKENS           Generation cap. Default: 1024
  SPS_BLOCK_SIZE           Target SPS block size. Default: 192
  SPS_BLOCK_NUM            Override computed SPS block count.
  SPS_TOP_K                Candidate blocks kept per SPS step. Default: 8
  SPS_CANDIDATE_POOL_SIZE  Candidate blocks sampled per SPS step. Default: 8
  SPS_ROLLOUTS_PER_CANDIDATE Lookahead rollouts per candidate. Default: 8
  SPS_ROLLOUT_HORIZON      Lookahead horizon tokens. Default: 128
  SPS_CHAIN_BATCH_SIZE     Archive-prompt SPS chains per vLLM batch. Default: GPU-aware.
  GPU_PROFILE              auto, a100, h100, or generic. Default: auto
  VLLM_PARITY_ARTIFACT     Existing calibration_summary.json. If unset, calibration is run.
  SKIP_INSTALL=1           Reuse the current environment.
  INSTALL_PROFILE          standard or full. Default: standard.
  SKIP_CALIBRATION=1       Require VLLM_PARITY_ARTIFACT instead of running calibration.
  SKIP_SPS_MATH500_CALIBRATION=0 Run the slow SPS-vs-MCMC MATH500 gate. Default: skipped.
  SMOKE_ONLY=1             Run only the harness smoke.
  INCLUDE_CANDIDATES=1     Include candidates.jsonl in the final bundle.
  PROICL_DISABLE_TQDM=1    Use plain progress lines instead of tqdm.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
INSTALL_PROFILE="${INSTALL_PROFILE:-standard}"
SKIP_CALIBRATION="${SKIP_CALIBRATION:-0}"
SKIP_SPS_MATH500_CALIBRATION="${SKIP_SPS_MATH500_CALIBRATION:-1}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
INCLUDE_CANDIDATES="${INCLUDE_CANDIDATES:-0}"

if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

RUN_ROOT="${RUN_ROOT:-runs/experiment}"
RUN_TAG="${RUN_TAG:-heldout}"
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
PY="$REPO_ROOT/$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

detect_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES"
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

GPU_COUNT="$(detect_gpu_count)"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi
GPU_NAMES_DETECTED="$(detect_gpu_names)"
GPU_MIN_MEMORY_MIB="$(detect_min_gpu_memory_mib)"
NUM_SHARDS="${NUM_SHARDS:-$GPU_COUNT}"
GPU_PROFILE="${GPU_PROFILE:-auto}"
if [[ "$GPU_PROFILE" == "auto" ]]; then
  GPU_PROFILE="$(detect_gpu_profile)"
fi
case "$GPU_PROFILE" in
  h100)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.88"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.60"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="3600"
    DEFAULT_SPS_CHAIN_BATCH_SIZE="2"
    ;;
  a100)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.85"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.55"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="7200"
    DEFAULT_SPS_CHAIN_BATCH_SIZE="2"
    ;;
  generic)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.80"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.50"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="7200"
    DEFAULT_SPS_CHAIN_BATCH_SIZE="1"
    ;;
  *)
    echo "GPU_PROFILE must be one of auto, a100, h100, generic; got $GPU_PROFILE" >&2
    exit 1
    ;;
esac

if [[ "$GPU_MIN_MEMORY_MIB" -gt 0 && "$GPU_MIN_MEMORY_MIB" -lt 60000 ]]; then
  DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.80"
  DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.50"
  DEFAULT_SPS_CHAIN_BATCH_SIZE="1"
fi

ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL="${ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL:-$DEFAULT_WALL_CLOCK_SECONDS_PER_CELL}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}"
CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="${CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION}"
SPS_CHAIN_BATCH_SIZE="${SPS_CHAIN_BATCH_SIZE:-$DEFAULT_SPS_CHAIN_BATCH_SIZE}"
PARALLELISM_STRATEGY="one cell worker per visible GPU; GEPA/archive build reserves GPU 0 while direct baseline cells overlap on remaining GPUs"

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
    "gpu_names": os.environ.get("PROICL_GPU_NAMES", "unknown"),
    "gpu_min_memory_mib": int(os.environ.get("PROICL_GPU_MIN_MEMORY_MIB", "0")),
    "gpu_count": int(os.environ["PROICL_GPU_COUNT"]),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "num_shards": int(os.environ["PROICL_NUM_SHARDS"]),
    "parallelism_strategy": os.environ["PROICL_PARALLELISM_STRATEGY"],
    "vllm_gpu_memory_utilization": float(os.environ["PROICL_VLLM_GPU_MEMORY_UTILIZATION"]),
    "calibration_vllm_gpu_memory_utilization": float(
        os.environ["PROICL_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION"]
    ),
    "estimated_wall_clock_seconds_per_cell": int(
        os.environ["PROICL_ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"]
    ),
    "run_root": os.environ["PROICL_RUN_ROOT"],
    "run_tag": os.environ["PROICL_RUN_TAG"],
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
    },
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

print_launch_summary() {
  echo "ProICL launch profile:"
  echo "  gpu_profile=$GPU_PROFILE gpu_count=$GPU_COUNT cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all}"
  echo "  gpu_names=$GPU_NAMES_DETECTED"
  echo "  gpu_min_memory_mib=$GPU_MIN_MEMORY_MIB"
  echo "  num_shards=$NUM_SHARDS strategy=$PARALLELISM_STRATEGY"
  echo "  vllm_gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION calibration=$CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION"
  echo "  sps_chain_batch_size=$SPS_CHAIN_BATCH_SIZE"
  echo "  estimated_wall_clock_seconds_per_cell=$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"
}

export PROICL_GPU_PROFILE="$GPU_PROFILE"
export PROICL_GPU_NAMES="$GPU_NAMES_DETECTED"
export PROICL_GPU_MIN_MEMORY_MIB="$GPU_MIN_MEMORY_MIB"
export PROICL_GPU_COUNT="$GPU_COUNT"
export PROICL_NUM_SHARDS="$NUM_SHARDS"
export PROICL_PARALLELISM_STRATEGY="$PARALLELISM_STRATEGY"
export PROICL_VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION"
export PROICL_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="$CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION"
export PROICL_ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL="$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"
export PROICL_RUN_ROOT="$RUN_ROOT"
export PROICL_RUN_TAG="$RUN_TAG"
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

print_launch_summary
if [[ "$DRY_RUN" != "1" ]]; then
  write_launch_config "$RUN_ROOT/launch_config.json"
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
    run_cmd python -m venv "$VENV"
    PY="$REPO_ROOT/$VENV/bin/python"
  fi
  run_cmd "$PY" -m pip install -U pip wheel setuptools
  case "$INSTALL_PROFILE" in
    standard|light)
      run_cmd "$PY" -m pip install -r requirements.txt
      ;;
    full)
      run_cmd "$PY" -m pip install -e ".[code,dc,gepa_reflection]"
      run_cmd "$PY" -m pip install "vllm==0.9.2"
      ;;
    *)
      echo "INSTALL_PROFILE must be standard or full; got $INSTALL_PROFILE" >&2
      exit 1
      ;;
  esac
fi

if [[ "$DRY_RUN" != "1" ]]; then
  section "Checking CUDA GPU host"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; this run needs a CUDA GPU host." >&2
    exit 1
  fi
fi

CALIB_DIR="$RUN_ROOT/calibration/deepseek_r1_distill_qwen_1p5b_vllm"
CALIB_ARTIFACT="${VLLM_PARITY_ARTIFACT:-$CALIB_DIR/calibration_summary.json}"
if [[ "$SKIP_CALIBRATION" == "1" && -z "${VLLM_PARITY_ARTIFACT:-}" ]]; then
  echo "SKIP_CALIBRATION=1 requires VLLM_PARITY_ARTIFACT=/path/to/calibration_summary.json" >&2
  exit 1
fi

if [[ -z "${VLLM_PARITY_ARTIFACT:-}" && ! -f "$CALIB_ARTIFACT" ]]; then
  section "Running vLLM/HF calibration"
  run_cmd "$PY" scripts/vllm_hf_calibration.py \
    --model-key deepseek-r1-distill-qwen-1.5b \
    --out "$CALIB_DIR" \
    --temperature 0.25 \
    --segment-lens 1 2 8 32 128 \
    --hf-dtype float32 \
    --vllm-dtype float32 \
    --vllm-model-impl transformers \
    --vllm-gpu-memory-utilization "$CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-4096}"
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
  --memory-num-shards 1
  --estimated-wall-clock-seconds-per-cell "$ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL"
  --reflection-provider "$REFLECTION_PROVIDER"
  --reflection-model-id "$REFLECTION_MODEL_ID"
)

if [[ "$SMOKE_ONLY" == "1" ]]; then
  MAIN_CMD+=(--smoke-only)
fi
if [[ "${SKIP_PREFETCH:-0}" == "1" ]]; then
  MAIN_CMD+=(--skip-prefetch)
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  MAIN_CMD+=(--local-files-only)
fi
if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  MAIN_CMD+=(--vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
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
echo "Result bundle: $BUNDLE_ABS"
