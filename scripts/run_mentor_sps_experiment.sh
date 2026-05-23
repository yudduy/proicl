#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the mentor-facing POLARIS SPS experiment.

Default run:
  bash scripts/run_mentor_sps_experiment.sh

Useful overrides:
  EVAL_END=30 ROLLOUT_BUDGET=2 bash scripts/run_mentor_sps_experiment.sh
  DRY_RUN=1 bash scripts/run_mentor_sps_experiment.sh

Environment knobs:
  RUN_ROOT                 Output series directory. Default: runs/mentor_sps
  RUN_TAG                  Short tag inside the standardized run id. Default: sps-recovery
  TRACKS                   Eval tracks. Default: boxnet acre game_of_life_halting graph_color_n12
  ARCHIVE_TRAIN_TRACKS     GEPA training tracks. Default: five in-distribution Reasoning Gym tasks
  ARCHIVE_HELDOUT_TRACKS   Held-out tracks for cross-family provenance. Default: TRACKS
  CONDITIONS               Default: base_greedy sps_only gepa_sps_fixed prorl_v2_greedy
  EVAL_START/EVAL_END      Held-out eval slice. Default: 20/70
  GEPA_DEV_START/END       GEPA dev slice. Default: 0/50
  ROLLOUT_BUDGET           Samples per non-greedy condition. Default: 8
  NUM_SHARDS               Per-condition shards. Default: detected GPU count, at least 1
  MAX_NEW_TOKENS           Generation cap. Default: 1024
  SPS_BLOCK_SIZE           Target SPS block size. Default: 192
  SPS_BLOCK_NUM            Override computed SPS block count.
  SPS_TOP_K                Candidate blocks kept per SPS step. Default: 8
  SPS_CANDIDATE_POOL_SIZE  Candidate blocks sampled per SPS step. Default: 8
  SPS_ROLLOUTS_PER_CANDIDATE Lookahead rollouts per candidate. Default: 8
  SPS_ROLLOUT_HORIZON      Lookahead horizon tokens. Default: 128
  GPU_PROFILE              auto, a100, h100, or generic. Default: auto
  VLLM_PARITY_ARTIFACT     Existing calibration_summary.json. If unset, calibration is run.
  COST_CAP_DOLLARS         Required for paid/cloud RUN_KIND values; default 0.0 for local/farmshare.
  ESTIMATED_DOLLAR_COST_PER_CELL Default 0.0 for local/farmshare; required for paid/cloud.
  SKIP_INSTALL=1           Reuse the current environment.
  SKIP_CALIBRATION=1       Require VLLM_PARITY_ARTIFACT instead of running calibration.
  SKIP_SPS_MATH500_CALIBRATION=1 Skip the SPS-vs-MCMC MATH500 gate.
  SMOKE_ONLY=1             Run only the harness smoke.
  INCLUDE_CANDIDATES=1     Include candidates.jsonl in the final bundle.
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
SKIP_CALIBRATION="${SKIP_CALIBRATION:-0}"
SKIP_SPS_MATH500_CALIBRATION="${SKIP_SPS_MATH500_CALIBRATION:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
INCLUDE_CANDIDATES="${INCLUDE_CANDIDATES:-0}"

RUN_ROOT="${RUN_ROOT:-runs/mentor_sps}"
RUN_TAG="${RUN_TAG:-sps-recovery}"
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

GPU_COUNT="$(detect_gpu_count)"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi
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
    ;;
  a100)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.85"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.55"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="7200"
    ;;
  generic)
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.80"
    DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="0.50"
    DEFAULT_WALL_CLOCK_SECONDS_PER_CELL="7200"
    ;;
  *)
    echo "GPU_PROFILE must be one of auto, a100, h100, generic; got $GPU_PROFILE" >&2
    exit 1
    ;;
esac

if [[ "$RUN_KIND" == "local" || "$RUN_KIND" == "farmshare" ]]; then
  COST_CAP_DOLLARS="${COST_CAP_DOLLARS:-0.0}"
  ESTIMATED_DOLLAR_COST_PER_CELL="${ESTIMATED_DOLLAR_COST_PER_CELL:-0.0}"
else
  if [[ -z "${COST_CAP_DOLLARS:-}" ]]; then
    echo "COST_CAP_DOLLARS is required for paid/cloud RUN_KIND=$RUN_KIND" >&2
    exit 1
  fi
  if [[ -z "${ESTIMATED_DOLLAR_COST_PER_CELL:-}" ]]; then
    echo "ESTIMATED_DOLLAR_COST_PER_CELL is required for paid/cloud RUN_KIND=$RUN_KIND" >&2
    exit 1
  fi
fi
ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL="${ESTIMATED_WALL_CLOCK_SECONDS_PER_CELL:-$DEFAULT_WALL_CLOCK_SECONDS_PER_CELL}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}"
CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="${CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION}"

CACHE_ROOT="${POLARIS_CACHE_ROOT:-$REPO_ROOT/.cache/polaris}"
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

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" != "1" && ! -f pyproject.toml ]]; then
  echo "Run this script from the POLARIS repository root." >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" && "$SKIP_INSTALL" != "1" ]]; then
  if [[ "$PY" == "python" ]]; then
    run_cmd python -m venv "$VENV"
    PY="$REPO_ROOT/$VENV/bin/python"
  fi
  run_cmd "$PY" -m pip install -U pip wheel setuptools
  run_cmd "$PY" -m pip install -e ".[code,dc,gepa_reflection]"
  run_cmd "$PY" -m pip install "vllm==0.9.2"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  run_cmd bash scripts/check_protocol_sync.sh
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
    --cost-cap-dollars "$COST_CAP_DOLLARS" \
    --estimated-dollar-cost-per-cell "$ESTIMATED_DOLLAR_COST_PER_CELL" \
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
  --cost-cap-dollars "$COST_CAP_DOLLARS"
  --estimated-dollar-cost-per-cell "$ESTIMATED_DOLLAR_COST_PER_CELL"
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

RUN_MARKER="$RUN_ROOT/.mentor_sps_launch_marker"
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

PACKAGE_ROOT="$RUN_DIR"
if [[ "$SMOKE_ONLY" == "1" ]]; then
  PACKAGE_ROOT="$RUN_DIR/smoke"
else
  run_cmd "$PY" scripts/audit_proicl_artifacts.py \
    --plan "$RUN_DIR/full/proicl_signal_plan.json" \
    --out-dir "$RUN_DIR/full/artifact_audit" \
    --require-passed
fi

PACKAGE_CMD=("$PY" scripts/package_sps_results.py --run-root "$PACKAGE_ROOT")
if [[ "$INCLUDE_CANDIDATES" == "1" ]]; then
  PACKAGE_CMD+=(--include-candidates)
fi
run_cmd "${PACKAGE_CMD[@]}"

echo
echo "Run directory: $RUN_DIR"
echo "Result bundle: $PACKAGE_ROOT/sps_results_bundle.tar.gz"
