#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${COST_CAP_DOLLARS:-}" ]]; then
  echo "Set COST_CAP_DOLLARS before launching Modal, e.g. COST_CAP_DOLLARS=2.00 $0" >&2
  exit 1
fi

ESTIMATED_DOLLAR_COST="${ESTIMATED_DOLLAR_COST:-0.50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SPS_TOP_K="${SPS_TOP_K:-2}"
SPS_CANDIDATE_POOL_SIZE="${SPS_CANDIDATE_POOL_SIZE:-2}"
SPS_ROLLOUTS_PER_CANDIDATE="${SPS_ROLLOUTS_PER_CANDIDATE:-1}"
SPS_ROLLOUT_HORIZON="${SPS_ROLLOUT_HORIZON:-16}"
MODAL_GPU="${MODAL_GPU:-a100-80gb}"
LOG_ROOT="${LOG_ROOT:-runs/modal_smoke}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$MODAL_GPU}"

PYTHON_BIN="${PYTHON_BIN:-./.venv-eval/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

case "$MODAL_GPU" in
  a100|a100-80gb|A100|A100-80GB)
    MODAL_FUNCTION="smoke_experiment_one_problem_a100_80gb"
    ;;
  h100|H100)
    MODAL_FUNCTION="smoke_experiment_one_problem_h100"
    ;;
  *)
    echo "Unsupported MODAL_GPU=$MODAL_GPU; use a100-80gb or h100." >&2
    exit 1
    ;;
esac

LOG_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$LOG_DIR"
{
  echo "modal_gpu=$MODAL_GPU"
  echo "modal_function=$MODAL_FUNCTION"
  echo "estimated_dollar_cost=$ESTIMATED_DOLLAR_COST"
  echo "cost_cap_dollars=$COST_CAP_DOLLARS"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "sps_top_k=$SPS_TOP_K"
  echo "sps_candidate_pool_size=$SPS_CANDIDATE_POOL_SIZE"
  echo "sps_rollouts_per_candidate=$SPS_ROLLOUTS_PER_CANDIDATE"
  echo "sps_rollout_horizon=$SPS_ROLLOUT_HORIZON"
} > "$LOG_DIR/launch.env"

"$PYTHON_BIN" -m modal run "scripts/modal_vllm_app.py::$MODAL_FUNCTION" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --sps-top-k "$SPS_TOP_K" \
  --sps-candidate-pool-size "$SPS_CANDIDATE_POOL_SIZE" \
  --sps-rollouts-per-candidate "$SPS_ROLLOUTS_PER_CANDIDATE" \
  --sps-rollout-horizon "$SPS_ROLLOUT_HORIZON" \
  --estimated-dollar-cost "$ESTIMATED_DOLLAR_COST" \
  --cost-cap-dollars "$COST_CAP_DOLLARS" \
  --user-authorized-paid-run | tee "$LOG_DIR/modal.log"

echo "Modal smoke log: $LOG_DIR/modal.log"
