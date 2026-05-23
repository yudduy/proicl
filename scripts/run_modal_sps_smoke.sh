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

PYTHON_BIN="${PYTHON_BIN:-./.venv-eval/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m modal run scripts/modal_vllm_app.py::smoke_sps_recovery_one_problem_a100_80gb \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --sps-top-k "$SPS_TOP_K" \
  --sps-candidate-pool-size "$SPS_CANDIDATE_POOL_SIZE" \
  --sps-rollouts-per-candidate "$SPS_ROLLOUTS_PER_CANDIDATE" \
  --sps-rollout-horizon "$SPS_ROLLOUT_HORIZON" \
  --estimated-dollar-cost "$ESTIMATED_DOLLAR_COST" \
  --cost-cap-dollars "$COST_CAP_DOLLARS" \
  --user-authorized-paid-run
