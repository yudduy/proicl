#!/usr/bin/env bash
set -euo pipefail

# H100 profile. GPU count is auto-detected; worker concurrency is still capped
# by host RAM so a multi-GPU allocation does not imply one vLLM process per GPU.
export GPU_PROFILE="${GPU_PROFILE:-h100}"
export REFLECTION_PROVIDER="${REFLECTION_PROVIDER:-local-hf}"
export SPS_CHAIN_BATCH_SIZE="${SPS_CHAIN_BATCH_SIZE:-2}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
export CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION="${CALIBRATION_VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
export PROFILE_MAX_PARALLEL_CELLS="${PROFILE_MAX_PARALLEL_CELLS:-6}"
export HOST_MEMORY_RESERVE_MIB="${HOST_MEMORY_RESERVE_MIB:-32768}"
export HOST_MEMORY_PER_CELL_MIB="${HOST_MEMORY_PER_CELL_MIB:-40960}"
export SMOKE_MAX_PARALLEL_CELLS="${SMOKE_MAX_PARALLEL_CELLS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_experiment.sh" "$@"
