#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_case() {
  local name="$1"
  shift
  local output
  output="$(env -i PATH="$PATH" HOME="${HOME:-}" "$@" 2>&1)"
  printf '%s\n' "$output" | sed "s/^/[$name] /"
  echo "$output"
}

assert_contains() {
  local output="$1"
  local needle="$2"
  if ! grep -Fq "$needle" <<<"$output"; then
    echo "Expected output to contain: $needle" >&2
    exit 1
  fi
}

common_env=(
  DRY_RUN=1
  SKIP_INSTALL=1
  SKIP_CALIBRATION=0
  GPU_MEMORY_MIB=49140
  GPU_NAMES="NVIDIA L40S"
)

out="$(run_case l40-one \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=65536 \
  SLURM_STEP_GPUS=0 \
  bash "$ROOT/scripts/run_experiment_l40.sh")"
assert_contains "$out" "gpu_profile=l40 gpu_count=1"
assert_contains "$out" "gpu_detection_source=SLURM_STEP_GPUS"
assert_contains "$out" "num_shards=1"
assert_contains "$out" "max_parallel_cells=1 smoke_max_parallel_cells=1"

out="$(run_case l40-five-tight-host \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=65536 \
  SLURM_STEP_GPUS=0,1,2,3,4 \
  bash "$ROOT/scripts/run_experiment_l40.sh")"
assert_contains "$out" "gpu_profile=l40 gpu_count=5"
assert_contains "$out" "num_shards=5"
assert_contains "$out" "max_parallel_cells=1 smoke_max_parallel_cells=1"

out="$(run_case l40-two-roomy-host \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=163840 \
  SLURM_GPUS_ON_NODE=2 \
  bash "$ROOT/scripts/run_experiment_l40.sh")"
assert_contains "$out" "gpu_profile=l40 gpu_count=2"
assert_contains "$out" "gpu_detection_source=SLURM_GPUS_ON_NODE"
assert_contains "$out" "max_parallel_cells=2 smoke_max_parallel_cells=1"

out="$(run_case h100-four \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  HOST_MEMORY_MIB=262144 \
  GPU_MEMORY_MIB=81559 \
  GPU_NAMES="NVIDIA H100 80GB HBM3" \
  SLURM_STEP_GPUS=0,1,2,3 \
  bash "$ROOT/scripts/run_experiment_h100.sh")"
assert_contains "$out" "gpu_profile=h100 gpu_count=4"
assert_contains "$out" "num_shards=4"
assert_contains "$out" "max_parallel_cells=4 smoke_max_parallel_cells=1"
assert_contains "$out" "sps_chain_batch_size=2"

out="$(run_case override-gpus \
  "${common_env[@]}" \
  HOST_MEMORY_MIB=163840 \
  GPUS=2,4 \
  SLURM_STEP_GPUS=0,1,2,3,4 \
  bash "$ROOT/scripts/run_experiment_l40.sh")"
assert_contains "$out" "gpu_detection_source=GPUS"
assert_contains "$out" "gpu_profile=l40 gpu_count=2 cuda_visible_devices=2,4"

tmpbin="$(mktemp -d)"
trap 'rm -rf "$tmpbin"' EXIT
cat >"$tmpbin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=index --format=csv,noheader,nounits"*)
    printf "0\n1\n2\n"
    ;;
  *"--query-gpu=name --format=csv,noheader"*)
    printf "NVIDIA L40S\nNVIDIA L40S\nNVIDIA L40S\n"
    ;;
  *"--query-gpu=index,memory.total --format=csv,noheader,nounits"*)
    printf "0, 49140\n1, 49140\n2, 49140\n"
    ;;
  *"--query-gpu=index,name,memory.total --format=csv,noheader"*)
    printf "0, NVIDIA L40S, 49140 MiB\n1, NVIDIA L40S, 49140 MiB\n2, NVIDIA L40S, 49140 MiB\n"
    ;;
  -L)
    printf "GPU 0: NVIDIA L40S\nGPU 1: NVIDIA L40S\nGPU 2: NVIDIA L40S\n"
    ;;
  *)
    exit 1
    ;;
esac
SH
chmod +x "$tmpbin/nvidia-smi"
out="$(run_case nvidia-smi-fallback \
  DRY_RUN=1 \
  SKIP_INSTALL=1 \
  HOST_MEMORY_MIB=200000 \
  PATH="$tmpbin:$PATH" \
  bash "$ROOT/scripts/run_experiment_l40.sh")"
assert_contains "$out" "gpu_profile=l40 gpu_count=3"
assert_contains "$out" "gpu_detection_source=nvidia-smi"
assert_contains "$out" "max_parallel_cells=2 smoke_max_parallel_cells=1"

echo "launch profile tests passed"
