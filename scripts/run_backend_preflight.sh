#!/usr/bin/env bash
set -euo pipefail

# Optional runtime check. The normal experiment runner does not call this.
# Usage: bash scripts/run_backend_preflight.sh [auto|l40|a100|h100|generic]

PROFILE="${1:-}"
if [[ -n "$PROFILE" ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_BACKEND_PREFLIGHT=1
export PREFLIGHT_ONLY=1

case "$PROFILE" in
  "")
    exec bash "$SCRIPT_DIR/run_experiment.sh" "$@"
    ;;
  auto|l40|a100|h100|generic)
    exec bash "$SCRIPT_DIR/run_experiment.sh" "$PROFILE" "$@"
    ;;
  *)
    echo "Usage: bash scripts/run_backend_preflight.sh [auto|l40|a100|h100|generic]" >&2
    exit 2
    ;;
esac
