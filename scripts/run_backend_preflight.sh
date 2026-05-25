#!/usr/bin/env bash
set -euo pipefail

# Optional runtime check. The normal experiment runners do not call this.
# Usage: bash scripts/run_backend_preflight.sh [l40|a100|h100]

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
  l40|a100|h100)
    exec bash "$SCRIPT_DIR/run_experiment_${PROFILE}.sh" "$@"
    ;;
  *)
    echo "Usage: bash scripts/run_backend_preflight.sh [l40|a100|h100]" >&2
    exit 2
    ;;
esac
