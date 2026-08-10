#!/usr/bin/env bash
# Pin or release GPU clocks for measurement stability, and record what was done.
#
# Why a state file: `nvidia-smi -lgc` sets a locked clock *range* that this
# driver exposes no --query-gpu field for. The obvious-looking
# `clocks.applications.graphics` reports the DEFAULT applications clock, which
# on an A40 equals the max (1740 MHz) whether or not a lock is active — so
# reading it would report "locked" for an unlocked card. Rather than infer the
# lock, we record it here and cross-check it behaviourally at preflight.
#
# Usage:
#   sudo ./scripts/lock_clocks.sh <gpu_index> lock [sm_mhz]
#   sudo ./scripts/lock_clocks.sh <gpu_index> unlock
#   ./scripts/lock_clocks.sh <gpu_index> status
set -euo pipefail

GPU_INDEX="${1:?usage: lock_clocks.sh <gpu_index> <lock|unlock|status> [sm_mhz]}"
ACTION="${2:?usage: lock_clocks.sh <gpu_index> <lock|unlock|status> [sm_mhz]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="${REPO_ROOT}/results/.clock_policy.json"

max_sm() { nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits -i "$GPU_INDEX"; }
max_mem() { nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits -i "$GPU_INDEX"; }
cur_sm() { nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i "$GPU_INDEX"; }
uuid() { nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU_INDEX"; }

case "$ACTION" in
  lock)
    SM="${3:-$(max_sm)}"
    echo "Locking GPU ${GPU_INDEX} SM clock to ${SM} MHz..."
    # Persistence mode keeps the driver resident so the lock survives having no
    # active CUDA context between runs.
    nvidia-smi -i "$GPU_INDEX" -pm 1 >/dev/null
    nvidia-smi -i "$GPU_INDEX" -lgc "${SM},${SM}"
    mkdir -p "$(dirname "$STATE_FILE")"
    cat > "$STATE_FILE" <<EOF
{
  "gpu_index": ${GPU_INDEX},
  "gpu_uuid": "$(uuid)",
  "locked": true,
  "sm_clock_mhz": ${SM},
  "mem_clock_mhz": $(max_mem),
  "locked_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "Recorded to ${STATE_FILE}"
    echo "Current SM clock: $(cur_sm) MHz"
    ;;

  unlock)
    echo "Releasing GPU ${GPU_INDEX} clock lock..."
    nvidia-smi -i "$GPU_INDEX" -rgc
    mkdir -p "$(dirname "$STATE_FILE")"
    cat > "$STATE_FILE" <<EOF
{
  "gpu_index": ${GPU_INDEX},
  "gpu_uuid": "$(uuid)",
  "locked": false,
  "sm_clock_mhz": null,
  "mem_clock_mhz": null,
  "locked_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "Current SM clock: $(cur_sm) MHz"
    ;;

  status)
    echo "GPU ${GPU_INDEX}: current SM $(cur_sm) MHz, max $(max_sm) MHz"
    if [[ -f "$STATE_FILE" ]]; then cat "$STATE_FILE"; else echo "no clock policy recorded"; fi
    ;;

  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
