#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
read -r -a SEEDS <<< "${SAFE_FAIR_SEEDS:-0 1 2 3 4}"
for seed in "${SEEDS[@]}"; do
  SAFE_FAIR_SEED_FILTER="${seed}" \
    "${ROOT_DIR}/scripts/internal/run_safefairbench_matched_recovery_controls.sh"
done
