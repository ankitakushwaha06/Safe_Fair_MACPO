#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SAFE_FAIR_CFG_RUN_TAG="${SAFE_FAIR_CFG_RUN_TAG:-paper_ablations_v1}"
export SAFE_FAIR_SEEDS="${SAFE_FAIR_SEEDS:-0 1 2 3 4}"
exec "${ROOT_DIR}/scripts/internal/run_safefairbench_ablations.sh" "$@"
