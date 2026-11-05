#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SAFE_FAIR_500K_RUN_ID="${SAFE_FAIR_500K_RUN_ID:-paper_long_budget_v1}"
exec "${ROOT_DIR}/scripts/internal/run_500k_completion_matrix.sh" "$@"
