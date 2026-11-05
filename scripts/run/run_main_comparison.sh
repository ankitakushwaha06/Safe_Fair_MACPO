#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SAFE_FAIR_PROMOTED_RUN_ID="${SAFE_FAIR_PROMOTED_RUN_ID:-paper_main_v1}"
export SAFE_FAIR_PROMOTED_CONFIRM_SEEDS="${SAFE_FAIR_PROMOTED_CONFIRM_SEEDS:-0 1 2 3 4}"
export SAFE_FAIR_PROMOTED_ABLATION_SEEDS="${SAFE_FAIR_PROMOTED_ABLATION_SEEDS:-0 1 2 3 4}"
exec "${ROOT_DIR}/scripts/internal/run_multiseed_benchmark.sh" "$@"
