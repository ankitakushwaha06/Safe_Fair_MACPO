#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SAFE_FAIR_WORKLOAD_RUN_ID="${SAFE_FAIR_WORKLOAD_RUN_ID:-paper_workload_v1}"
export SAFE_FAIR_WORKLOAD_WAIT_FOR_SESSION="${SAFE_FAIR_WORKLOAD_WAIT_FOR_SESSION:-}"
exec "${ROOT_DIR}/scripts/internal/run_workload_imbalance_study.sh" "$@"
