#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/safe_fair_macpo_runs/logs/queue"
mkdir -p "${LOG_DIR}"

TOTAL_STEPS="${TOTAL_STEPS:-180000}"
RUN_SEED="${RUN_SEED:-0}"

declare -a JOBS=(
  "mafocops Ant-v2 2x4"
  "mafocops HalfCheetah-v2 2x3"
)

for job in "${JOBS[@]}"; do
  read -r algo scenario conf <<<"${job}"
  short="${scenario%%-*}"
  log_file="${LOG_DIR}/${algo}_${short}_${conf}_s${RUN_SEED}.log"
  echo "starting ${algo} ${scenario} ${conf} seed=${RUN_SEED} steps=${TOTAL_STEPS}"
  "${ROOT_DIR}/scripts/run_mafocops_native_audit.sh" \
    "${algo}" "${scenario}" "${conf}" "${TOTAL_STEPS}" "${RUN_SEED}" "native_${algo}_audit" \
    >"${log_file}" 2>&1
  echo "finished ${algo} ${scenario} ${conf}; log=${log_file}"
done
