#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-sg/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
SUMMARY_SCRIPT="${ROOT_DIR}/scripts/internal/summarize_safefairbench_weak_env_repairs.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"
STATUS_FILE="${QUEUE_DIR}/safefairbench_matched_recovery_controls_status.tsv"

mkdir -p "${QUEUE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

export SAFE_FAIR_MACPO_ROOT="${RUN_ROOT}"
export SAFE_FAIR_INCLUDE_SAFEFAIRBENCH=1
export SAFE_FAIR_QUICK_TEST=0
export SAFE_FAIR_RUN_TRAINING=1
export SAFE_FAIR_RUN_EVALUATION=1
export SAFE_FAIR_CLONE_OFFICIAL_REPOS=0
export SAFE_FAIR_USE_CUDA="${SAFE_FAIR_USE_CUDA:-0}"
export SAFE_FAIR_RUN_GROUP_FILTER="${SAFE_FAIR_RUN_GROUP_FILTER:-safefairbench}"
export SAFE_FAIR_SEED_FILTER="${SAFE_FAIR_SEED_FILTER:-0}"
export SAFE_FAIR_CFG_TOTAL_STEPS="${SAFE_FAIR_CFG_TOTAL_STEPS:-4096}"
export SAFE_FAIR_CFG_ROLLOUT_STEPS="${SAFE_FAIR_CFG_ROLLOUT_STEPS:-256}"
export SAFE_FAIR_CFG_PPO_EPOCHS="${SAFE_FAIR_CFG_PPO_EPOCHS:-6}"
export SAFE_FAIR_CFG_MINIBATCH_SIZE="${SAFE_FAIR_CFG_MINIBATCH_SIZE:-256}"
export SAFE_FAIR_CFG_EVAL_EPISODES="${SAFE_FAIR_CFG_EVAL_EPISODES:-8}"
export SAFE_FAIR_CFG_ENTROPY_COEF="${SAFE_FAIR_CFG_ENTROPY_COEF:-0.004}"

export CPATH="${ROOT_DIR}/vendor/apt/extracted/libosmesa6-dev/usr/include:${CPATH:-}"
export LIBRARY_PATH="${ROOT_DIR}/vendor/apt/extracted/libosmesa6/usr/lib/x86_64-linux-gnu:${ROOT_DIR}/vendor/apt/extracted/libosmesa6-dev/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ROOT_DIR}/vendor/apt/extracted/libosmesa6/usr/lib/x86_64-linux-gnu:${HOME}/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

printf 'tag\ttask\tmethod\tstatus\telapsed_s\tutc_time\n' > "${STATUS_FILE}"

log_status() {
  local tag="$1"
  local task="$2"
  local method="$3"
  local status="$4"
  local elapsed="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${task}" "${method}" "${status}" "${elapsed}" "$(date -u +%FT%TZ)" >> "${STATUS_FILE}"
}

run_control() {
  local tag="$1"
  local task="$2"
  local method="$3"
  local kwargs="$4"
  local budget_scale="$5"
  local margin="$6"
  local blend="$7"
  local speed="$8"
  local burden_recovery="$9"
  local burden_strength="${10}"
  local start_ts
  start_ts="$(date +%s)"
  echo "==== MATCHED REPAIR CONTROL ${tag} / ${task} / ${method} ===="
  log_status "${tag}" "${task}" "${method}" "started" "0"
  if SAFE_FAIR_TASK_FILTER="${task}" \
    SAFE_FAIR_METHOD_FILTER="${method}" \
    SAFE_FAIR_TASK_KWARGS_JSON="${kwargs}" \
    SAFE_FAIR_TASK_SAFETY_BUDGET_SCALE="${budget_scale}" \
    SAFE_FAIR_CFG_RUN_TAG="${tag}" \
    SAFE_FAIR_METHOD_SAFETY_RECOVERY_MARGIN="${margin}" \
    SAFE_FAIR_METHOD_SAFETY_RECOVERY_BLEND="${blend}" \
    SAFE_FAIR_METHOD_SAFETY_RECOVERY_SPEED="${speed}" \
    SAFE_FAIR_METHOD_USE_BURDEN_AWARE_RECOVERY="${burden_recovery}" \
    SAFE_FAIR_METHOD_BURDEN_RECOVERY_STRENGTH="${burden_strength}" \
    "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
    log_status "${tag}" "${task}" "${method}" "finished" "$(( $(date +%s) - start_ts ))"
  else
    log_status "${tag}" "${task}" "${method}" "failed" "$(( $(date +%s) - start_ts ))"
  fi
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" || true
}

# Exact recovery-only control for the strongest ChargeShare repair candidate.
run_control "cs_route_a_control_s4096" "safefairbench_vmas_chargeshare" "macpo_recovery" \
  '{"collision_radius":0.117,"hazard_radius":0.135,"yield_distance":0.216,"station_radius":0.216,"chargeshare_route_recovery":true,"chargeshare_lane_span":0.30,"chargeshare_route_strength":0.70,"chargeshare_route_blend_floor":0.70}' \
  "2.0" "0.08" "0.55" "1.00" "0" "0.45"

# Exact recovery-only control for the only Highway repair that avoided hard
# violations while modestly improving burden balance.
run_control "highway_early_a_control_s4096" "safefairbench_highway_merge" "macpo_recovery" \
  '{"collision_radius":0.108,"hazard_radius":0.120,"yield_distance":0.200,"lane_conflict_radius":0.200,"highway_merge_x":-0.45,"highway_route_strength":1.25,"highway_route_blend_floor":0.90}' \
  "4.0" "0.08" "0.55" "0.95" "0" "0.45"

echo "==== MATCHED REPAIR CONTROL QUEUE COMPLETE ===="
"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" || true
