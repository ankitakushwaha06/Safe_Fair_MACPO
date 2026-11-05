#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-sg/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
SUMMARY_SCRIPT="${ROOT_DIR}/scripts/internal/summarize_safefairbench_weak_env_repairs.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"
STATUS_FILE="${QUEUE_DIR}/highway_repair_v3_status.tsv"

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
export SAFE_FAIR_CFG_EVAL_EPISODES="${SAFE_FAIR_CFG_EVAL_EPISODES:-12}"
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

run_variant() {
  local tag="$1"
  local method="$2"
  local kwargs="$3"
  local budget_scale="$4"
  local margin="$5"
  local blend="$6"
  local speed="$7"
  local burden_recovery="$8"
  local burden_strength="$9"
  local task="safefairbench_highway_merge"
  local start_ts
  start_ts="$(date +%s)"
  echo "==== HIGHWAY V3 ${tag} / ${method} ===="
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

BASE_IMMEDIATE='{"collision_radius":0.108,"hazard_radius":0.120,"yield_distance":0.200,"lane_conflict_radius":0.200,"highway_merge_x":-1.20,"highway_route_strength":1.20,"highway_route_blend_floor":0.96,"highway_shoulder_y":0.280,"highway_stagger_goals":true,"highway_goal_x_spacing":0.32,"highway_project_lane_conflicts":true,"recovery_projection_clearance":0.10}'
FAIR_ASSIST='{"collision_radius":0.108,"hazard_radius":0.120,"yield_distance":0.200,"lane_conflict_radius":0.200,"highway_merge_x":-1.20,"highway_route_strength":1.20,"highway_route_blend_floor":0.96,"highway_shoulder_y":0.280,"highway_stagger_goals":true,"highway_goal_x_spacing":0.32,"highway_project_lane_conflicts":true,"highway_fair_merge_assist":true,"highway_fair_merge_strength":1.55,"highway_fair_merge_main_strength_scale":0.82,"recovery_projection_clearance":0.10}'

run_variant "highway_immediate_macpo_control_s4096" "macpo_recovery" "${BASE_IMMEDIATE}" "4.0" "0.10" "0.55" "0.90" "0" "0.45"
run_variant "highway_immediate_lag_control_s4096" "mappo_lagrangian_recovery" "${BASE_IMMEDIATE}" "4.0" "0.10" "0.55" "0.90" "0" "0.45"
run_variant "highway_immediate_sf_plain_s4096" "safe_fair_macpo" "${BASE_IMMEDIATE}" "4.0" "0.10" "0.55" "0.90" "1" "0.80"
run_variant "highway_fair_assist_sf_s4096" "safe_fair_macpo" "${FAIR_ASSIST}" "4.0" "0.10" "0.55" "0.90" "1" "0.80"

echo "==== HIGHWAY V3 QUEUE COMPLETE ===="
"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" || true
