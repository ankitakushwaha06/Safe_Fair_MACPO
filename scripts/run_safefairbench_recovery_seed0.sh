#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-sg/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
STATUS_SCRIPT="${ROOT_DIR}/scripts/summarize_safefairbench_status.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

export SAFE_FAIR_MACPO_ROOT="${RUN_ROOT}"
export SAFE_FAIR_INCLUDE_SAFEFAIRBENCH=1
export SAFE_FAIR_QUICK_TEST=0
export SAFE_FAIR_RUN_TRAINING=1
export SAFE_FAIR_CLONE_OFFICIAL_REPOS=0
export SAFE_FAIR_RUN_EVALUATION=1
export SAFE_FAIR_USE_CUDA="${SAFE_FAIR_USE_CUDA:-0}"
export SAFE_FAIR_RUN_GROUP_FILTER="safefairbench"
export SAFE_FAIR_CFG_TOTAL_STEPS="${SAFE_FAIR_CFG_TOTAL_STEPS:-2048}"
export SAFE_FAIR_CFG_ROLLOUT_STEPS="${SAFE_FAIR_CFG_ROLLOUT_STEPS:-256}"
export SAFE_FAIR_CFG_PPO_EPOCHS="${SAFE_FAIR_CFG_PPO_EPOCHS:-8}"
export SAFE_FAIR_CFG_MINIBATCH_SIZE="${SAFE_FAIR_CFG_MINIBATCH_SIZE:-512}"
export SAFE_FAIR_CFG_EVAL_EPISODES="${SAFE_FAIR_CFG_EVAL_EPISODES:-12}"
export SAFE_FAIR_CFG_RUN_TAG="${SAFE_FAIR_CFG_RUN_TAG:-safefairbench_repaired_s0}"

export CPATH="${ROOT_DIR}/vendor/apt/extracted/libosmesa6-dev/usr/include:${CPATH:-}"
export LIBRARY_PATH="${ROOT_DIR}/vendor/apt/extracted/libosmesa6/usr/lib/x86_64-linux-gnu:${ROOT_DIR}/vendor/apt/extracted/libosmesa6-dev/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ROOT_DIR}/vendor/apt/extracted/libosmesa6/usr/lib/x86_64-linux-gnu:${HOME}/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

read -r -a TASKS <<< "${SAFE_FAIR_REPAIRED_TASKS:-safefairbench_vmas_multigoal safefairbench_pogema_warehouse}"
if [[ "${SAFE_FAIR_INCLUDE_HIGHWAY_REPAIR:-0}" == "1" ]]; then
  TASKS+=("safefairbench_highway_merge")
fi

read -r -a METHODS <<< "${SAFE_FAIR_REPAIRED_METHODS:-macpo mappo_lagrangian safe_fair_macpo}"
if [[ "${SAFE_FAIR_INCLUDE_SECONDARY_CONTROLS:-0}" == "1" ]]; then
  METHODS+=("mappo" "dfrl_soto_ggf" "reward_shaped_macpo_fair")
fi

task_kwargs_json() {
  case "$1" in
    safefairbench_vmas_multigoal)
      printf '%s' '{"collision_radius":0.13,"hazard_radius":0.15,"yield_distance":0.24}'
      ;;
    safefairbench_highway_merge)
      printf '%s' '{"collision_radius":0.108,"hazard_radius":0.12,"yield_distance":0.20,"lane_conflict_radius":0.20}'
      ;;
    *)
      printf '%s' '{}'
      ;;
  esac
}

task_budget_scale() {
  case "$1" in
    safefairbench_vmas_multigoal|safefairbench_pogema_warehouse|safefairbench_highway_merge)
      printf '%s' '4.0'
      ;;
    *)
      printf '%s' '1.0'
      ;;
  esac
}

for task in "${TASKS[@]}"; do
  kwargs="$(task_kwargs_json "${task}")"
  budget_scale="$(task_budget_scale "${task}")"
  for method in "${METHODS[@]}"; do
    start_ts="$(date +%s)"
    echo "==== REPAIRED ${task} / ${method} / seed0 ===="
    if ! SAFE_FAIR_TASK_FILTER="${task}" \
      SAFE_FAIR_METHOD_FILTER="${method}" \
      SAFE_FAIR_SEED_FILTER="0" \
      SAFE_FAIR_TASK_KWARGS_JSON="${kwargs}" \
      SAFE_FAIR_TASK_SAFETY_BUDGET_SCALE="${budget_scale}" \
      "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
      echo "!!!! FAILED ${task} / ${method} / seed0"
    fi
    end_ts="$(date +%s)"
    echo "==== FINISHED ${task} / ${method} in $((end_ts - start_ts))s ===="
    "${PYTHON_BIN}" "${STATUS_SCRIPT}" --run-group safefairbench
  done
done
