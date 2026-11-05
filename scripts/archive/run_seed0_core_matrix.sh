#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-sg/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

export SAFE_FAIR_MACPO_ROOT="${RUN_ROOT}"
export SAFE_FAIR_QUICK_TEST=0
export SAFE_FAIR_RUN_TRAINING=1
export SAFE_FAIR_CLONE_OFFICIAL_REPOS=0
export SAFE_FAIR_RUN_EVALUATION=1
export SAFE_FAIR_USE_CUDA=1
export SAFE_FAIR_RUN_GROUP_FILTER="main"
export SAFE_FAIR_SEED_FILTER="0"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

TASKS=(
  "smamujoco_ant_2x4_fair"
  "safetygym_point_multigoal1_fair"
  "safetygym_chargeshare_fair"
)

METHODS=(
  "mappo"
  "macpo"
  "mappo_lagrangian"
  "dfrl_soto_ggf"
  "reward_shaped_mappo_fair"
  "reward_shaped_macpo_fair"
  "safe_fair_macpo"
)

for task in "${TASKS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "==== RUN ${task} / ${method} / seed0 ===="
    if ! SAFE_FAIR_TASK_FILTER="${task}" \
      SAFE_FAIR_METHOD_FILTER="${method}" \
      "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
      echo "!!!! FAILED ${task} / ${method} / seed0"
    fi
  done
done
