#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/_external_baseline_env.sh"

PYTHON_BIN="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

export SAFE_FAIR_MACPO_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
export SAFE_FAIR_QUICK_TEST=0
export SAFE_FAIR_RUN_TRAINING=1
export SAFE_FAIR_RUN_EVALUATION=1
export SAFE_FAIR_CLONE_OFFICIAL_REPOS=0
export SAFE_FAIR_USE_CUDA=1
export SAFE_FAIR_RUN_GROUP_FILTER="main"
export SAFE_FAIR_SEED_FILTER="0"
export SAFE_FAIR_CFG_TOTAL_STEPS=180000
export SAFE_FAIR_CFG_ROLLOUT_STEPS=2048
export SAFE_FAIR_CFG_PPO_EPOCHS=8
export SAFE_FAIR_CFG_MINIBATCH_SIZE=512
export SAFE_FAIR_CFG_EVAL_EPISODES=12
export SAFE_FAIR_CFG_RUN_TAG="core_benchmark_180k"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
START_AFTER_TASK="${START_AFTER_TASK:-}"
START_AFTER_METHOD="${START_AFTER_METHOD:-}"

TASKS=(
  "smamujoco_halfcheetah_2x3_fair"
  "smamujoco_ant_2x4_fair"
  "safetygym_point_multigoal1_fair"
  "safetygym_chargeshare_fair"
)

METHODS=(
  "ippo"
  "mappo"
  "macpo"
  "mappo_lagrangian"
  "dfrl_soto_ggf"
  "reward_shaped_mappo_fair"
  "reward_shaped_macpo_fair"
  "safe_fair_macpo"
  "safe_fair_no_debt"
  "safe_fair_no_envy"
  "safe_fair_no_recovery"
)

mkdir -p "${ROOT_DIR}/safe_fair_macpo_runs/logs/queue"

skip_until_marker=0
if [[ -n "${START_AFTER_TASK}" && -n "${START_AFTER_METHOD}" ]]; then
  skip_until_marker=1
fi

for task in "${TASKS[@]}"; do
  for method in "${METHODS[@]}"; do
    if [[ "${skip_until_marker}" -eq 1 ]]; then
      if [[ "${task}" == "${START_AFTER_TASK}" && "${method}" == "${START_AFTER_METHOD}" ]]; then
        skip_until_marker=0
      fi
      continue
    fi
    echo "==== RUN ${task} / ${method} / seed0 / core_benchmark_180k ===="
    if ! SAFE_FAIR_TASK_FILTER="${task}" \
      SAFE_FAIR_METHOD_FILTER="${method}" \
      "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
      echo "!!!! FAILED ${task} / ${method} / seed0 / core_benchmark_180k"
    fi
  done
done
