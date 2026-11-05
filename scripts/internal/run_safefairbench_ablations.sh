#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-sg/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
STATUS_SCRIPT="${ROOT_DIR}/scripts/internal/summarize_safefairbench_status.py"
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
export SAFE_FAIR_USE_CUDA="${SAFE_FAIR_USE_CUDA:-1}"
export SAFE_FAIR_RUN_GROUP_FILTER="safefairbench_ablation"
export SAFE_FAIR_CFG_TOTAL_STEPS="${SAFE_FAIR_CFG_TOTAL_STEPS:-180000}"
export SAFE_FAIR_CFG_ROLLOUT_STEPS="${SAFE_FAIR_CFG_ROLLOUT_STEPS:-2048}"
export SAFE_FAIR_CFG_PPO_EPOCHS="${SAFE_FAIR_CFG_PPO_EPOCHS:-8}"
export SAFE_FAIR_CFG_MINIBATCH_SIZE="${SAFE_FAIR_CFG_MINIBATCH_SIZE:-512}"
export SAFE_FAIR_CFG_EVAL_EPISODES="${SAFE_FAIR_CFG_EVAL_EPISODES:-12}"
export SAFE_FAIR_CFG_RUN_TAG="${SAFE_FAIR_CFG_RUN_TAG:-safefairbench_ablation}"

TASKS=(
  "safefairbench_vmas_multigoal"
  "safefairbench_vmas_chargeshare"
  "safefairbench_pogema_warehouse"
  "safefairbench_highway_merge"
)

METHODS=(
  "safe_fair_macpo"
  "safe_fair_no_debt"
  "safe_fair_no_hazard_burden"
  "safe_fair_no_contested_access_burden"
  "safe_fair_no_yield_burden"
  "safe_fair_jain_only"
  "safe_fair_envy_only"
  "safe_fair_scalar_critic"
  "safe_fair_per_agent_critic"
)

read -r -a SEEDS <<< "${SAFE_FAIR_SEEDS:-0 1 2 3 4}"

for task in "${TASKS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      start_ts="$(date +%s)"
      echo "==== ABLATION ${task} / ${method} / seed${seed} ===="
      if ! SAFE_FAIR_TASK_FILTER="${task}" SAFE_FAIR_METHOD_FILTER="${method}" SAFE_FAIR_SEED_FILTER="${seed}" "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
        echo "!!!! FAILED ${task} / ${method} / seed${seed}"
      fi
      end_ts="$(date +%s)"
      echo "==== FINISHED ${task} / ${method} / seed${seed} in $((end_ts - start_ts))s ===="
      "${PYTHON_BIN}" "${STATUS_SCRIPT}" --run-group safefairbench_ablation
    done
  done
done
