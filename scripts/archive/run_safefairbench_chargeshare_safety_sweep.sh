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
export SAFE_FAIR_RUN_GROUP_FILTER="safefairbench"
export SAFE_FAIR_TASK_FILTER="safefairbench_vmas_chargeshare"
export SAFE_FAIR_SEED_FILTER="0"
export SAFE_FAIR_TASK_KWARGS_JSON='{"collision_radius":0.117,"hazard_radius":0.135,"yield_distance":0.216,"station_radius":0.216}'
export SAFE_FAIR_TASK_SAFETY_BUDGET_SCALE="2.0"
export SAFE_FAIR_CFG_TOTAL_STEPS="${SAFE_FAIR_CFG_TOTAL_STEPS:-8192}"
export SAFE_FAIR_CFG_ROLLOUT_STEPS="${SAFE_FAIR_CFG_ROLLOUT_STEPS:-1024}"
export SAFE_FAIR_CFG_PPO_EPOCHS="${SAFE_FAIR_CFG_PPO_EPOCHS:-4}"
export SAFE_FAIR_CFG_MINIBATCH_SIZE="${SAFE_FAIR_CFG_MINIBATCH_SIZE:-256}"
export SAFE_FAIR_CFG_EVAL_EPISODES="${SAFE_FAIR_CFG_EVAL_EPISODES:-6}"
export SAFE_FAIR_CFG_ENTROPY_COEF="${SAFE_FAIR_CFG_ENTROPY_COEF:-0.003}"

read -r -a METHODS <<< "${SAFE_FAIR_METHODS:-macpo safe_fair_macpo}"
read -r -a TAGS <<< "${SAFE_FAIR_SAFETY_TAGS:-cs_safety_a cs_safety_b cs_safety_c}"

init_lambda_for_tag() {
  case "$1" in
    cs_safety_a) printf '%s' '0.10' ;;
    cs_safety_b) printf '%s' '0.25' ;;
    cs_safety_c) printf '%s' '0.50' ;;
  esac
}

dual_lr_for_tag() {
  case "$1" in
    cs_safety_a) printf '%s' '0.10' ;;
    cs_safety_b) printf '%s' '0.20' ;;
    cs_safety_c) printf '%s' '0.30' ;;
  esac
}

target_scale_for_tag() {
  case "$1" in
    cs_safety_a) printf '%s' '0.75' ;;
    cs_safety_b) printf '%s' '0.60' ;;
    cs_safety_c) printf '%s' '0.50' ;;
  esac
}

for tag in "${TAGS[@]}"; do
  for method in "${METHODS[@]}"; do
    start_ts="$(date +%s)"
    echo "==== CHARGESHARE SAFETY ${tag} / ${method} / seed0 ===="
    if ! SAFE_FAIR_METHOD_FILTER="${method}" \
      SAFE_FAIR_CFG_RUN_TAG="${tag}" \
      SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY="$(init_lambda_for_tag "${tag}")" \
      SAFE_FAIR_CFG_SAFETY_DUAL_LR="$(dual_lr_for_tag "${tag}")" \
      SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY="40.0" \
      SAFE_FAIR_CFG_SAFETY_TARGET_SCALE="$(target_scale_for_tag "${tag}")" \
      "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
      echo "!!!! FAILED ${tag} / ${method} / seed0"
    fi
    end_ts="$(date +%s)"
    echo "==== FINISHED ${tag} / ${method} in $((end_ts - start_ts))s ===="
    "${PYTHON_BIN}" "${STATUS_SCRIPT}" --run-group safefairbench
  done
done
