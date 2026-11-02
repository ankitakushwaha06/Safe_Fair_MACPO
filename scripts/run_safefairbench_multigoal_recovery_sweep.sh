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
export SAFE_FAIR_USE_CUDA="${SAFE_FAIR_USE_CUDA:-1}"
export SAFE_FAIR_RUN_GROUP_FILTER="safefairbench"
export SAFE_FAIR_TASK_FILTER="safefairbench_vmas_multigoal"
export SAFE_FAIR_METHOD_FILTER="safe_fair_macpo"
export SAFE_FAIR_SEED_FILTER="0"
export SAFE_FAIR_TASK_KWARGS_JSON='{"collision_radius":0.13,"hazard_radius":0.15,"yield_distance":0.24}'
export SAFE_FAIR_TASK_SAFETY_BUDGET_SCALE="4.0"
export SAFE_FAIR_CFG_TOTAL_STEPS="${SAFE_FAIR_CFG_TOTAL_STEPS:-2048}"
export SAFE_FAIR_CFG_ROLLOUT_STEPS="${SAFE_FAIR_CFG_ROLLOUT_STEPS:-256}"
export SAFE_FAIR_CFG_PPO_EPOCHS="${SAFE_FAIR_CFG_PPO_EPOCHS:-8}"
export SAFE_FAIR_CFG_MINIBATCH_SIZE="${SAFE_FAIR_CFG_MINIBATCH_SIZE:-512}"
export SAFE_FAIR_CFG_EVAL_EPISODES="${SAFE_FAIR_CFG_EVAL_EPISODES:-12}"

read -r -a TAGS <<< "${SAFE_FAIR_RECOVERY_TAGS:-mg_burden_router_base}"

margin_for_tag() {
  case "$1" in
    mg_margin004) printf '%s' '0.04' ;;
    mg_margin006) printf '%s' '0.06' ;;
    mg_margin006_blend075) printf '%s' '0.06' ;;
    mg_burden_router_base) printf '%s' '0.08' ;;
  esac
}

blend_for_tag() {
  case "$1" in
    mg_margin004) printf '%s' '0.85' ;;
    mg_margin006) printf '%s' '0.85' ;;
    mg_margin006_blend075) printf '%s' '0.75' ;;
    mg_burden_router_base) printf '%s' '0.85' ;;
  esac
}

speed_for_tag() {
  case "$1" in
    mg_margin004) printf '%s' '0.95' ;;
    mg_margin006) printf '%s' '0.95' ;;
    mg_margin006_blend075) printf '%s' '0.95' ;;
    mg_burden_router_base) printf '%s' '0.95' ;;
  esac
}

burden_recovery_for_tag() {
  case "$1" in
    mg_burden_router_base) printf '%s' '1' ;;
    *) printf '%s' '0' ;;
  esac
}

burden_strength_for_tag() {
  case "$1" in
    mg_burden_router_base) printf '%s' '0.45' ;;
    *) printf '%s' '0.45' ;;
  esac
}

for tag in "${TAGS[@]}"; do
  margin="$(margin_for_tag "${tag}")"
  blend="$(blend_for_tag "${tag}")"
  speed="$(speed_for_tag "${tag}")"
  burden_recovery="$(burden_recovery_for_tag "${tag}")"
  burden_strength="$(burden_strength_for_tag "${tag}")"
  if [[ -z "${margin}" || -z "${blend}" || -z "${speed}" ]]; then
    echo "Unknown recovery tag: ${tag}" >&2
    continue
  fi
  run_tag="${tag}_s${SAFE_FAIR_CFG_TOTAL_STEPS}_r${SAFE_FAIR_CFG_ROLLOUT_STEPS}"
  start_ts="$(date +%s)"
  echo "==== MULTIGOAL RECOVERY ${tag} / margin=${margin} blend=${blend} speed=${speed} burden_recovery=${burden_recovery} burden_strength=${burden_strength} / seed0 ===="
  if ! SAFE_FAIR_CFG_RUN_TAG="${run_tag}" \
    SAFE_FAIR_METHOD_SAFETY_RECOVERY_MARGIN="${margin}" \
    SAFE_FAIR_METHOD_SAFETY_RECOVERY_BLEND="${blend}" \
    SAFE_FAIR_METHOD_SAFETY_RECOVERY_SPEED="${speed}" \
    SAFE_FAIR_METHOD_USE_BURDEN_AWARE_RECOVERY="${burden_recovery}" \
    SAFE_FAIR_METHOD_BURDEN_RECOVERY_STRENGTH="${burden_strength}" \
    "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
    echo "!!!! FAILED ${tag} / seed0"
  fi
  end_ts="$(date +%s)"
  echo "==== FINISHED ${tag} in $((end_ts - start_ts))s ===="
  "${PYTHON_BIN}" "${STATUS_SCRIPT}" --run-group safefairbench
done
