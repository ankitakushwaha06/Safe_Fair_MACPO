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
export SAFE_FAIR_METHOD_FILTER="safe_fair_macpo"
export SAFE_FAIR_TASK_FILTER="smamujoco_halfcheetah_2x3_fair"
export SAFE_FAIR_SEED_FILTER="0"
export SAFE_FAIR_CFG_TOTAL_STEPS=180000
export SAFE_FAIR_CFG_ROLLOUT_STEPS=2048
export SAFE_FAIR_CFG_PPO_EPOCHS=8
export SAFE_FAIR_CFG_MINIBATCH_SIZE=512
export SAFE_FAIR_CFG_EVAL_EPISODES=16
export SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=0.80
export SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY=0.10
export SAFE_FAIR_CFG_SAFETY_DUAL_LR=0.08
export SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY=10.0
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

run_variant() {
  local tag="$1"
  local fair_lr="$2"
  local max_fair="$3"
  local warmup="$4"
  local budget_init="$5"
  local budget_ramp="$6"
  local margin="$7"
  local init_lf="$8"
  local entropy="$9"
  local safety_scale="${10}"

  echo "==== PARETO smamujoco_halfcheetah_2x3_fair / safe_fair_macpo / seed0 / ${tag} ===="
  SAFE_FAIR_CFG_RUN_TAG="${tag}" \
  SAFE_FAIR_CFG_FAIRNESS_DUAL_LR="${fair_lr}" \
  SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS="${max_fair}" \
  SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES="${warmup}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT="${budget_init}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES="${budget_ramp}" \
  SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN="${margin}" \
  SAFE_FAIR_CFG_INIT_LAMBDA_FAIRNESS="${init_lf}" \
  SAFE_FAIR_CFG_ENTROPY_COEF="${entropy}" \
  SAFE_FAIR_CFG_SAFETY_TARGET_SCALE="${safety_scale}" \
  "${PYTHON_BIN}" "${BACKEND_SCRIPT}"
}

# Local search around hc_sf_sweet_d:
# - slightly stricter fairness target than sweet_d
# - mild nonzero initial fairness pressure
# - lower entropy on some runs to reduce late fairness drift
# - one stricter safety-target variant to keep the solution in the safe region
run_variant "hc_sf_pareto_a" 0.016 1.50 10 3.5 45 0.0010 0.01 0.0050 0.80
run_variant "hc_sf_pareto_b" 0.018 1.75 10 3.0 50 0.0010 0.02 0.0040 0.80
run_variant "hc_sf_pareto_c" 0.020 2.00 12 3.0 60 0.0015 0.02 0.0040 0.78
run_variant "hc_sf_pareto_d" 0.015 1.25 8  2.5 45 0.0010 0.03 0.0035 0.80
run_variant "hc_sf_pareto_e" 0.017 1.50 6  3.5 35 0.0005 0.01 0.0035 0.78
run_variant "hc_sf_pareto_f" 0.014 1.10 8  2.0 55 0.0015 0.03 0.0030 0.75
