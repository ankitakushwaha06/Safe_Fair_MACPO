#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

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
export SAFE_FAIR_SEED_FILTER="0"
export SAFE_FAIR_CFG_TOTAL_STEPS=180000
export SAFE_FAIR_CFG_ROLLOUT_STEPS=2048
export SAFE_FAIR_CFG_PPO_EPOCHS=8
export SAFE_FAIR_CFG_MINIBATCH_SIZE=512
export SAFE_FAIR_CFG_EVAL_EPISODES=12
export SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=0.80
export SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY=0.10
export SAFE_FAIR_CFG_SAFETY_DUAL_LR=0.08
export SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY=10.0
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

run_variant() {
  local task="$1"
  local tag="$2"
  local fair_lr="$3"
  local max_fair="$4"
  local warmup="$5"
  local budget_init="$6"
  local budget_ramp="$7"
  local margin="$8"

  echo "==== TUNE ${task} / safe_fair_macpo / seed0 / ${tag} ===="
  SAFE_FAIR_TASK_FILTER="${task}" \
  SAFE_FAIR_CFG_RUN_TAG="${tag}" \
  SAFE_FAIR_CFG_FAIRNESS_DUAL_LR="${fair_lr}" \
  SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS="${max_fair}" \
  SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES="${warmup}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT="${budget_init}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES="${budget_ramp}" \
  SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN="${margin}" \
  "${PYTHON_BIN}" "${BACKEND_SCRIPT}"
}

# HalfCheetah: previous Safe-Fair was too fairness-dominated. These variants
# delay/soften fairness pressure while keeping strict safety pressure.
run_variant "smamujoco_halfcheetah_2x3_fair" "hc_sf_sweet_a" 0.003 0.50 30 16.0 80 0.002
run_variant "smamujoco_halfcheetah_2x3_fair" "hc_sf_sweet_b" 0.006 0.80 24 10.0 70 0.002
run_variant "smamujoco_halfcheetah_2x3_fair" "hc_sf_sweet_c" 0.010 1.20 16 6.0 55 0.001
run_variant "smamujoco_halfcheetah_2x3_fair" "hc_sf_sweet_d" 0.015 1.50 10 4.0 40 0.001

# ChargeShare: return is acceptable but fairness cost is too high. These variants
# increase fairness pressure gradually to search for a lower-cost Pareto point.
run_variant "safetygym_chargeshare_fair" "cs_sf_sweet_a" 0.008 1.00 24 8.0 70 0.002
run_variant "safetygym_chargeshare_fair" "cs_sf_sweet_b" 0.015 1.50 16 5.0 55 0.001
run_variant "safetygym_chargeshare_fair" "cs_sf_sweet_c" 0.025 2.50 10 3.0 45 0.001
run_variant "safetygym_chargeshare_fair" "cs_sf_sweet_d" 0.040 3.50 6 2.0 35 0.000
