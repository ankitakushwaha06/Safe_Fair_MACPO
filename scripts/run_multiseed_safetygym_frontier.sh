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
export SAFE_FAIR_CFG_TOTAL_STEPS=180000
export SAFE_FAIR_CFG_ROLLOUT_STEPS=2048
export SAFE_FAIR_CFG_PPO_EPOCHS=8
export SAFE_FAIR_CFG_MINIBATCH_SIZE=512
export SAFE_FAIR_CFG_EVAL_EPISODES=16
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

SEEDS=(0 1 2)
TASKS=(
  "safetygym_point_multigoal1_fair"
  "safetygym_chargeshare_fair"
)

run_main_method() {
  local task="$1"
  local method="$2"
  local seed="$3"
  local tag="$4"
  echo "==== REVIEW MAIN ${task} / ${method} / seed${seed} / ${tag} ===="
  SAFE_FAIR_RUN_GROUP_FILTER="main" \
  SAFE_FAIR_TASK_FILTER="${task}" \
  SAFE_FAIR_METHOD_FILTER="${method}" \
  SAFE_FAIR_SEED_FILTER="${seed}" \
  SAFE_FAIR_CFG_RUN_TAG="${tag}" \
  "${PYTHON_BIN}" "${BACKEND_SCRIPT}"
}

run_sf_variant() {
  local task="$1"
  local seed="$2"
  local tag="$3"
  local fair_lr="$4"
  local max_fair="$5"
  local warmup="$6"
  local budget_init="$7"
  local budget_ramp="$8"
  local margin="$9"
  local entropy="${10}"
  local safety_scale="${11}"

  echo "==== REVIEW SF ${task} / safe_fair_macpo / seed${seed} / ${tag} ===="
  SAFE_FAIR_RUN_GROUP_FILTER="main" \
  SAFE_FAIR_TASK_FILTER="${task}" \
  SAFE_FAIR_METHOD_FILTER="safe_fair_macpo" \
  SAFE_FAIR_SEED_FILTER="${seed}" \
  SAFE_FAIR_CFG_RUN_TAG="${tag}" \
  SAFE_FAIR_CFG_FAIRNESS_DUAL_LR="${fair_lr}" \
  SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS="${max_fair}" \
  SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES="${warmup}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT="${budget_init}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES="${budget_ramp}" \
  SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN="${margin}" \
  SAFE_FAIR_CFG_ENTROPY_COEF="${entropy}" \
  SAFE_FAIR_CFG_SAFETY_TARGET_SCALE="${safety_scale}" \
  "${PYTHON_BIN}" "${BACKEND_SCRIPT}"
}

run_ablation_variant() {
  local task="$1"
  local method="$2"
  local seed="$3"
  local tag="$4"
  local fair_lr="$5"
  local max_fair="$6"
  local warmup="$7"
  local budget_init="$8"
  local budget_ramp="$9"
  local margin="${10}"
  local entropy="${11}"
  local safety_scale="${12}"

  echo "==== REVIEW ABLATION ${task} / ${method} / seed${seed} / ${tag} ===="
  SAFE_FAIR_RUN_GROUP_FILTER="ablation" \
  SAFE_FAIR_TASK_FILTER="${task}" \
  SAFE_FAIR_METHOD_FILTER="${method}" \
  SAFE_FAIR_SEED_FILTER="${seed}" \
  SAFE_FAIR_CFG_RUN_TAG="${tag}" \
  SAFE_FAIR_CFG_FAIRNESS_DUAL_LR="${fair_lr}" \
  SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS="${max_fair}" \
  SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES="${warmup}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT="${budget_init}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES="${budget_ramp}" \
  SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN="${margin}" \
  SAFE_FAIR_CFG_ENTROPY_COEF="${entropy}" \
  SAFE_FAIR_CFG_SAFETY_TARGET_SCALE="${safety_scale}" \
  "${PYTHON_BIN}" "${BACKEND_SCRIPT}"
}

for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    short="mg"
    if [[ "${task}" == "safetygym_chargeshare_fair" ]]; then
      short="cs"
    fi

    # Reward-only / safety-only / fairness-only / reward-shaped comparators
    run_main_method "${task}" "mappo" "${seed}" "review_${short}_mappo_s${seed}"
    run_main_method "${task}" "macpo" "${seed}" "review_${short}_macpo_s${seed}"
    run_main_method "${task}" "mappo_lagrangian" "${seed}" "review_${short}_mappolag_s${seed}"
    run_main_method "${task}" "dfrl_soto_ggf" "${seed}" "review_${short}_dfrl_s${seed}"
    run_main_method "${task}" "reward_shaped_macpo_fair" "${seed}" "review_${short}_rsmacpo_s${seed}"

    if [[ "${task}" == "safetygym_point_multigoal1_fair" ]]; then
      # MultiGoal frontier centered on repair_audit.
      run_sf_variant "${task}" "${seed}" "review_mg_sf_strict_s${seed}" 0.030 6.0 12 2.0 30 0.010 0.0030 0.35
      run_sf_variant "${task}" "${seed}" "review_mg_sf_medium_s${seed}" 0.020 5.0 18 3.0 24 0.010 0.0030 0.35
      run_sf_variant "${task}" "${seed}" "review_mg_sf_loose_s${seed}"  0.012 3.0 24 5.0 35 0.005 0.0035 0.35

      run_ablation_variant "${task}" "safe_fair_no_debt" "${seed}" "review_mg_nodebt_s${seed}" 0.020 5.0 18 3.0 24 0.010 0.0030 0.35
      run_ablation_variant "${task}" "safe_fair_no_envy" "${seed}" "review_mg_noenvy_s${seed}" 0.020 5.0 18 3.0 24 0.010 0.0030 0.35
      run_ablation_variant "${task}" "safe_fair_no_recovery" "${seed}" "review_mg_norecovery_s${seed}" 0.020 5.0 18 3.0 24 0.010 0.0030 0.35
    else
      # ChargeShare frontier spanning stricter fairness vs higher-return settings.
      run_sf_variant "${task}" "${seed}" "review_cs_sf_strict_s${seed}" 0.008 1.0 24 8.0 70 0.002 0.0050 0.80
      run_sf_variant "${task}" "${seed}" "review_cs_sf_medium_s${seed}" 0.012 2.0 20 5.0 50 0.005 0.0040 0.50
      run_sf_variant "${task}" "${seed}" "review_cs_sf_loose_s${seed}"  0.020 5.0 18 3.0 24 0.010 0.0030 0.35

      run_ablation_variant "${task}" "safe_fair_no_debt" "${seed}" "review_cs_nodebt_s${seed}" 0.012 2.0 20 5.0 50 0.005 0.0040 0.50
      run_ablation_variant "${task}" "safe_fair_no_envy" "${seed}" "review_cs_noenvy_s${seed}" 0.012 2.0 20 5.0 50 0.005 0.0040 0.50
      run_ablation_variant "${task}" "safe_fair_no_recovery" "${seed}" "review_cs_norecovery_s${seed}" 0.012 2.0 20 5.0 50 0.005 0.0040 0.50
    fi
  done
done
