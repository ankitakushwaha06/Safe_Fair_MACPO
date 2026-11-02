#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAFOCOPS_ROOT="${ROOT_DIR}/safe_fair_macpo_runs/official_sources/mafocops_supplemental"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/_external_baseline_env.sh"

ALGO="${1:-mafocops}"
SCENARIO="${2:-Ant-v2}"
AGENT_CONF="${3:-2x4}"
TOTAL_STEPS="${4:-180000}"
RUN_SEED="${5:-0}"
EXP_NAME="${6:-native_mafocops_audit}"

if [[ "${ALGO}" != "mafocops" ]]; then
  echo "Unsupported algo: ${ALGO}" >&2
  exit 2
fi

if [[ "${SCENARIO}" != "HalfCheetah-v2" && "${SCENARIO}" != "Ant-v2" ]]; then
  echo "Unsupported scenario: ${SCENARIO}" >&2
  exit 2
fi

if [[ "${SCENARIO}" == "HalfCheetah-v2" && "${AGENT_CONF}" != "2x3" ]]; then
  echo "HalfCheetah expects 2x3 for our paper-aligned split." >&2
  exit 2
fi

if [[ "${SCENARIO}" == "Ant-v2" && "${AGENT_CONF}" != "2x4" ]]; then
  echo "Ant expects 2x4 for our paper-aligned split." >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/safe_fair_macpo_runs/logs/external"

(
  cd "${MAFOCOPS_ROOT}/macpo/scripts"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python train/train_mujoco.py \
    --env_name mujoco \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP_NAME}" \
    --scenario "${SCENARIO}" \
    --agent_conf "${AGENT_CONF}" \
    --agent_obsk 1 \
    --lr 9e-5 \
    --critic_lr 5e-3 \
    --std_x_coef 1 \
    --std_y_coef 5e-1 \
    --seed "${RUN_SEED}" \
    --n_training_threads 8 \
    --n_rollout_threads 4 \
    --n_eval_rollout_threads 1 \
    --num_mini_batch 8 \
    --episode_length 1000 \
    --num_env_steps "${TOTAL_STEPS}" \
    --ppo_epoch 1 \
    --use_value_active_masks \
    --use_eval \
    --eval_episodes 8 \
    --save_interval 1 \
    --log_interval 1 \
    --eval_interval 1 \
    --add_center_xy \
    --use_state_agent \
    --kl_threshold 0.0125 \
    --focops_eta 0.025 \
    --safety_bound 50 \
    --ls_step 5 \
    --entropy_coef 0 \
    --focops_lam 2.2 \
    --nu_max 1.3 \
    --nu_lr 5e-5 \
    --batch_size 256
)
