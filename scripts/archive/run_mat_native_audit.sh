#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAT_ROOT="${ROOT_DIR}/safe_fair_macpo_runs/official_sources/mat"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

ALGO="${1:-mat}"
SCENARIO="${2:-HalfCheetah-v2}"
AGENT_CONF="${3:-2x3}"
TOTAL_STEPS="${4:-180000}"
RUN_SEED="${5:-0}"
EXP_NAME="${6:-native_mat_audit}"

if [[ "${ALGO}" != "mat" && "${ALGO}" != "mat_dec" ]]; then
  echo "Unsupported MAT algo: ${ALGO}" >&2
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
  cd "${MAT_ROOT}/mat/scripts"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python train/train_mujoco.py \
    --seed "${RUN_SEED}" \
    --env_name mujoco \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP_NAME}" \
    --scenario "${SCENARIO}" \
    --agent_conf "${AGENT_CONF}" \
    --agent_obsk 1 \
    --critic_lr 5e-5 \
    --lr 5e-5 \
    --entropy_coef 0.001 \
    --max_grad_norm 0.5 \
    --eval_episodes 8 \
    --n_training_threads 8 \
    --n_rollout_threads 2 \
    --num_mini_batch 2 \
    --episode_length 1000 \
    --eval_interval 5 \
    --log_interval 5 \
    --save_interval 5 \
    --num_env_steps "${TOTAL_STEPS}" \
    --ppo_epoch 5 \
    --clip_param 0.05 \
    --use_eval \
    --add_center_xy \
    --use_state_agent \
    --use_value_active_masks \
    --use_policy_active_masks
)
