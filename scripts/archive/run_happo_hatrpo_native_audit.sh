#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HAPPO_ROOT="${ROOT_DIR}/safe_fair_macpo_runs/official_sources/happo_hatrpo"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

ALGO="${1:-happo}"
SCENARIO="${2:-HalfCheetah-v2}"
AGENT_CONF="${3:-2x3}"
TOTAL_STEPS="${4:-180000}"
RUN_SEED="${5:-0}"
EXP_NAME="${6:-native_audit}"

if [[ "${ALGO}" != "happo" && "${ALGO}" != "hatrpo" ]]; then
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
  cd "${HAPPO_ROOT}/scripts"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python train/train_mujoco.py \
    --env_name mujoco \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP_NAME}" \
    --scenario "${SCENARIO}" \
    --agent_conf "${AGENT_CONF}" \
    --agent_obsk 2 \
    --lr 5e-6 \
    --critic_lr 5e-3 \
    --std_x_coef 1 \
    --std_y_coef 5e-1 \
    --seed_specify \
    --runing_id "${RUN_SEED}" \
    --n_training_threads 8 \
    --n_rollout_threads 2 \
    --n_eval_rollout_threads 1 \
    --num_mini_batch 2 \
    --episode_length 1000 \
    --num_env_steps "${TOTAL_STEPS}" \
    --ppo_epoch 5 \
    --kl_threshold 1e-4 \
    --use_value_active_masks \
    --use_eval \
    --eval_episodes 8 \
    --save_interval 1 \
    --log_interval 1 \
    --eval_interval 1 \
    --add_center_xy \
    --use_state_agent \
    --share_policy
)
