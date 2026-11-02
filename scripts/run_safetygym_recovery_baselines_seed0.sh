#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH="${MUJOCO210_BIN:-$HOME/.mujoco/mujoco210/bin}:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

TASKS=(
  safetygym_point_multigoal1_fair
  safetygym_chargeshare_fair
)

METHODS=(
  mappo
  macpo
  mappo_lagrangian
  dfrl_soto_ggf
  reward_shaped_mappo_fair
  reward_shaped_macpo_fair
)

COMMON_ENV=(
  SAFE_FAIR_QUICK_TEST=0
  SAFE_FAIR_RUN_TRAINING=1
  SAFE_FAIR_RUN_EVALUATION=1
  SAFE_FAIR_CLONE_OFFICIAL_REPOS=0
  SAFE_FAIR_USE_CUDA=1
  SAFE_FAIR_RUN_GROUP_FILTER=main
  SAFE_FAIR_SEED_FILTER=0
  SAFE_FAIR_CFG_TOTAL_STEPS=180000
  SAFE_FAIR_CFG_ROLLOUT_STEPS=2048
  SAFE_FAIR_CFG_PPO_EPOCHS=8
  SAFE_FAIR_CFG_MINIBATCH_SIZE=512
  SAFE_FAIR_CFG_EVAL_EPISODES=12
  SAFE_FAIR_CFG_RUN_TAG=repaired_seed0
)

for task in "${TASKS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "==== RUN ${task} ${method} repaired_seed0"
    env "${COMMON_ENV[@]}" \
      SAFE_FAIR_TASK_FILTER="${task}" \
      SAFE_FAIR_METHOD_FILTER="${method}" \
      .venv-sg/bin/python codes/safe_fair_macpo_experiments.py
  done
done
