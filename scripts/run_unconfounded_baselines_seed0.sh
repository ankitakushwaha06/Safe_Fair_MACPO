#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH="${MUJOCO210_BIN:-$HOME/.mujoco/mujoco210/bin}:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

TASKS=(
  smamujoco_halfcheetah_2x3_fair
  smamujoco_ant_2x4_fair
  safetygym_point_multigoal1_fair
  safetygym_chargeshare_fair
)

METHODS=(
  mappo
  macpo
  mappo_lagrangian
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
  SAFE_FAIR_CFG_RUN_TAG=clean_baseline_nofairaux_s0
)

for task in "${TASKS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "==== RUN ${task} ${method} clean_baseline_nofairaux_s0"
    env "${COMMON_ENV[@]}" \
      SAFE_FAIR_TASK_FILTER="${task}" \
      SAFE_FAIR_METHOD_FILTER="${method}" \
      .venv-sg/bin/python codes/safe_fair_macpo_experiments.py
  done
done

.venv-sg/bin/python scripts/regenerate_result_tables.py

LOG_DIR="$ROOT/safe_fair_macpo_runs/logs/queue"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extended_baseline_after_unconfounded.log"
echo "[$(date -u +%FT%TZ)] clean baseline queue finished; starting extended baseline queue" >> "$LOG_FILE"
nohup bash "$ROOT/scripts/run_long_baselines_seed0.sh" >> "$LOG_FILE" 2>&1 &
echo $! > "$LOG_DIR/extended_baseline_after_unconfounded.pid"
echo "[$(date -u +%FT%TZ)] extended baseline queue PID $(cat "$LOG_DIR/extended_baseline_after_unconfounded.pid")" >> "$LOG_FILE"
