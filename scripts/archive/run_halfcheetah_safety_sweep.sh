#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source .venv-sg/bin/activate

run_variant() {
  local tag="$1"
  local safety_target_scale="$2"
  local init_lambda_safety="$3"
  local safety_dual_lr="$4"
  local fairness_warmup_updates="$5"
  local fairness_safety_margin="$6"
  local entropy_coef="$7"

  echo "[sweep] starting ${tag}"
  SAFE_FAIR_QUICK_TEST=0 \
  SAFE_FAIR_RUN_TRAINING=1 \
  SAFE_FAIR_RUN_EVALUATION=1 \
  SAFE_FAIR_USE_CUDA=1 \
  SAFE_FAIR_TASK_FILTER=smamujoco_halfcheetah_2x3_fair \
  SAFE_FAIR_METHOD_FILTER=safe_fair_macpo \
  SAFE_FAIR_MAX_RUNS=1 \
  SAFE_FAIR_CFG_TOTAL_STEPS=150000 \
  SAFE_FAIR_CFG_EVAL_EPISODES=6 \
  SAFE_FAIR_CFG_RUN_TAG="$tag" \
  SAFE_FAIR_CFG_SAFETY_TARGET_SCALE="$safety_target_scale" \
  SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY="$init_lambda_safety" \
  SAFE_FAIR_CFG_SAFETY_DUAL_LR="$safety_dual_lr" \
  SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES="$fairness_warmup_updates" \
  SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN="$fairness_safety_margin" \
  SAFE_FAIR_CFG_ENTROPY_COEF="$entropy_coef" \
  python codes/safe_fair_macpo_experiments.py
}

run_named_variant() {
  case "$1" in
    sretune_a) run_variant sretune_a 0.70 0.25 0.20 4 0.0005 0.005 ;;
    sretune_b) run_variant sretune_b 0.55 0.50 0.30 8 0.0010 0.001 ;;
    sretune_c) run_variant sretune_c 0.60 0.35 0.25 6 0.0010 0.0005 ;;
    *) echo "unknown variant: $1" >&2; exit 1 ;;
  esac
}

if [ "$#" -eq 0 ]; then
  run_named_variant sretune_a
  run_named_variant sretune_b
  run_named_variant sretune_c
else
  for variant in "$@"; do
    run_named_variant "$variant"
  done
fi
