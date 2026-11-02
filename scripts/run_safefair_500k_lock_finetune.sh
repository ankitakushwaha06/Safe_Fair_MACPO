#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/_external_baseline_env.sh"

set +e
set -uo pipefail

PYTHON_BIN="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
SUMMARY_SCRIPT="${ROOT_DIR}/scripts/summarize_safefair_500k_fairness_tuning.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"
RUN_ID="${SAFE_FAIR_SF500K_LOCK_RUN_ID:-safefair_500k_lock_finetune_$(date -u +%Y%m%d_%H%M%S)}"
MAX_SECONDS="${SAFE_FAIR_SF500K_LOCK_MAX_SECONDS:-32400}"
MIN_START_SECONDS="${SAFE_FAIR_SF500K_LOCK_MIN_START_SECONDS:-1200}"

mkdir -p "${QUEUE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

START_TS="$(date +%s)"
STATUS_FILE="${QUEUE_DIR}/${RUN_ID}_status.tsv"

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'phase\ttask\tmethod\tseed\tsteps\tvariant\ttag\trollout_steps\tppo_epochs\tminibatch_size\tsafety_scale\tinit_lambda_safety\tsafety_lr\tmax_lambda_safety\tfair_budget\tfair_lr\tmax_lambda_fairness\tinit_lambda_fairness\tburden_recovery\tstatus\telapsed_s\tutc_time\n' > "${STATUS_FILE}"
fi

elapsed() {
  echo $(( $(date +%s) - START_TS ))
}

time_left() {
  echo $(( MAX_SECONDS - $(elapsed) ))
}

can_start_next() {
  [[ "$(time_left)" -gt "${MIN_START_SECONDS}" ]]
}

log_status() {
  local phase="$1"
  local task="$2"
  local seed="$3"
  local steps="$4"
  local variant="$5"
  local tag="$6"
  local rollout_steps="$7"
  local ppo_epochs="$8"
  local minibatch_size="$9"
  local safety_scale="${10}"
  local init_ls="${11}"
  local safety_lr="${12}"
  local max_ls="${13}"
  local fair_budget="${14}"
  local fair_lr="${15}"
  local max_lf="${16}"
  local init_lf="${17}"
  local burden_recovery="${18}"
  local status="${19}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${phase}" "${task}" "safe_fair_macpo" "${seed}" "${steps}" "${variant}" "${tag}" \
    "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" \
    "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" \
    "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" \
    "${status}" "$(elapsed)" "$(date -u +%FT%TZ)" >> "${STATUS_FILE}"
}

run_tag_finished() {
  local tag="$1"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${tag}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
tag = sys.argv[2]
for cfg_path in (root / "logs").rglob("config.json"):
    if not (cfg_path.parent / "eval_summary.json").exists():
        continue
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if cfg.get("run_tag") == tag:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

COMMON_ENV=(
  "SAFE_FAIR_MACPO_ROOT=${RUN_ROOT}"
  "SAFE_FAIR_QUICK_TEST=0"
  "SAFE_FAIR_RUN_TRAINING=1"
  "SAFE_FAIR_RUN_EVALUATION=1"
  "SAFE_FAIR_CLONE_OFFICIAL_REPOS=0"
  "SAFE_FAIR_USE_CUDA=1"
  "SAFE_FAIR_INCLUDE_SAFEFAIRBENCH=0"
  "SAFE_FAIR_RUN_GROUP_FILTER=main"
  "SAFE_FAIR_METHOD_FILTER=safe_fair_macpo"
  "SAFE_FAIR_SEED_FILTER=0"
  "SAFE_FAIR_CFG_TOTAL_STEPS=500000"
  "SAFE_FAIR_CFG_EVAL_EPISODES=16"
  "MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210"
)

run_variant() {
  local task="$1"
  local variant="$2"
  local rollout_steps="$3"
  local ppo_epochs="$4"
  local minibatch_size="$5"
  local safety_scale="$6"
  local init_ls="$7"
  local safety_lr="$8"
  local max_ls="$9"
  local fair_lr="${10}"
  local max_lf="${11}"
  local warmup="${12}"
  local budget_init="${13}"
  local budget_ramp="${14}"
  local margin="${15}"
  local init_lf="${16}"
  local entropy="${17}"
  local fair_budget="${18}"
  local burden_recovery="${19}"
  local burden_strength="${20}"
  local recovery_margin="${21}"
  local recovery_blend="${22}"
  local recovery_speed="${23}"
  local tag="${RUN_ID}_${variant}_s0"

  if ! can_start_next; then
    echo "[$(date -u +%FT%TZ)] stopping before ${variant}; time_left=$(time_left)s < min_start=${MIN_START_SECONDS}s"
    log_status "lock_finetune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "paused_before_start"
    return 99
  fi

  if run_tag_finished "${tag}"; then
    echo "==== SKIP ${task} / safe_fair_macpo / seed0 / 500k / ${variant}; eval summary already exists ===="
    log_status "lock_finetune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "skipped"
    return 0
  fi

  echo "==== START ${task} / safe_fair_macpo / seed0 / 500k / ${variant} / tag=${tag} ===="
  log_status "lock_finetune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "started"
  local start
  start="$(date +%s)"

  env "${COMMON_ENV[@]}" \
    "SAFE_FAIR_TASK_FILTER=${task}" \
    "SAFE_FAIR_CFG_RUN_TAG=${tag}" \
    "SAFE_FAIR_CFG_ROLLOUT_STEPS=${rollout_steps}" \
    "SAFE_FAIR_CFG_PPO_EPOCHS=${ppo_epochs}" \
    "SAFE_FAIR_CFG_MINIBATCH_SIZE=${minibatch_size}" \
    "SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=${safety_scale}" \
    "SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY=${init_ls}" \
    "SAFE_FAIR_CFG_SAFETY_DUAL_LR=${safety_lr}" \
    "SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY=${max_ls}" \
    "SAFE_FAIR_CFG_FAIRNESS_DUAL_LR=${fair_lr}" \
    "SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS=${max_lf}" \
    "SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES=${warmup}" \
    "SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT=${budget_init}" \
    "SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES=${budget_ramp}" \
    "SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN=${margin}" \
    "SAFE_FAIR_CFG_INIT_LAMBDA_FAIRNESS=${init_lf}" \
    "SAFE_FAIR_CFG_ENTROPY_COEF=${entropy}" \
    "SAFE_FAIR_TASK_FAIRNESS_BUDGET_PER_STEP=${fair_budget}" \
    "SAFE_FAIR_METHOD_USE_BURDEN_AWARE_RECOVERY=${burden_recovery}" \
    "SAFE_FAIR_METHOD_BURDEN_RECOVERY_STRENGTH=${burden_strength}" \
    "SAFE_FAIR_METHOD_SAFETY_RECOVERY_MARGIN=${recovery_margin}" \
    "SAFE_FAIR_METHOD_SAFETY_RECOVERY_BLEND=${recovery_blend}" \
    "SAFE_FAIR_METHOD_SAFETY_RECOVERY_SPEED=${recovery_speed}" \
    "${PYTHON_BIN}" "${BACKEND_SCRIPT}"
  local rc=$?
  local duration=$(( $(date +%s) - start ))
  if [[ "${rc}" -eq 0 ]]; then
    echo "==== FINISH ${task} / ${variant} / duration=${duration}s ===="
    log_status "lock_finetune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "finished"
  else
    echo "==== FAIL ${task} / ${variant} / rc=${rc} / duration=${duration}s ===="
    log_status "lock_finetune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "failed"
  fi
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-id "${RUN_ID}" || true
  return "${rc}"
}

# Targets:
# - HalfCheetah: exact scalar safety cost 0.00, hard 0.00, return above the
#   low-return fullsafe row if possible.
# - Ant-2x4: exact safety and fairness closer to IPPO's 1.28 while keeping
#   return comparable or better than IPPO's 1843.96.
QUEUE=(
  "smamujoco_halfcheetah_2x3_fair hc_lock_exact_a 4096 5 1024 0.45 0.60 0.30 25.0 0.006 1.0 32 6.0 120 0.0040 0.00 0.0005 0.050 0 0.45 0.16 0.95 0.99"
  "smamujoco_halfcheetah_2x3_fair hc_lock_exact_b 4096 5 1024 0.35 0.80 0.35 40.0 0.004 0.8 40 8.0 140 0.0050 0.00 0.0003 0.055 0 0.45 0.18 0.98 0.99"
  "smamujoco_halfcheetah_2x3_fair hc_lock_guard_c 4096 5 1024 0.50 0.50 0.28 30.0 0.010 1.5 28 5.0 100 0.0030 0.00 0.0008 0.045 1 0.65 0.14 0.95 0.99"
  "smamujoco_halfcheetah_2x3_fair hc_lock_return_d 4096 5 1024 0.55 0.45 0.24 25.0 0.012 2.0 24 4.0 90 0.0020 0.02 0.0010 0.045 1 0.55 0.12 0.92 0.98"
  "smamujoco_ant_2x4_fair ant_lock_fair_a 2048 8 512 0.90 0.10 0.08 12.0 0.080 16.0 0 1.0 20 0.0000 0.20 0.0010 0.008 0 0.45 0.10 0.90 0.98"
  "smamujoco_ant_2x4_fair ant_lock_fair_b 2048 8 512 0.95 0.08 0.06 12.0 0.060 14.0 0 1.0 25 0.0000 0.18 0.0012 0.010 0 0.45 0.10 0.90 0.98"
  "smamujoco_ant_2x4_fair ant_lock_balanced_c 2048 8 512 1.00 0.05 0.05 10.0 0.050 12.0 4 1.2 30 0.0000 0.12 0.0015 0.012 0 0.45 0.10 0.90 0.98"
  "smamujoco_ant_2x4_fair ant_lock_recovery_d 2048 8 512 0.95 0.08 0.06 12.0 0.060 14.0 0 1.0 25 0.0000 0.18 0.0012 0.010 1 0.75 0.10 0.90 0.98"
)

echo "[$(date -u +%FT%TZ)] Safe-Fair 500k lock finetune queue ${RUN_ID}: ${#QUEUE[@]} rows, max_seconds=${MAX_SECONDS}"
for row in "${QUEUE[@]}"; do
  # shellcheck disable=SC2086
  run_variant ${row}
  rc=$?
  if [[ "${rc}" -eq 99 ]]; then
    break
  fi
done

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-id "${RUN_ID}" || true
echo "[$(date -u +%FT%TZ)] Safe-Fair 500k lock finetune queue ${RUN_ID} done; elapsed=$(elapsed)s"
