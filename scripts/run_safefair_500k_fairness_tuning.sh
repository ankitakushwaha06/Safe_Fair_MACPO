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
RUN_ID="${SAFE_FAIR_SF500K_RUN_ID:-safefair_500k_fairness_tune_$(date -u +%Y%m%d_%H%M%S)}"
MAX_SECONDS="${SAFE_FAIR_SF500K_MAX_SECONDS:-72000}"
MIN_START_SECONDS="${SAFE_FAIR_SF500K_MIN_START_SECONDS:-1200}"
WAIT_FOR_SESSION="${SAFE_FAIR_SF500K_WAIT_FOR_SESSION:-safe_fair_500k_missing_20260429}"

mkdir -p "${QUEUE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[$(date -u +%FT%TZ)] waiting for tmux session '${WAIT_FOR_SESSION}' before starting Safe-Fair 500k fairness tuning"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 300
  done
  echo "[$(date -u +%FT%TZ)] dependency tmux session '${WAIT_FOR_SESSION}' is gone; starting Safe-Fair 500k fairness tuning"
fi

START_TS="$(date +%s)"
STATUS_FILE="${QUEUE_DIR}/${RUN_ID}_status.tsv"

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'phase\ttask\tmethod\tseed\tsteps\tvariant\ttag\tfair_budget\tfair_lr\tmax_lambda_fairness\tinit_lambda_fairness\tburden_recovery\tstatus\telapsed_s\tutc_time\n' > "${STATUS_FILE}"
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
  local fair_budget="$7"
  local fair_lr="$8"
  local max_lf="$9"
  local init_lf="${10}"
  local burden_recovery="${11}"
  local status="${12}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${phase}" "${task}" "safe_fair_macpo" "${seed}" "${steps}" "${variant}" "${tag}" \
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
  "SAFE_FAIR_CFG_ROLLOUT_STEPS=2048"
  "SAFE_FAIR_CFG_PPO_EPOCHS=8"
  "SAFE_FAIR_CFG_MINIBATCH_SIZE=512"
  "SAFE_FAIR_CFG_EVAL_EPISODES=16"
  "MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210"
)

run_variant() {
  local task="$1"
  local variant="$2"
  local fair_lr="$3"
  local max_lf="$4"
  local warmup="$5"
  local budget_init="$6"
  local budget_ramp="$7"
  local margin="$8"
  local init_lf="$9"
  local entropy="${10}"
  local safety_scale="${11}"
  local fair_budget="${12}"
  local burden_recovery="${13}"
  local burden_strength="${14}"
  local tag="${RUN_ID}_${variant}_s0"

  if ! can_start_next; then
    echo "[$(date -u +%FT%TZ)] stopping before ${variant}; time_left=$(time_left)s < min_start=${MIN_START_SECONDS}s"
    log_status "fairness_tune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "paused_before_start"
    return 99
  fi

  if run_tag_finished "${tag}"; then
    echo "==== SKIP ${task} / safe_fair_macpo / seed0 / ${variant}; eval summary already exists ===="
    log_status "fairness_tune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "skipped"
    return 0
  fi

  echo "==== START ${task} / safe_fair_macpo / seed0 / 500k / ${variant} / tag=${tag} ===="
  log_status "fairness_tune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "started"
  local start
  start="$(date +%s)"

  env "${COMMON_ENV[@]}" \
    "SAFE_FAIR_TASK_FILTER=${task}" \
    "SAFE_FAIR_CFG_RUN_TAG=${tag}" \
    "SAFE_FAIR_CFG_FAIRNESS_DUAL_LR=${fair_lr}" \
    "SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS=${max_lf}" \
    "SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES=${warmup}" \
    "SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT=${budget_init}" \
    "SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES=${budget_ramp}" \
    "SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN=${margin}" \
    "SAFE_FAIR_CFG_INIT_LAMBDA_FAIRNESS=${init_lf}" \
    "SAFE_FAIR_CFG_ENTROPY_COEF=${entropy}" \
    "SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=${safety_scale}" \
    "SAFE_FAIR_TASK_FAIRNESS_BUDGET_PER_STEP=${fair_budget}" \
    "SAFE_FAIR_METHOD_USE_BURDEN_AWARE_RECOVERY=${burden_recovery}" \
    "SAFE_FAIR_METHOD_BURDEN_RECOVERY_STRENGTH=${burden_strength}" \
    "${PYTHON_BIN}" "${BACKEND_SCRIPT}"
  local rc=$?
  local duration=$(( $(date +%s) - start ))
  if [[ "${rc}" -eq 0 ]]; then
    echo "==== FINISH ${task} / ${variant} / duration=${duration}s ===="
    log_status "fairness_tune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "finished"
  else
    echo "==== FAIL ${task} / ${variant} / rc=${rc} / duration=${duration}s ===="
    log_status "fairness_tune_500k" "${task}" "0" "500000" "${variant}" "${tag}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "failed"
  fi
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-id "${RUN_ID}" || true
  return "${rc}"
}

# Fairness-first 500k search. Targets are deliberately stricter than the
# current selected 500k rows, with burden-aware recovery probes where repeated
# safety recovery may be assigning burden to the same agent.
QUEUE=(
  "smamujoco_halfcheetah_2x3_fair hc_fair_zero_c500k 0.050 6.0 6 1.0 0 0.0010 0.10 0.0005 0.60 0.035 0 0.45"
  "smamujoco_halfcheetah_2x3_fair hc_fair_balanced_c500k 0.030 4.0 8 1.5 25 0.0010 0.05 0.0010 0.65 0.030 0 0.45"
  "smamujoco_halfcheetah_2x3_fair hc_fair_recovery_c500k 0.025 4.0 8 1.5 30 0.0010 0.05 0.0010 0.65 0.030 1 0.75"
  "smamujoco_halfcheetah_2x3_fair hc_fair_return_floor_c500k 0.018 3.0 10 2.0 35 0.0015 0.03 0.0020 0.70 0.035 1 0.55"
  "safetygym_point_multigoal1_fair mg_fair_strict_c500k 0.040 8.0 8 1.5 20 0.0100 0.05 0.0025 0.35 0.025 0 0.45"
  "safetygym_point_multigoal1_fair mg_fair_hard_c500k 0.060 10.0 0 1.0 20 0.0100 0.15 0.0020 0.30 0.015 0 0.45"
  "safetygym_point_multigoal1_fair mg_fair_recovery_c500k 0.030 7.0 8 1.5 25 0.0080 0.08 0.0025 0.35 0.020 1 0.70"
  "safetygym_chargeshare_fair cs_fair_strict_c500k 0.020 3.0 12 2.0 40 0.0040 0.05 0.0030 0.65 0.030 0 0.45"
  "safetygym_chargeshare_fair cs_fair_hard_c500k 0.040 6.0 6 1.5 25 0.0060 0.12 0.0020 0.50 0.020 0 0.45"
  "safetygym_chargeshare_fair cs_fair_recovery_c500k 0.016 3.0 12 2.0 35 0.0040 0.05 0.0030 0.65 0.030 1 0.70"
  "safetygym_chargeshare_fair cs_fair_return_floor_c500k 0.012 2.0 18 3.0 45 0.0040 0.03 0.0040 0.80 0.040 1 0.55"
  "smamujoco_ant_2x4_fair ant_fair_strict_c500k 0.020 5.0 8 2.0 40 0.0000 0.05 0.0030 1.00 0.025 0 0.45"
  "smamujoco_ant_2x4_fair ant_fair_hard_c500k 0.040 8.0 0 1.0 20 0.0000 0.10 0.0020 1.00 0.015 0 0.45"
  "smamujoco_ant_2x4_fair ant_fair_recovery_c500k 0.025 6.0 6 1.5 30 0.0000 0.08 0.0025 1.00 0.020 1 0.70"
  "smamujoco_ant_2x4_fair ant_fair_return_floor_c500k 0.012 4.0 10 2.0 45 0.0000 0.03 0.0035 1.00 0.030 1 0.55"
)

echo "[$(date -u +%FT%TZ)] Safe-Fair 500k fairness-tuning queue ${RUN_ID}: ${#QUEUE[@]} rows, max_seconds=${MAX_SECONDS}"
for row in "${QUEUE[@]}"; do
  # shellcheck disable=SC2086
  run_variant ${row}
  rc=$?
  if [[ "${rc}" -eq 99 ]]; then
    break
  fi
done

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-id "${RUN_ID}" || true
echo "[$(date -u +%FT%TZ)] Safe-Fair 500k fairness-tuning queue ${RUN_ID} done; elapsed=$(elapsed)s"
