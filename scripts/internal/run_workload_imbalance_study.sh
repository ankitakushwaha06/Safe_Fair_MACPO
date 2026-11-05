#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

set +e
set -uo pipefail

PYTHON_BIN="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
SUMMARY_SCRIPT="${ROOT_DIR}/scripts/internal/summarize_workload_imbalance.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"
RUN_ID="${SAFE_FAIR_WORKLOAD_RUN_ID:-workload_imbalance_study_$(date -u +%Y%m%d_%H%M%S)}"
MAX_SECONDS="${SAFE_FAIR_WORKLOAD_MAX_SECONDS:-172800}"
MIN_START_SECONDS="${SAFE_FAIR_WORKLOAD_MIN_START_SECONDS:-0}"
WAIT_FOR_SESSION="${SAFE_FAIR_WORKLOAD_WAIT_FOR_SESSION:-safe_fair_500k_lock_20260503}"

mkdir -p "${QUEUE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[$(date -u +%FT%TZ)] waiting for tmux session '${WAIT_FOR_SESSION}' before starting workload-imbalance scheduled queue"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 300
  done
  echo "[$(date -u +%FT%TZ)] dependency tmux session '${WAIT_FOR_SESSION}' is gone; starting workload-imbalance scheduled queue"
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
  local task="$1"
  local method="$2"
  local variant="$3"
  local tag="$4"
  local rollout_steps="$5"
  local ppo_epochs="$6"
  local minibatch_size="$7"
  local safety_scale="$8"
  local init_ls="$9"
  local safety_lr="${10}"
  local max_ls="${11}"
  local fair_budget="${12}"
  local fair_lr="${13}"
  local max_lf="${14}"
  local init_lf="${15}"
  local burden_recovery="${16}"
  local status="${17}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "workload_imbalance" "${task}" "${method}" "0" "500000" "${variant}" "${tag}" \
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
  "SAFE_FAIR_RUN_GROUP_FILTER=workload_imbalance"
  "SAFE_FAIR_SEED_FILTER=0"
  "SAFE_FAIR_CFG_TOTAL_STEPS=500000"
  "SAFE_FAIR_CFG_EVAL_EPISODES=16"
  "MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210"
)

run_row() {
  local task="$1"
  local method="$2"
  local variant="$3"
  local rollout_steps="$4"
  local ppo_epochs="$5"
  local minibatch_size="$6"
  local safety_scale="$7"
  local init_ls="$8"
  local safety_lr="$9"
  local max_ls="${10}"
  local fair_lr="${11}"
  local max_lf="${12}"
  local warmup="${13}"
  local budget_init="${14}"
  local budget_ramp="${15}"
  local margin="${16}"
  local init_lf="${17}"
  local entropy="${18}"
  local fair_budget="${19}"
  local burden_recovery="${20}"
  local burden_strength="${21}"
  local recovery_margin="${22}"
  local recovery_blend="${23}"
  local recovery_speed="${24}"
  local tag="${RUN_ID}_${variant}_s0"

  if ! can_start_next; then
    echo "[$(date -u +%FT%TZ)] stopping before ${variant}; time_left=$(time_left)s < min_start=${MIN_START_SECONDS}s"
    log_status "${task}" "${method}" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "paused_before_start"
    return 99
  fi

  if run_tag_finished "${tag}"; then
    echo "==== SKIP ${task} / ${method} / seed0 / 500k / ${variant}; eval summary already exists ===="
    log_status "${task}" "${method}" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "skipped"
    return 0
  fi

  echo "==== START ${task} / ${method} / seed0 / 500k / ${variant} / tag=${tag} ===="
  log_status "${task}" "${method}" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "started"
  local start
  start="$(date +%s)"

  env "${COMMON_ENV[@]}" \
    "SAFE_FAIR_TASK_FILTER=${task}" \
    "SAFE_FAIR_METHOD_FILTER=${method}" \
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
    echo "==== FINISH ${task} / ${method} / ${variant} / duration=${duration}s ===="
    log_status "${task}" "${method}" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "finished"
  else
    echo "==== FAIL ${task} / ${method} / ${variant} / rc=${rc} / duration=${duration}s ===="
    log_status "${task}" "${method}" "${variant}" "${tag}" "${rollout_steps}" "${ppo_epochs}" "${minibatch_size}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_budget}" "${fair_lr}" "${max_lf}" "${init_lf}" "${burden_recovery}" "failed"
  fi
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-id "${RUN_ID}" || true
  return 0
}

TASKS=(
  "smamujoco_halfcheetah_2x3_fair hc2x3 4096 5 1024 0.045 0.50 0.30 30.0"
  "smamujoco_halfcheetah_3x2_workload hc3x2 4096 5 1024 0.045 0.50 0.30 30.0"
  "smamujoco_halfcheetah_6x1_workload hc6x1 4096 5 1024 0.040 0.55 0.30 35.0"
  "smamujoco_ant_2x4_fair ant2x4 2048 8 512 0.010 0.10 0.08 12.0"
  "smamujoco_ant_2x4d_workload ant2x4d 2048 8 512 0.010 0.10 0.08 12.0"
  "smamujoco_ant_4x2_workload ant4x2 2048 8 512 0.012 0.12 0.08 14.0"
  "smamujoco_ant_8x1_workload ant8x1 2048 8 512 0.014 0.15 0.10 18.0"
  "smamujoco_manyagent_ant_fair manyant4x2 2048 8 512 0.018 0.15 0.10 18.0"
  "smamujoco_walker2d_2x3_workload walker2x3 4096 5 1024 0.020 0.40 0.20 25.0"
)

echo "[$(date -u +%FT%TZ)] Workload imbalance scheduled queue ${RUN_ID}: max_seconds=${MAX_SECONDS}"
for spec in "${TASKS[@]}"; do
  # shellcheck disable=SC2086
  set -- ${spec}
  task="$1"; prefix="$2"; rollout="$3"; epochs="$4"; minibatch="$5"; fair_budget="$6"; init_ls="$7"; safety_lr="$8"; max_ls="$9"

  run_row "${task}" "ippo" "${prefix}_ippo_ref" "${rollout}" "${epochs}" "${minibatch}" "1.00" "0.00" "0.00" "0.0" "0.000" "0.0" "0" "1.0" "0" "0.0000" "0.00" "0.0020" "${fair_budget}" "0" "0.45" "0.10" "0.90" "0.98"
  [[ "$?" -eq 99 ]] && break
  run_row "${task}" "macpo" "${prefix}_macpo_safety_ref" "${rollout}" "${epochs}" "${minibatch}" "0.70" "${init_ls}" "${safety_lr}" "${max_ls}" "0.000" "0.0" "0" "1.0" "0" "0.0000" "0.00" "0.0015" "${fair_budget}" "0" "0.45" "0.12" "0.90" "0.98"
  [[ "$?" -eq 99 ]] && break
  run_row "${task}" "mappo_lagrangian" "${prefix}_lag_safety_ref" "${rollout}" "${epochs}" "${minibatch}" "0.70" "${init_ls}" "${safety_lr}" "${max_ls}" "0.000" "0.0" "0" "1.0" "0" "0.0000" "0.00" "0.0015" "${fair_budget}" "0" "0.45" "0.12" "0.90" "0.98"
  [[ "$?" -eq 99 ]] && break
  run_row "${task}" "safe_fair_macpo" "${prefix}_sf_safety_first" "${rollout}" "${epochs}" "${minibatch}" "0.70" "${init_ls}" "${safety_lr}" "${max_ls}" "0.020" "4.0" "8" "2.0" "40" "0.0020" "0.05" "0.0010" "${fair_budget}" "1" "0.55" "0.12" "0.92" "0.98"
  [[ "$?" -eq 99 ]] && break
  run_row "${task}" "safe_fair_macpo" "${prefix}_sf_low_imbalance" "${rollout}" "${epochs}" "${minibatch}" "0.80" "${init_ls}" "${safety_lr}" "${max_ls}" "0.060" "12.0" "0" "1.0" "25" "0.0000" "0.15" "0.0010" "${fair_budget}" "0" "0.45" "0.10" "0.90" "0.98"
  [[ "$?" -eq 99 ]] && break
  run_row "${task}" "safe_fair_macpo" "${prefix}_sf_return_balanced" "${rollout}" "${epochs}" "${minibatch}" "0.90" "${init_ls}" "${safety_lr}" "${max_ls}" "0.030" "6.0" "6" "1.5" "35" "0.0010" "0.08" "0.0015" "${fair_budget}" "1" "0.65" "0.10" "0.90" "0.98"
  [[ "$?" -eq 99 ]] && break
done

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-id "${RUN_ID}" || true
echo "[$(date -u +%FT%TZ)] Workload imbalance scheduled queue ${RUN_ID} done; elapsed=$(elapsed)s"
