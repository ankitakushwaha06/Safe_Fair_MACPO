#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/_external_baseline_env.sh"

set +e
set -uo pipefail

PYTHON_BIN="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
TABLE_SCRIPT="${ROOT_DIR}/scripts/regenerate_result_tables.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"

RUN_ID="${SAFE_FAIR_PROMOTED_RUN_ID:-promoted_multiseed_$(date -u +%Y%m%d_%H%M%S)}"
MAX_SECONDS="${SAFE_FAIR_PROMOTED_MAX_SECONDS:-54000}"
MIN_START_SECONDS="${SAFE_FAIR_PROMOTED_MIN_START_SECONDS:-900}"
CONFIRM_SEEDS_STR="${SAFE_FAIR_PROMOTED_CONFIRM_SEEDS:-1 2 3 4}"
ABLATION_SEEDS_STR="${SAFE_FAIR_PROMOTED_ABLATION_SEEDS:-0 1 2}"
START_TS="$(date +%s)"
STATUS_FILE="${QUEUE_DIR}/${RUN_ID}_status.tsv"

mkdir -p "${QUEUE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

read -r -a CONFIRM_SEEDS <<< "${CONFIRM_SEEDS_STR}"
read -r -a ABLATION_SEEDS <<< "${ABLATION_SEEDS_STR}"

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'phase\ttask\tmethod\tseed\tsteps\ttag\tstatus\telapsed_s\tutc_time\n' > "${STATUS_FILE}"
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
  local method="$3"
  local seed="$4"
  local steps="$5"
  local tag="$6"
  local status="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "${status}" "$(elapsed)" "$(date -u +%FT%TZ)" \
    >> "${STATUS_FILE}"
}

run_already_finished() {
  local tag="$1"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${tag}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
tag = sys.argv[2]
for cfg_path in (root / "logs").rglob("config.json"):
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        continue
    if cfg.get("run_tag") == tag and (cfg_path.parent / "eval_summary.json").exists():
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
  "SAFE_FAIR_EXPERIMENT_SEEDS=0,1,2,3,4"
  "SAFE_FAIR_ABLATION_EXPERIMENT_SEEDS=0,1,2,3,4"
  "SAFE_FAIR_SAFEFAIRBENCH_EXPERIMENT_SEEDS=0,1,2,3,4"
  "SAFE_FAIR_SAFEFAIRBENCH_ABLATION_EXPERIMENT_SEEDS=0,1,2,3,4"
  "MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210"
)

TUNED_ENV=()
TUNED_SUFFIX="clean"

set_tuned_env() {
  local task="$1"
  local method="$2"
  local steps="$3"
  TUNED_ENV=()
  TUNED_SUFFIX="clean"

  case "${method}" in
    macpo|mappo_lagrangian)
      if [[ "${task}" == "smamujoco_halfcheetah_2x3_fair" ]]; then
        TUNED_SUFFIX="hc_safety_tuned"
        TUNED_ENV+=(
          "SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=0.80"
          "SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY=0.10"
          "SAFE_FAIR_CFG_SAFETY_DUAL_LR=0.08"
          "SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY=10.0"
        )
      fi
      ;;
  esac

  case "${method}" in
    safe_fair*)
      case "${task}" in
        smamujoco_halfcheetah_2x3_fair)
          TUNED_SUFFIX="hc_pareto_d"
          TUNED_ENV+=(
            "SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=0.80"
            "SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY=0.10"
            "SAFE_FAIR_CFG_SAFETY_DUAL_LR=0.08"
            "SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY=10.0"
            "SAFE_FAIR_CFG_FAIRNESS_DUAL_LR=0.015"
            "SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS=1.25"
            "SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES=8"
            "SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT=2.5"
            "SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES=45"
            "SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN=0.0010"
            "SAFE_FAIR_CFG_INIT_LAMBDA_FAIRNESS=0.03"
            "SAFE_FAIR_CFG_ENTROPY_COEF=0.0035"
          )
          ;;
        safetygym_point_multigoal1_fair)
          TUNED_SUFFIX="mg_loose"
          TUNED_ENV+=(
            "SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=0.35"
            "SAFE_FAIR_CFG_FAIRNESS_DUAL_LR=0.012"
            "SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS=3.0"
            "SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES=24"
            "SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT=5.0"
            "SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES=35"
            "SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN=0.005"
            "SAFE_FAIR_CFG_ENTROPY_COEF=0.0035"
          )
          ;;
        safetygym_chargeshare_fair)
          TUNED_SUFFIX="cs_sweet_a"
          TUNED_ENV+=(
            "SAFE_FAIR_CFG_SAFETY_TARGET_SCALE=0.80"
            "SAFE_FAIR_CFG_FAIRNESS_DUAL_LR=0.008"
            "SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS=1.00"
            "SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES=24"
            "SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT=8.0"
            "SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES=70"
            "SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN=0.002"
            "SAFE_FAIR_CFG_ENTROPY_COEF=0.0050"
          )
          ;;
        safefairbench_vmas_multigoal)
          TUNED_SUFFIX="mg_zero_safety_projection"
          TUNED_ENV+=(
            "SAFE_FAIR_METHOD_USE_SAFETY_RECOVERY=1"
            "SAFE_FAIR_TASK_KWARGS_JSON={\"collision_radius\":0.13,\"hazard_radius\":0.15,\"yield_distance\":0.24}"
            "SAFE_FAIR_TASK_SAFETY_BUDGET_SCALE=4.0"
          )
          ;;
      esac
      ;;
  esac

  if [[ "${steps}" == "500000" && "${method}" == "safe_fair_macpo" ]]; then
    case "${task}" in
      safetygym_chargeshare_fair)
        TUNED_SUFFIX="cs_sweet_a_500k"
        ;;
    esac
  fi
}

slug() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_]/_/g'
}

make_run_tag() {
  local phase="$1"
  local task="$2"
  local method="$3"
  local suffix="$4"
  local seed="$5"
  printf '%s_%s_%s_%s_%s_s%s' \
    "${RUN_ID}" "$(slug "${phase}")" "$(slug "${task}")" "$(slug "${method}")" "$(slug "${suffix}")" "${seed}"
}

run_one() {
  local phase="$1"
  local run_group="$2"
  local task="$3"
  local method="$4"
  local seed="$5"
  local steps="$6"
  local rollout="$7"
  local eval_episodes="$8"
  local ppo_epochs="${9:-8}"

  if ! can_start_next; then
    echo "[$(date -u +%FT%TZ)] stopping before ${phase}; time_left=$(time_left)s < min_start=${MIN_START_SECONDS}s"
    return 99
  fi

  set_tuned_env "${task}" "${method}" "${steps}"
  local tag
  tag="$(make_run_tag "${phase}" "${task}" "${method}" "${TUNED_SUFFIX}" "${seed}")"
  local start
  start="$(date +%s)"

  if run_already_finished "${tag}"; then
    echo "==== SKIP ${phase} ${task} / ${method} / seed${seed} / ${tag}; already finished ===="
    log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "skipped"
    return 0
  fi

  local include_sfb="SAFE_FAIR_INCLUDE_SAFEFAIRBENCH=0"
  if [[ "${task}" == safefairbench_* ]]; then
    include_sfb="SAFE_FAIR_INCLUDE_SAFEFAIRBENCH=1"
  fi

  echo "==== START ${phase} ${task} / ${method} / seed${seed} / steps=${steps} / tag=${tag} ===="
  log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "started"

  if env "${COMMON_ENV[@]}" "${TUNED_ENV[@]}" "${include_sfb}" \
    "SAFE_FAIR_RUN_GROUP_FILTER=${run_group}" \
    "SAFE_FAIR_TASK_FILTER=${task}" \
    "SAFE_FAIR_METHOD_FILTER=${method}" \
    "SAFE_FAIR_SEED_FILTER=${seed}" \
    "SAFE_FAIR_CFG_TOTAL_STEPS=${steps}" \
    "SAFE_FAIR_CFG_ROLLOUT_STEPS=${rollout}" \
    "SAFE_FAIR_CFG_PPO_EPOCHS=${ppo_epochs}" \
    "SAFE_FAIR_CFG_MINIBATCH_SIZE=512" \
    "SAFE_FAIR_CFG_EVAL_EPISODES=${eval_episodes}" \
    "SAFE_FAIR_CFG_RUN_TAG=${tag}" \
    "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
    local dur=$(( $(date +%s) - start ))
    echo "==== DONE ${phase} ${task} / ${method} seed${seed} in ${dur}s; elapsed=$(elapsed)s left=$(time_left)s ===="
    log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "finished"
    return 0
  else
    local dur=$(( $(date +%s) - start ))
    echo "!!!! FAILED ${phase} ${task} / ${method} seed${seed} after ${dur}s; continuing"
    log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "failed"
    return 1
  fi
}

run_hc_guard_a() {
  local seed="$1"
  local base_tag
  local tag
  base_tag="$(make_run_tag "180k_base" "smamujoco_halfcheetah_2x3_fair" "safe_fair_macpo" "hc_pareto_d" "${seed}")"
  tag="${RUN_ID}_hc_guard_a_320k_s${seed}"

  if ! can_start_next; then
    echo "[$(date -u +%FT%TZ)] stopping before hc_guard; time_left=$(time_left)s < min_start=${MIN_START_SECONDS}s"
    return 99
  fi
  if run_already_finished "${tag}"; then
    echo "==== SKIP hc_guard HalfCheetah seed${seed}; ${tag} already finished ===="
    log_status "hc_guard" "smamujoco_halfcheetah_2x3_fair" "safe_fair_macpo" "${seed}" "320000" "${tag}" "skipped"
    return 0
  fi

  echo "==== START hc_guard HalfCheetah / safe_fair_macpo / seed${seed} / tag=${tag} ===="
  log_status "hc_guard" "smamujoco_halfcheetah_2x3_fair" "safe_fair_macpo" "${seed}" "320000" "${tag}" "started"
  local start
  start="$(date +%s)"
  if env "${COMMON_ENV[@]}" \
    "SAFE_FAIR_RUN_TRAINING=0" \
    "SAFE_FAIR_HC_GUARD_SEED=${seed}" \
    "SAFE_FAIR_HC_GUARD_BASE_TAGS=${base_tag}" \
    "SAFE_FAIR_HC_GUARD_VARIANTS=a" \
    "SAFE_FAIR_HC_GUARD_TAG_PREFIX=${RUN_ID}_hc_guard" \
    "SAFE_FAIR_HC_GUARD_STATUS_FILE=${RUN_ID}_halfcheetah_guard_status.tsv" \
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_halfcheetah_guarded_finetune.py"; then
    local dur=$(( $(date +%s) - start ))
    echo "==== DONE hc_guard seed${seed} in ${dur}s; elapsed=$(elapsed)s left=$(time_left)s ===="
    log_status "hc_guard" "smamujoco_halfcheetah_2x3_fair" "safe_fair_macpo" "${seed}" "320000" "${tag}" "finished"
    return 0
  else
    local dur=$(( $(date +%s) - start ))
    echo "!!!! FAILED hc_guard seed${seed} after ${dur}s; continuing"
    log_status "hc_guard" "smamujoco_halfcheetah_2x3_fair" "safe_fair_macpo" "${seed}" "320000" "${tag}" "failed"
    return 1
  fi
}

refresh_tables() {
  "${PYTHON_BIN}" "${TABLE_SCRIPT}" || true
}

echo "[$(date -u +%FT%TZ)] multiseed benchmark run_id=${RUN_ID} max_seconds=${MAX_SECONDS}"
echo "status file: ${STATUS_FILE}"
echo "confirm seeds: ${CONFIRM_SEEDS[*]}"
echo "ablation seeds: ${ABLATION_SEEDS[*]}"

phase_complete=1

run_checked() {
  local rc=0
  run_one "$@" || rc=$?
  if [[ "${rc}" -eq 99 ]]; then
    phase_complete=0
    return 99
  fi
  return 0
}

guard_checked() {
  local rc=0
  run_hc_guard_a "$@" || rc=$?
  if [[ "${rc}" -eq 99 ]]; then
    phase_complete=0
    return 99
  fi
  return 0
}

for seed in "${CONFIRM_SEEDS[@]}"; do
  run_checked "180k_base" "main" "smamujoco_halfcheetah_2x3_fair" "safe_fair_macpo" "${seed}" "180000" "2048" "12" "8" || break
  guard_checked "${seed}" || break
  run_checked "180k_baseline" "main" "smamujoco_halfcheetah_2x3_fair" "reward_shaped_macpo_fair" "${seed}" "180000" "2048" "12" "8" || break
done

refresh_tables

if [[ "${phase_complete}" -eq 1 ]]; then
  for seed in "${CONFIRM_SEEDS[@]}"; do
    run_checked "500k_confirm" "main" "safetygym_point_multigoal1_fair" "safe_fair_macpo" "${seed}" "500000" "2048" "16" "8" || break
    run_checked "180k_baseline" "main" "safetygym_point_multigoal1_fair" "ippo" "${seed}" "180000" "2048" "12" "8" || break
    run_checked "180k_baseline" "main" "safetygym_point_multigoal1_fair" "reward_shaped_macpo_fair" "${seed}" "180000" "2048" "12" "8" || break
  done
fi

refresh_tables

if [[ "${phase_complete}" -eq 1 ]]; then
  for seed in "${CONFIRM_SEEDS[@]}"; do
    run_checked "500k_confirm" "main" "safetygym_chargeshare_fair" "safe_fair_macpo" "${seed}" "500000" "2048" "16" "8" || break
    run_checked "180k_baseline" "main" "safetygym_chargeshare_fair" "ippo" "${seed}" "180000" "2048" "12" "8" || break
    run_checked "180k_baseline" "main" "safetygym_chargeshare_fair" "macpo" "${seed}" "180000" "2048" "12" "8" || break
    run_checked "180k_baseline" "main" "safetygym_chargeshare_fair" "dfrl_soto_ggf" "${seed}" "180000" "2048" "12" "8" || break
  done
fi

refresh_tables

if [[ "${phase_complete}" -eq 1 ]]; then
  for seed in "${CONFIRM_SEEDS[@]}"; do
    for method in safe_fair_macpo macpo_recovery mappo_lagrangian_recovery; do
      run_checked "sfb_confirm" "safefairbench" "safefairbench_vmas_multigoal" "${method}" "${seed}" "4096" "256" "16" "8" || break 2
    done
  done
fi

refresh_tables

if [[ "${phase_complete}" -eq 1 ]]; then
  for seed in "${ABLATION_SEEDS[@]}"; do
    for task in safetygym_point_multigoal1_fair safetygym_chargeshare_fair; do
      for method in safe_fair_macpo safe_fair_no_fairness safe_fair_no_safety safe_fair_no_debt safe_fair_no_envy safe_fair_no_recovery safe_fair_per_agent_critic; do
        run_checked "180k_ablation" "ablation" "${task}" "${method}" "${seed}" "180000" "2048" "12" "8" || break 3
      done
    done
  done
fi

refresh_tables

if [[ "${phase_complete}" -eq 1 ]]; then
  for seed in "${ABLATION_SEEDS[@]}"; do
    for method in safe_fair_macpo safe_fair_no_fairness safe_fair_no_safety safe_fair_no_recovery safe_fair_no_debt safe_fair_no_hazard_burden safe_fair_no_contested_access_burden safe_fair_no_yield_burden safe_fair_jain_only safe_fair_envy_only safe_fair_per_agent_critic; do
      run_checked "sfb_ablation" "safefairbench_ablation" "safefairbench_vmas_multigoal" "${method}" "${seed}" "4096" "256" "16" "8" || break 2
    done
  done
fi

refresh_tables

if [[ "${phase_complete}" -eq 1 ]]; then
  for seed in 1 2; do
    for method in safe_fair_no_fairness safe_fair_no_safety safe_fair_no_debt safe_fair_per_agent_critic; do
      run_checked "hc_180k_ablation" "ablation" "smamujoco_halfcheetah_2x3_fair" "${method}" "${seed}" "180000" "2048" "12" "8" || break 2
    done
  done
fi

refresh_tables

echo "[$(date -u +%FT%TZ)] multiseed benchmark finished/paused; elapsed=$(elapsed)s time_left=$(time_left)s"
echo "status file: ${STATUS_FILE}"
