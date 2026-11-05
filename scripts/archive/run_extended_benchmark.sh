#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

# _external_baseline_env.sh enables `set -e`; this queue needs to continue
# after individual run failures and after the time-budget stop sentinel.
set +e
set -uo pipefail

PYTHON_BIN="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
TABLE_SCRIPT="${ROOT_DIR}/scripts/internal/regenerate_result_tables.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${QUEUE_DIR}"

RUN_ID="${SAFE_FAIR_EXTENDED_RUN_ID:-extended_benchmark_$(date -u +%Y%m%d_%H%M%S)}"
MAX_SECONDS="${SAFE_FAIR_EXTENDED_MAX_SECONDS:-36000}"
MIN_START_SECONDS="${SAFE_FAIR_EXTENDED_MIN_START_SECONDS:-900}"
SEED="${SAFE_FAIR_EXTENDED_SEED:-0}"
START_TS="$(date +%s)"
STATUS_FILE="${QUEUE_DIR}/${RUN_ID}_status.tsv"

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'phase\ttask\tmethod\tseed\tsteps\ttag\tstatus\telapsed_s\tutc_time\n' > "${STATUS_FILE}"
fi

time_left() {
  local now
  now="$(date +%s)"
  echo $((MAX_SECONDS - (now - START_TS)))
}

elapsed() {
  local now
  now="$(date +%s)"
  echo $((now - START_TS))
}

can_start_next() {
  local left
  left="$(time_left)"
  [[ "${left}" -gt "${MIN_START_SECONDS}" ]]
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
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
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
  "SAFE_FAIR_CFG_ROLLOUT_STEPS=2048"
  "SAFE_FAIR_CFG_PPO_EPOCHS=8"
  "SAFE_FAIR_CFG_MINIBATCH_SIZE=512"
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
    safe_fair_macpo|safe_fair_no_debt|safe_fair_no_envy|safe_fair_no_recovery|safe_fair_local_k1|safe_fair_heterogeneous_q)
      case "${task}" in
        smamujoco_halfcheetah_2x3_fair)
          # hc_sf_pareto_d: best current zero-hard-violation, zero-safety-cost
          # HalfCheetah Safe-Fair 180k trade-off.
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
        safetygym_chargeshare_fair)
          # cs_sf_sweet_a / review_cs_sf_strict: safest current ChargeShare
          # Safe-Fair setting with good burden balance.
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
        safetygym_point_multigoal1_fair)
          # review_mg_sf_loose: best current Safety-Gymnasium MultiGoal
          # return/fairness balance.
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
        smamujoco_ant_2x4_fair)
          # The default 180k Ant setting is currently the best verified
          # Safe-Fair point; do not perturb it with unverified tuning.
          TUNED_SUFFIX="ant_default"
          ;;
        smamujoco_manyagent_ant_fair)
          # No environment-specific tuning has been verified yet.
          TUNED_SUFFIX="manyagent_default"
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

run_one() {
  local phase="$1"
  local run_group="$2"
  local task="$3"
  local method="$4"
  local seed="$5"
  local steps="$6"
  local eval_episodes="$7"

  if [[ "${phase}" == "500k_promotion" && "${task}" == "smamujoco_halfcheetah_2x3_fair" && "${method}" == "safe_fair_macpo" ]]; then
    local tag="${RUN_ID}_${phase}_hc_guard_a_locked_s${seed}"
    echo "==== SKIP ${phase} ${task} / ${method} / seed${seed}; safety-first lock is guarded checkpoint selection, not terminal 500k training ===="
    echo "Use scripts/run_halfcheetah_guarded_finetune.py and report hc_sf_guard_a_320k for safety-first HalfCheetah."
    log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "skipped_guarded_lock"
    return 0
  fi

  if ! can_start_next; then
    echo "[$(date -u +%FT%TZ)] stopping before ${phase}; time_left=$(time_left)s < min_start=${MIN_START_SECONDS}s"
    return 99
  fi

  set_tuned_env "${task}" "${method}" "${steps}"
  local tag="${RUN_ID}_${phase}_${TUNED_SUFFIX}_s${seed}"
  local start
  start="$(date +%s)"

  if run_already_finished "${tag}"; then
    echo "==== SKIP ${phase} ${task} / ${method} / seed${seed} / ${steps} / ${tag}; eval summary already exists ===="
    log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "skipped"
    return 0
  fi

  echo "==== START ${phase} ${task} / ${method} / seed${seed} / ${steps} / ${tag} ===="
  log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "started"

  if env "${COMMON_ENV[@]}" "${TUNED_ENV[@]}" \
    "SAFE_FAIR_RUN_GROUP_FILTER=${run_group}" \
    "SAFE_FAIR_TASK_FILTER=${task}" \
    "SAFE_FAIR_METHOD_FILTER=${method}" \
    "SAFE_FAIR_SEED_FILTER=${seed}" \
    "SAFE_FAIR_CFG_TOTAL_STEPS=${steps}" \
    "SAFE_FAIR_CFG_EVAL_EPISODES=${eval_episodes}" \
    "SAFE_FAIR_CFG_RUN_TAG=${tag}" \
    "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
    local dur=$(( $(date +%s) - start ))
    echo "==== DONE ${phase} ${task} / ${method} in ${dur}s; queue_elapsed=$(elapsed)s; time_left=$(time_left)s ===="
    log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "finished"
    return 0
  else
    local dur=$(( $(date +%s) - start ))
    echo "!!!! FAILED ${phase} ${task} / ${method} after ${dur}s; continuing"
    log_status "${phase}" "${task}" "${method}" "${seed}" "${steps}" "${tag}" "failed"
    return 1
  fi
}

run_table_refresh() {
  if [[ -x "${PYTHON_BIN}" && -f "${TABLE_SCRIPT}" ]]; then
    "${PYTHON_BIN}" "${TABLE_SCRIPT}" || true
  fi
}

MAIN_TASKS=(
  "smamujoco_halfcheetah_2x3_fair"
  "smamujoco_ant_2x4_fair"
  "smamujoco_manyagent_ant_fair"
  "safetygym_point_multigoal1_fair"
  "safetygym_chargeshare_fair"
)

MAIN_METHODS=(
  "ippo"
  "mappo"
  "macpo"
  "mappo_lagrangian"
  "dfrl_soto_ggf"
  "reward_shaped_mappo_fair"
  "reward_shaped_macpo_fair"
  "safe_fair_macpo"
)

PROMOTION_ROWS=(
  "smamujoco_halfcheetah_2x3_fair mappo"
  "smamujoco_halfcheetah_2x3_fair macpo"
  "smamujoco_halfcheetah_2x3_fair reward_shaped_macpo_fair"
  "smamujoco_ant_2x4_fair mappo"
  "smamujoco_ant_2x4_fair reward_shaped_mappo_fair"
  "smamujoco_ant_2x4_fair reward_shaped_macpo_fair"
  "smamujoco_ant_2x4_fair safe_fair_macpo"
  "safetygym_point_multigoal1_fair ippo"
  "safetygym_point_multigoal1_fair reward_shaped_macpo_fair"
  "safetygym_point_multigoal1_fair safe_fair_macpo"
  "safetygym_chargeshare_fair ippo"
  "safetygym_chargeshare_fair macpo"
  "safetygym_chargeshare_fair dfrl_soto_ggf"
  "safetygym_chargeshare_fair safe_fair_macpo"
  "smamujoco_manyagent_ant_fair macpo"
  "smamujoco_manyagent_ant_fair safe_fair_macpo"
)

ABLATION_ROWS=(
  "smamujoco_ant_2x4_fair safe_fair_macpo"
  "smamujoco_ant_2x4_fair safe_fair_no_debt"
  "smamujoco_ant_2x4_fair safe_fair_no_envy"
  "smamujoco_ant_2x4_fair safe_fair_no_recovery"
  "smamujoco_ant_2x4_fair safe_fair_local_k1"
  "safetygym_chargeshare_fair safe_fair_macpo"
  "safetygym_chargeshare_fair safe_fair_no_debt"
  "safetygym_chargeshare_fair safe_fair_no_envy"
  "safetygym_chargeshare_fair safe_fair_no_recovery"
  "safetygym_chargeshare_fair safe_fair_local_k1"
)

SCALABILITY_ROWS=(
  "smamujoco_manyagent_ant_fair macpo"
  "smamujoco_manyagent_ant_fair safe_fair_macpo"
  "smamujoco_manyagent_ant_fair safe_fair_local_k1"
)

echo "[$(date -u +%FT%TZ)] extended benchmark run_id=${RUN_ID}, max_seconds=${MAX_SECONDS}, seed=${SEED}"
echo "status file: ${STATUS_FILE}"

phase_complete=1
for task in "${MAIN_TASKS[@]}"; do
  for method in "${MAIN_METHODS[@]}"; do
    run_one "180k_main" "main" "${task}" "${method}" "${SEED}" "180000" "12"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      phase_complete=0
      break 2
    fi
  done
done

run_table_refresh

if [[ "${phase_complete}" -eq 1 ]]; then
  echo "[$(date -u +%FT%TZ)] 180k main block finished; starting 500k promotion block if time remains."
  for row in "${PROMOTION_ROWS[@]}"; do
    read -r task method <<< "${row}"
    run_one "500k_promotion" "main" "${task}" "${method}" "${SEED}" "500000" "16"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      phase_complete=0
      break
    fi
  done
fi

run_table_refresh

if [[ "${phase_complete}" -eq 1 ]]; then
  echo "[$(date -u +%FT%TZ)] promotion block finished; starting 180k ablations if time remains."
  for row in "${ABLATION_ROWS[@]}"; do
    read -r task method <<< "${row}"
    run_one "180k_ablation" "ablation" "${task}" "${method}" "${SEED}" "180000" "12"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      phase_complete=0
      break
    fi
  done
fi

run_table_refresh

if [[ "${phase_complete}" -eq 1 ]]; then
  echo "[$(date -u +%FT%TZ)] ablation block finished; starting scalability probes if time remains."
  for row in "${SCALABILITY_ROWS[@]}"; do
    read -r task method <<< "${row}"
    run_one "180k_scalability" "scalability" "${task}" "${method}" "${SEED}" "180000" "12"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      phase_complete=0
      break
    fi
  done
fi

run_table_refresh

echo "[$(date -u +%FT%TZ)] extended benchmark finished/paused; elapsed=$(elapsed)s; time_left=$(time_left)s"
echo "status file: ${STATUS_FILE}"
