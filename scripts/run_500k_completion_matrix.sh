#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/_external_baseline_env.sh"

set +e
set -uo pipefail

PYTHON_BIN="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
MANIFEST_SCRIPT="${ROOT_DIR}/scripts/generate_500k_completion_manifest.py"
TABLE_500K_SCRIPT="${ROOT_DIR}/scripts/generate_500k_comparison_table.py"
TABLE_SCRIPT="${ROOT_DIR}/scripts/regenerate_result_tables.py"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"
QUEUE_CSV="${SAFE_FAIR_500K_QUEUE_CSV:-${RUN_ROOT}/configs/missing_500k_matrix_queue.csv}"
RUN_ID="${SAFE_FAIR_500K_RUN_ID:-missing_500k_matrix_$(date -u +%Y%m%d_%H%M%S)}"
MAX_SECONDS="${SAFE_FAIR_500K_MAX_SECONDS:-43200}"
MIN_START_SECONDS="${SAFE_FAIR_500K_MIN_START_SECONDS:-1200}"
REFRESH_EVERY="${SAFE_FAIR_500K_REFRESH_EVERY:-3}"
WAIT_FOR_SESSION="${SAFE_FAIR_500K_WAIT_FOR_SESSION:-}"

mkdir -p "${QUEUE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[$(date -u +%FT%TZ)] waiting for tmux session '${WAIT_FOR_SESSION}' before starting 500k queue"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 300
  done
  echo "[$(date -u +%FT%TZ)] dependency tmux session '${WAIT_FOR_SESSION}' is gone; starting 500k queue"
fi

START_TS="$(date +%s)"
STATUS_FILE="${QUEUE_DIR}/${RUN_ID}_status.tsv"

if [[ ! -f "${QUEUE_CSV}" ]]; then
  "${PYTHON_BIN}" "${MANIFEST_SCRIPT}" || exit 1
fi

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'priority\tphase\trun_group\ttask\tmethod\tseed\tsteps\tfair_budget\ttag\tstatus\telapsed_s\tutc_time\n' > "${STATUS_FILE}"
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
  local priority="$1"
  local phase="$2"
  local run_group="$3"
  local task="$4"
  local method="$5"
  local seed="$6"
  local steps="$7"
  local fair_budget="$8"
  local tag="$9"
  local status="${10}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${priority}" "${phase}" "${run_group}" "${task}" "${method}" "${seed}" "${steps}" "${fair_budget}" \
    "${tag}" "${status}" "$(elapsed)" "$(date -u +%FT%TZ)" >> "${STATUS_FILE}"
}

semantic_already_finished() {
  local run_group="$1"
  local task="$2"
  local method="$3"
  local seed="$4"
  local steps="$5"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${run_group}" "${task}" "${method}" "${seed}" "${steps}" <<'PY'
from pathlib import Path
import json
import sys

root, run_group, task, method, seed, steps = sys.argv[1:]
root = Path(root)
seed = int(seed)
steps = int(steps)

for cfg_path in (root / "logs").rglob("config.json"):
    if not (cfg_path.parent / "eval_summary.json").exists():
        continue
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if cfg.get("run_group") != run_group:
        continue
    task_obj = cfg.get("task")
    method_obj = cfg.get("method")
    task_name = task_obj.get("name") if isinstance(task_obj, dict) else task_obj
    method_name = method_obj.get("name") if isinstance(method_obj, dict) else method_obj
    if task_name != task or method_name != method:
        continue
    try:
        if int(cfg.get("seed")) != seed:
            continue
        if int(cfg.get("total_steps")) < steps:
            continue
    except Exception:
        continue
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
  "MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210"
)

TUNED_ENV=()
TUNED_SUFFIX="clean"

set_tuned_env() {
  local task="$1"
  local method="$2"
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
          TUNED_SUFFIX="cs_sweet_a_500k"
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
        smamujoco_ant_2x4_fair)
          TUNED_SUFFIX="ant_default"
          ;;
        smamujoco_manyagent_ant_fair)
          TUNED_SUFFIX="manyagent_default"
          ;;
      esac
      ;;
  esac
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

refresh_tables() {
  "${PYTHON_BIN}" "${TABLE_SCRIPT}" || true
  "${PYTHON_BIN}" "${TABLE_500K_SCRIPT}" || true
}

run_one() {
  local priority="$1"
  local phase="$2"
  local run_group="$3"
  local task="$4"
  local method="$5"
  local seed="$6"
  local steps="$7"
  local rollout="$8"
  local eval_episodes="$9"
  local ppo_epochs="${10}"
  local minibatch_size="${11}"
  local fair_budget="${12}"

  if ! can_start_next; then
    echo "[$(date -u +%FT%TZ)] stopping before ${phase}; time_left=$(time_left)s < min_start=${MIN_START_SECONDS}s"
    return 99
  fi

  set_tuned_env "${task}" "${method}"
  local tag
  tag="$(make_run_tag "${phase}" "${task}" "${method}" "${TUNED_SUFFIX}" "${seed}")"
  local start
  start="$(date +%s)"

  if semantic_already_finished "${run_group}" "${task}" "${method}" "${seed}" "${steps}"; then
    echo "==== SKIP ${phase} ${task} / ${method} / seed${seed}; completed 500k+ row already exists ===="
    log_status "${priority}" "${phase}" "${run_group}" "${task}" "${method}" "${seed}" "${steps}" "${fair_budget}" "${tag}" "skipped"
    return 0
  fi

  echo "==== START p${priority} ${phase} ${task} / ${method} / seed${seed} / steps=${steps} / tag=${tag} ===="
  log_status "${priority}" "${phase}" "${run_group}" "${task}" "${method}" "${seed}" "${steps}" "${fair_budget}" "${tag}" "started"

  if env "${COMMON_ENV[@]}" "${TUNED_ENV[@]}" \
    "SAFE_FAIR_RUN_GROUP_FILTER=${run_group}" \
    "SAFE_FAIR_TASK_FILTER=${task}" \
    "SAFE_FAIR_METHOD_FILTER=${method}" \
    "SAFE_FAIR_SEED_FILTER=${seed}" \
    "SAFE_FAIR_FAIR_BUDGET_FILTER=${fair_budget}" \
    "SAFE_FAIR_CFG_TOTAL_STEPS=${steps}" \
    "SAFE_FAIR_CFG_ROLLOUT_STEPS=${rollout}" \
    "SAFE_FAIR_CFG_PPO_EPOCHS=${ppo_epochs}" \
    "SAFE_FAIR_CFG_MINIBATCH_SIZE=${minibatch_size}" \
    "SAFE_FAIR_CFG_EVAL_EPISODES=${eval_episodes}" \
    "SAFE_FAIR_CFG_RUN_TAG=${tag}" \
    "${PYTHON_BIN}" "${BACKEND_SCRIPT}"; then
    local dur=$(( $(date +%s) - start ))
    echo "==== DONE ${phase} ${task} / ${method} seed${seed} in ${dur}s; elapsed=$(elapsed)s left=$(time_left)s ===="
    log_status "${priority}" "${phase}" "${run_group}" "${task}" "${method}" "${seed}" "${steps}" "${fair_budget}" "${tag}" "finished"
    return 0
  else
    local dur=$(( $(date +%s) - start ))
    echo "!!!! FAILED ${phase} ${task} / ${method} seed${seed} after ${dur}s; continuing"
    log_status "${priority}" "${phase}" "${run_group}" "${task}" "${method}" "${seed}" "${steps}" "${fair_budget}" "${tag}" "failed"
    return 1
  fi
}

echo "[$(date -u +%FT%TZ)] 500k missing-matrix run_id=${RUN_ID} max_seconds=${MAX_SECONDS}"
echo "queue csv: ${QUEUE_CSV}"
echo "status file: ${STATUS_FILE}"

attempts=0
paused=0

{
  IFS= read -r _header
  while IFS=',' read -r priority phase run_group task method seed total_steps rollout_steps eval_episodes ppo_epochs minibatch_size fair_budget include_sfb; do
    [[ -z "${priority}" ]] && continue
    run_one "${priority}" "${phase}" "${run_group}" "${task}" "${method}" "${seed}" "${total_steps}" "${rollout_steps}" "${eval_episodes}" "${ppo_epochs}" "${minibatch_size}" "${fair_budget}"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      log_status "${priority}" "${phase}" "${run_group}" "${task}" "${method}" "${seed}" "${total_steps}" "${fair_budget}" "" "paused_before_start"
      paused=1
      break
    fi
    attempts=$((attempts + 1))
    if [[ "${REFRESH_EVERY}" -gt 0 && $((attempts % REFRESH_EVERY)) -eq 0 ]]; then
      refresh_tables
    fi
  done
} < "${QUEUE_CSV}"

refresh_tables

if [[ "${paused}" -eq 1 ]]; then
  echo "[$(date -u +%FT%TZ)] 500k missing matrix paused; elapsed=$(elapsed)s time_left=$(time_left)s"
else
  echo "[$(date -u +%FT%TZ)] 500k missing matrix finished queue scan; elapsed=$(elapsed)s time_left=$(time_left)s"
fi
echo "status file: ${STATUS_FILE}"
