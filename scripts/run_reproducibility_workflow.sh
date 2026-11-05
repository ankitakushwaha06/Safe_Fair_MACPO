#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/_external_baseline_env.sh"

PYTHON_BIN="${ROOT_DIR}/.venv-sg/bin/python"
MANIFEST="${SAFE_FAIR_EVALUATION_MANIFEST:-${ROOT_DIR}/configs/evaluation_standard.yaml}"
RUN_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
QUEUE_DIR="${RUN_ROOT}/logs/queue"
RUN_ID="${SAFE_FAIR_REPRODUCIBILITY_RUN_ID:-reproducibility_workflow_$(date -u +%Y%m%d_%H%M%S)}"
JOBS="${SAFE_FAIR_REPRODUCIBILITY_PARALLEL_JOBS:-4}"
MAX_SECONDS="${SAFE_FAIR_REPRODUCIBILITY_MAX_SECONDS:-43200}"
EVAL_SECONDS="${SAFE_FAIR_REPRODUCIBILITY_EVAL_TIMEOUT_SECONDS:-14400}"
START_TS="$(date +%s)"
JOB_FILE="${RUN_ROOT}/jobs/${RUN_ID}_jobs.txt"
STATUS_FILE="${QUEUE_DIR}/${RUN_ID}_status.tsv"

mkdir -p "${QUEUE_DIR}" "${RUN_ROOT}/jobs" "${RUN_ROOT}/tables" "${ROOT_DIR}/figures"
printf 'utc_time\tstatus\tdetail\telapsed_s\n' > "${STATUS_FILE}"

elapsed() {
  echo $(( $(date +%s) - START_TS ))
}

remaining() {
  local left=$(( MAX_SECONDS - $(elapsed) ))
  if [[ "${left}" -lt 0 ]]; then
    echo 0
  else
    echo "${left}"
  fi
}

log_status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$(elapsed)" >> "${STATUS_FILE}"
}

echo "[$(date -u +%FT%TZ)] reproducibility workflow run_id=${RUN_ID}"
echo "manifest=${MANIFEST}"
echo "jobs=${JOBS}"
echo "max_seconds=${MAX_SECONDS}"
log_status "started" "time-bounded evaluation and ordered training queue"

"${PYTHON_BIN}" "${ROOT_DIR}/codes/audit_result_integrity.py" || true
"${PYTHON_BIN}" "${ROOT_DIR}/codes/recompute_burden_summaries.py" || true
"${PYTHON_BIN}" "${ROOT_DIR}/codes/plot_safe_fair_frontiers.py" \
  --inputs \
    "${RUN_ROOT}/tables/main_results.csv" \
    "${RUN_ROOT}/tables/all_eval_rows_current.csv" \
    "${RUN_ROOT}/tables/ablation_results.csv" \
    "${RUN_ROOT}/tables/burden_summary_recomputed.csv" \
  --out-dir "${ROOT_DIR}/figures/frontiers" || true

log_status "eval_started" "episodes=20 seeds=0..4 stress=standard,fairness_stress"
timeout "${EVAL_SECONDS}" "${PYTHON_BIN}" "${ROOT_DIR}/codes/eval_many_seeds.py" \
  --manifest "${MANIFEST}" \
  --episodes "${SAFE_FAIR_REPRODUCIBILITY_EVAL_EPISODES:-20}" \
  --eval-seeds 0 1 2 3 4 \
  --deterministic-policy true \
  --stress-modes standard fairness_stress \
  --out "${RUN_ROOT}/tables/multiseed_evaluation.csv"
eval_rc=$?
log_status "eval_exit_${eval_rc}" "${RUN_ROOT}/tables/multiseed_evaluation.csv"

if [[ -f "${RUN_ROOT}/tables/multiseed_evaluation.csv" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/codes/bootstrap_eval_ci.py" \
    --input "${RUN_ROOT}/tables/multiseed_evaluation.csv" \
    --out "${RUN_ROOT}/tables/multiseed_evaluation_ci.csv" || true
fi

"${PYTHON_BIN}" "${ROOT_DIR}/codes/generate_evaluation_jobs.py" \
  --manifest "${MANIFEST}" \
  --out "${JOB_FILE}"
make_rc=$?
if [[ "${make_rc}" -ne 0 ]]; then
  log_status "failed_make_jobs" "${make_rc}"
  exit "${make_rc}"
fi

train_left="$(remaining)"
if [[ "${train_left}" -lt 600 ]]; then
  log_status "skipped_training" "less_than_600s_remaining"
else
  log_status "training_queue_started" "${JOB_FILE}; seconds=${train_left}"
  if command -v parallel >/dev/null 2>&1 && parallel --version 2>/dev/null | head -1 | grep -qi 'GNU parallel'; then
    timeout "${train_left}" parallel -j "${JOBS}" --joblog "${QUEUE_DIR}/${RUN_ID}_parallel_joblog.tsv" < "${JOB_FILE}"
    train_rc=$?
  else
    timeout "${train_left}" xargs -P "${JOBS}" -L 1 -I {} bash -lc "{}" < "${JOB_FILE}"
    train_rc=$?
  fi
  log_status "training_queue_exit_${train_rc}" "${JOB_FILE}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/regenerate_result_tables.py" || true
"${PYTHON_BIN}" "${ROOT_DIR}/codes/recompute_burden_summaries.py" || true
"${PYTHON_BIN}" "${ROOT_DIR}/codes/collect_burden_decomposition.py" \
  --manifest "${MANIFEST}" \
  --out "${RUN_ROOT}/tables/burden_decomposition.csv" \
  --fig-dir "${ROOT_DIR}/figures/burden_decomposition" || true
"${PYTHON_BIN}" "${ROOT_DIR}/codes/plot_safe_fair_frontiers.py" \
  --inputs \
    "${RUN_ROOT}/tables/main_results.csv" \
    "${RUN_ROOT}/tables/all_eval_rows_current.csv" \
    "${RUN_ROOT}/tables/ablation_results.csv" \
    "${RUN_ROOT}/tables/burden_summary_recomputed.csv" \
  --out-dir "${ROOT_DIR}/figures/frontiers" || true
"${PYTHON_BIN}" "${ROOT_DIR}/codes/summarize_evaluation.py" \
  --inputs "${RUN_ROOT}/tables/*.csv" \
  --out "${RUN_ROOT}/tables/evaluation_summary.md" || true

log_status "finished" "${RUN_ID}"
echo "[$(date -u +%FT%TZ)] reproducibility workflow finished elapsed=$(elapsed)s"
