#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

PYTHON_BIN="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}/bin/python"
BACKEND_SCRIPT="${ROOT_DIR}/codes/safe_fair_macpo_experiments.py"
QUEUE_DIR="${ROOT_DIR}/safe_fair_macpo_runs/logs/queue"
LOG_FILE="${QUEUE_DIR}/halfcheetah_safefair_safety_finetune.log"
STATUS_FILE="${QUEUE_DIR}/halfcheetah_safefair_safety_finetune_status.tsv"

mkdir -p "${QUEUE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

BASE_CHECKPOINT="$(
  "${PYTHON_BIN}" - "${ROOT_DIR}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
log_root = root / "safe_fair_macpo_runs" / "logs" / "main" / "smamujoco_halfcheetah_2x3_fair" / "safe_fair_macpo"
preferred_tags = [
    "extended_benchmark_180k_main_hc_pareto_d_s0",
    "hc_sf_pareto_d",
]
best = None
for tag in preferred_tags:
    for cfg_path in log_root.glob("seed_*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        ckpt = cfg_path.parent / "model.pt"
        eval_path = cfg_path.parent / "eval_summary.json"
        if cfg.get("run_tag") == tag and ckpt.exists() and eval_path.exists():
            best = ckpt
            break
    if best is not None:
        break
if best is None:
    raise SystemExit("No safe HalfCheetah base checkpoint found for hc_pareto_d")
print(best)
PY
)"

printf 'tag\tsafety_scale\tinit_ls\tsafety_lr\tmax_ls\tfair_lr\tmax_lf\twarmup\tbudget_init\tbudget_ramp\tmargin\tinit_lf\tentropy\tstatus\telapsed_s\tutc_time\n' > "${STATUS_FILE}"

export SAFE_FAIR_MACPO_ROOT="${ROOT_DIR}/safe_fair_macpo_runs"
export SAFE_FAIR_QUICK_TEST=0
export SAFE_FAIR_RUN_TRAINING=1
export SAFE_FAIR_RUN_EVALUATION=1
export SAFE_FAIR_CLONE_OFFICIAL_REPOS=0
export SAFE_FAIR_USE_CUDA=1
export SAFE_FAIR_RUN_GROUP_FILTER="main"
export SAFE_FAIR_TASK_FILTER="smamujoco_halfcheetah_2x3_fair"
export SAFE_FAIR_METHOD_FILTER="safe_fair_macpo"
export SAFE_FAIR_SEED_FILTER="0"
export SAFE_FAIR_MAX_RUNS=1
export SAFE_FAIR_CFG_TOTAL_STEPS=320000
export SAFE_FAIR_CFG_ROLLOUT_STEPS=4096
export SAFE_FAIR_CFG_PPO_EPOCHS=8
export SAFE_FAIR_CFG_MINIBATCH_SIZE=512
export SAFE_FAIR_CFG_EVAL_EPISODES=16
export SAFE_FAIR_CFG_INIT_CHECKPOINT="${BASE_CHECKPOINT}"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

log_status() {
  local tag="$1"
  local safety_scale="$2"
  local init_ls="$3"
  local safety_lr="$4"
  local max_ls="$5"
  local fair_lr="$6"
  local max_lf="$7"
  local warmup="$8"
  local budget_init="$9"
  local budget_ramp="${10}"
  local margin="${11}"
  local init_lf="${12}"
  local entropy="${13}"
  local status="${14}"
  local elapsed="${15}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" \
    "${fair_lr}" "${max_lf}" "${warmup}" "${budget_init}" "${budget_ramp}" \
    "${margin}" "${init_lf}" "${entropy}" "${status}" "${elapsed}" "$(date -u +%FT%TZ)" \
    >> "${STATUS_FILE}"
}

run_variant() {
  local tag="$1"
  local safety_scale="$2"
  local init_ls="$3"
  local safety_lr="$4"
  local max_ls="$5"
  local fair_lr="$6"
  local max_lf="$7"
  local warmup="$8"
  local budget_init="$9"
  local budget_ramp="${10}"
  local margin="${11}"
  local init_lf="${12}"
  local entropy="${13}"
  local start
  start="$(date +%s)"

  echo "==== HC SAFETY FINETUNE ${tag}; base=${BASE_CHECKPOINT} ===="
  log_status "${tag}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_lr}" "${max_lf}" "${warmup}" "${budget_init}" "${budget_ramp}" "${margin}" "${init_lf}" "${entropy}" "started" "0"

  SAFE_FAIR_CFG_RUN_TAG="${tag}" \
  SAFE_FAIR_CFG_SAFETY_TARGET_SCALE="${safety_scale}" \
  SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY="${init_ls}" \
  SAFE_FAIR_CFG_SAFETY_DUAL_LR="${safety_lr}" \
  SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY="${max_ls}" \
  SAFE_FAIR_CFG_FAIRNESS_DUAL_LR="${fair_lr}" \
  SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS="${max_lf}" \
  SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES="${warmup}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT="${budget_init}" \
  SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES="${budget_ramp}" \
  SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN="${margin}" \
  SAFE_FAIR_CFG_INIT_LAMBDA_FAIRNESS="${init_lf}" \
  SAFE_FAIR_CFG_ENTROPY_COEF="${entropy}" \
  "${PYTHON_BIN}" "${BACKEND_SCRIPT}"

  log_status "${tag}" "${safety_scale}" "${init_ls}" "${safety_lr}" "${max_ls}" "${fair_lr}" "${max_lf}" "${warmup}" "${budget_init}" "${budget_ramp}" "${margin}" "${init_lf}" "${entropy}" "finished" "$(( $(date +%s) - start ))"
}

{
  echo "[$(date -u +%FT%TZ)] HalfCheetah Safe-Fair safety fine-tune started"
  echo "base checkpoint: ${BASE_CHECKPOINT}"
  run_variant "hc_sf_ftsafe_a_320k" 0.70 0.35 0.14 20.0 0.010 1.00 12 4.0 50 0.002 0.010 0.0030
  run_variant "hc_sf_ftsafe_b_320k" 0.60 0.60 0.20 25.0 0.008 0.80 16 5.0 60 0.003 0.000 0.0025
  run_variant "hc_sf_ftsafe_c_320k" 0.50 0.80 0.25 30.0 0.006 0.60 24 7.0 80 0.004 0.000 0.0015
  run_variant "hc_sf_ftsafe_d_320k" 0.80 0.20 0.10 15.0 0.015 1.50 10 4.0 40 0.001 0.000 0.0050
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/internal/regenerate_result_tables.py" || true
  echo "[$(date -u +%FT%TZ)] HalfCheetah Safe-Fair safety fine-tune finished"
} >> "${LOG_FILE}" 2>&1
