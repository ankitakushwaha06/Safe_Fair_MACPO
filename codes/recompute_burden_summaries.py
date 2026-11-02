#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "safe_fair_macpo_runs"
LOG_ROOT = RUN_ROOT / "logs"
TABLE_ROOT = RUN_ROOT / "tables"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gini(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    if np.sum(x) <= eps:
        return 0.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return diff / (2 * len(x) * np.sum(x) + eps)


def jain(x, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64)
    denom = len(x) * float(np.sum(x * x)) + eps
    return float(np.sum(x) ** 2 / denom) if len(x) else np.nan


def entropy_from_burdens(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1 or float(np.sum(x)) <= eps:
        return 1.0
    p = x / float(np.sum(x))
    h = -float(np.sum(p * np.log(np.maximum(p, eps))))
    return h / float(np.log(len(x)))


def parse_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return np.asarray([], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def row_from_eval(eval_path: Path) -> Dict[str, Any]:
    summary = load_json(eval_path)
    run_dir = eval_path.parent
    cfg_path = run_dir / "config.json"
    cfg = load_json(cfg_path) if cfg_path.exists() else {}
    task = cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}
    burden = parse_array(summary.get("per_agent_burden_mean"))
    burden_std = parse_array(summary.get("per_agent_burden_std"))
    raw_status = "summary_only" if burden.size else "not_logged"
    total = float(np.sum(burden)) if burden.size else np.nan
    burden_range = float(np.max(burden) - np.min(burden)) if burden.size else np.nan
    max_share = float(np.max(burden) / np.sum(burden)) if burden.size and np.sum(burden) > 1e-8 else np.nan
    return {
        "env": summary.get("task") or task.get("name", ""),
        "method": summary.get("method", ""),
        "run_tag": summary.get("run_tag", cfg.get("run_tag", "")),
        "seed": summary.get("seed", cfg.get("seed", "")),
        "budget": summary.get("global_step", cfg.get("total_steps", "")),
        "return_mean": summary.get("return_mean", np.nan),
        "safety_mean": summary.get("safety_cost_mean", np.nan),
        "hard_rate": summary.get("hard_violation_rate", summary.get("violation_rate", np.nan)),
        "fairness_mean": summary.get("fairness_cost_mean", np.nan),
        "jain": float(jain(burden)) if burden.size else summary.get("jain_burden_mean", np.nan),
        "gini": float(gini(burden)) if burden.size else summary.get("gini_burden_mean", np.nan),
        "mean_total_burden": total,
        "std_total_burden": float(np.sqrt(np.sum(burden_std * burden_std))) if burden_std.size else np.nan,
        "max_agent_burden": float(np.max(burden)) if burden.size else np.nan,
        "min_agent_burden": float(np.min(burden)) if burden.size else np.nan,
        "burden_range": burden_range,
        "overburdened_agent_share": max_share,
        "sacrifice_entropy": entropy_from_burdens(burden) if burden.size else np.nan,
        "burden_logging_status": raw_status,
        "log_dir": str(run_dir.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(TABLE_ROOT / "burden_summary_recomputed.csv"))
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = []
    for path in LOG_ROOT.rglob("eval_summary.json"):
        try:
            rows.append(row_from_eval(path))
        except Exception as exc:
            rows.append({"log_dir": str(path.parent.relative_to(ROOT)), "burden_logging_status": f"error:{type(exc).__name__}"})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(df)}")
    if len(df):
        missing = int((df.get("burden_logging_status") == "not_logged").sum())
        print(f"not_logged rows={missing}")


if __name__ == "__main__":
    main()
