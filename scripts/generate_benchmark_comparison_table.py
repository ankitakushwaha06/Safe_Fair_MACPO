#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs"
TABLE_ROOT = ROOT / "safe_fair_macpo_runs" / "tables"
RUN_TAG = "core_benchmark_180k"

FAIRNESS_MEASURE = {
    "smamujoco_halfcheetah_2x3_fair": "control effort + jerk + safety-share",
    "smamujoco_ant_2x4_fair": "control effort + bottleneck burden",
    "safetygym_point_multigoal1_fair": "waiting + contested access + hazard exposure",
    "safetygym_chargeshare_fair": "charger denial + queue delay + low-energy burden",
    "smamujoco_manyagent_ant_fair": "local workload + safety burden",
}

TASK_LABEL = {
    "smamujoco_halfcheetah_2x3_fair": "HalfCheetah-2x3-Fair",
    "smamujoco_ant_2x4_fair": "Ant-2x4-Fair",
    "safetygym_point_multigoal1_fair": "MultiGoal-Fair",
    "safetygym_chargeshare_fair": "ChargeShare-Fair",
    "smamujoco_manyagent_ant_fair": "ManyAgent-Ant-Fair",
}

METHOD_LABEL = {
    "ippo": "IPPO",
    "mappo": "MAPPO",
    "macpo": "MACPO",
    "mappo_lagrangian": "MAPPO-Lagrangian",
    "dfrl_soto_ggf": "DFRL / SOTO-GGF",
    "reward_shaped_mappo_fair": "Reward-shaped MAPPO-Fair",
    "reward_shaped_macpo_fair": "Reward-shaped MACPO-Fair",
    "safe_fair_macpo": "Safe-Fair MACPO",
    "safe_fair_no_debt": "Safe-Fair no debt",
    "safe_fair_no_envy": "Safe-Fair no envy",
    "safe_fair_no_recovery": "Safe-Fair no recovery",
}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_latest() -> pd.DataFrame:
    latest: Dict[Tuple[str, str, int, str], Tuple[float, Dict]] = {}
    for path in LOG_ROOT.rglob("eval_summary.json"):
        try:
            row = load_json(path)
        except Exception:
            continue
        if row.get("run_tag", "") != RUN_TAG:
            continue
        key = (row.get("task", ""), row.get("method", ""), int(row.get("seed", 0)), row.get("run_group", ""))
        mtime = path.stat().st_mtime
        if key not in latest or mtime > latest[key][0]:
            row["_path"] = str(path)
            latest[key] = (mtime, row)
    return pd.DataFrame([row for _, row in latest.values()])


def main() -> None:
    df = collect_latest()
    if df.empty:
        print("no benchmark rows found")
        return
    sub = df[df["run_group"] == "main"].copy()
    sub["Environment"] = sub["task"].map(lambda x: TASK_LABEL.get(x, x))
    sub["Method"] = sub["method"].map(lambda x: METHOD_LABEL.get(x, x))
    sub["Fairness Measure"] = sub["task"].map(lambda x: FAIRNESS_MEASURE.get(x, ""))
    sub["Time (min)"] = pd.to_numeric(sub.get("train_wallclock_sec", pd.Series(dtype=float)), errors="coerce") / 60.0
    cols = [
        "Environment",
        "Method",
        "Fairness Measure",
        "return_mean",
        "safety_cost_mean",
        "violation_rate",
        "fairness_cost_mean",
        "jain_burden_mean",
        "envy_gap_mean",
        "debt_auc_mean",
        "Time (min)",
    ]
    out = sub[cols].sort_values(["Environment", "Method"]).reset_index(drop=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_ROOT / "extended180k_comparison.csv", index=False)
    out.to_latex(TABLE_ROOT / "extended180k_comparison.tex", index=False, escape=False, float_format="%.3f")
    print(f"wrote {len(out)} rows -> {TABLE_ROOT / 'extended180k_comparison.csv'}")


if __name__ == "__main__":
    main()
