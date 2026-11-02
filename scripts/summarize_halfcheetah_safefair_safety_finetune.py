#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs" / "main" / "smamujoco_halfcheetah_2x3_fair" / "safe_fair_macpo"


def fmt_float(value, width: int, decimals: int = 2) -> str:
    if value is None:
        return "--".rjust(width)
    try:
        return f"{float(value):>{width}.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)[:width].rjust(width)


rows = []
for cfg_path in sorted(LOG_ROOT.glob("seed_*/config.json")):
    eval_path = cfg_path.parent / "eval_summary.json"
    if not eval_path.exists():
        continue
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tag = cfg.get("run_tag", "")
    if not (tag.startswith("hc_sf_ftsafe_") or tag.startswith("hc_sf_guard_")):
        continue
    with eval_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    rows.append(
        {
            "tag": tag,
            "return": result.get("return_mean"),
            "safety": result.get("safety_cost_mean"),
            "hard": result.get("hard_violation_rate", result.get("violation_rate")),
            "fair": result.get("fairness_cost_mean"),
            "jain": result.get("jain_burden_mean"),
            "p90": result.get("safety_cost_p90"),
            "max": result.get("safety_cost_max"),
            "safety_scale": cfg.get("safety_target_scale"),
            "init_ls": cfg.get("init_lambda_safety"),
            "safety_lr": cfg.get("safety_dual_lr"),
            "max_ls": cfg.get("max_lambda_safety"),
            "fair_lr": cfg.get("fairness_dual_lr"),
            "max_lf": cfg.get("max_lambda_fairness"),
            "selected": result.get("selected_guard_kind", ""),
        }
    )

rows.sort(key=lambda r: r["tag"])

headers = [
    ("tag", 20),
    ("return", 9),
    ("safety", 8),
    ("hard", 6),
    ("fair", 8),
    ("jain", 7),
    ("p90", 7),
    ("max", 7),
    ("s_scale", 7),
    ("init_ls", 7),
    ("s_lr", 6),
    ("max_ls", 7),
    ("f_lr", 6),
    ("max_lf", 7),
    ("selected", 10),
]

line = "+" + "+".join("-" * (width + 2) for _, width in headers) + "+"
print(line)
print("| " + " | ".join(name.ljust(width) for name, width in headers) + " |")
print(line)
for row in rows:
    values = [
        str(row["tag"])[:20].ljust(20),
        fmt_float(row["return"], 9),
        fmt_float(row["safety"], 8),
        fmt_float(row["hard"], 6),
        fmt_float(row["fair"], 8),
        fmt_float(row["jain"], 7, 3),
        fmt_float(row["p90"], 7),
        fmt_float(row["max"], 7),
        fmt_float(row["safety_scale"], 7, 2),
        fmt_float(row["init_ls"], 7, 2),
        fmt_float(row["safety_lr"], 6, 2),
        fmt_float(row["max_ls"], 7, 2),
        fmt_float(row["fair_lr"], 6, 3),
        fmt_float(row["max_lf"], 7, 2),
        str(row["selected"])[:10].ljust(10),
    ]
    print("| " + " | ".join(values) + " |")
print(line)

if not rows:
    print("No hc_sf_ftsafe_* rows found.")
