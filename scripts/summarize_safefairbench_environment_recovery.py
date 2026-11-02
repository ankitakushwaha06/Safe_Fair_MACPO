#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs" / "safefairbench"
PREFIXES = (
    "cs_route_",
    "highway_early_",
    "highway_slots_",
    "highway_immediate_",
    "highway_fair_assist_",
    "pogema_paired_",
    "pogema_biased_",
)


def num(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


rows = []
for cfg_path in sorted(LOG_ROOT.glob("*/*/seed_*/config.json")):
    eval_path = cfg_path.parent / "eval_summary.json"
    if not eval_path.exists():
        continue
    cfg = json.loads(cfg_path.read_text())
    tag = str(cfg.get("run_tag", ""))
    if not tag.startswith(PREFIXES):
        continue
    ev = json.loads(eval_path.read_text())
    task = cfg.get("task", {}).get("name", "")
    method = cfg.get("method", {}).get("name", "")
    rows.append(
        {
            "task": task,
            "method": method,
            "tag": tag,
            "return": num(ev.get("return_mean")),
            "safety": num(ev.get("safety_cost_mean")),
            "hard": num(ev.get("hard_violation_rate", ev.get("violation_rate"))),
            "fair": num(ev.get("fairness_cost_mean")),
            "jain": num(ev.get("jain_burden_mean")),
            "gini": num(ev.get("gini_burden_mean")),
            "burden": ev.get("per_agent_burden_mean", []),
        }
    )

rows.sort(key=lambda r: (r["task"], r["method"], r["tag"]))
headers = [
    ("task", 31),
    ("method", 22),
    ("tag", 31),
    ("return", 9),
    ("safety", 8),
    ("hard", 6),
    ("fair", 8),
    ("jain", 7),
    ("gini", 7),
    ("burden", 32),
]
line = "+" + "+".join("-" * (w + 2) for _, w in headers) + "+"
print(line)
print("| " + " | ".join(name.ljust(w) for name, w in headers) + " |")
print(line)
for r in rows:
    burden = "[" + ", ".join(f"{num(x):.2f}" for x in r["burden"]) + "]"
    values = [
        r["task"][:31].ljust(31),
        r["method"][:22].ljust(22),
        r["tag"][:31].ljust(31),
        f"{r['return']:9.2f}",
        f"{r['safety']:8.2f}",
        f"{r['hard']:6.2f}",
        f"{r['fair']:8.2f}",
        f"{r['jain']:7.3f}",
        f"{r['gini']:7.3f}",
        burden[:32].ljust(32),
    ]
    print("| " + " | ".join(values) + " |")
print(line)
if not rows:
    print("No weak-env repair rows found yet.")
