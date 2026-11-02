#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs" / "main" / "smamujoco_halfcheetah_2x3_fair" / "safe_fair_macpo"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


rows = []
for run_dir in sorted(LOG_ROOT.glob("seed_*")):
    cfg_path = run_dir / "config.json"
    eval_path = run_dir / "eval_summary.json"
    if not cfg_path.exists() or not eval_path.exists():
        continue
    cfg = load_json(cfg_path)
    result = load_json(eval_path)
    tag = cfg.get("run_tag", "")
    if not tag.startswith("sretune_"):
        continue
    rows.append({
        "tag": tag,
        "run_dir": run_dir.name,
        "return_mean": result.get("return_mean"),
        "safety_cost_mean": result.get("safety_cost_mean"),
        "violation_rate": result.get("violation_rate"),
        "fairness_cost_mean": result.get("fairness_cost_mean"),
        "jain_burden_mean": result.get("jain_burden_mean"),
        "safety_cost_p90": result.get("safety_cost_p90"),
        "safety_cost_max": result.get("safety_cost_max"),
    })

rows.sort(key=lambda x: x["tag"])

headers = [
    ("tag", 10),
    ("return_mean", 12),
    ("safety_cost", 12),
    ("viol_rate", 10),
    ("fair_cost", 12),
    ("jain", 8),
    ("safety_p90", 11),
    ("safety_max", 11),
]

line = "+" + "+".join("-" * (w + 2) for _, w in headers) + "+"
print(line)
print("| " + " | ".join(name.ljust(w) for name, w in headers) + " |")
print(line)
for row in rows:
    values = [
        str(row["tag"]).ljust(10),
        f"{row['return_mean']:.2f}".rjust(12),
        f"{row['safety_cost_mean']:.2f}".rjust(12),
        f"{row['violation_rate']:.2f}".rjust(10),
        f"{row['fairness_cost_mean']:.2f}".rjust(12),
        f"{row['jain_burden_mean']:.4f}".rjust(8),
        f"{row['safety_cost_p90']:.2f}".rjust(11),
        f"{row['safety_cost_max']:.2f}".rjust(11),
    ]
    print("| " + " | ".join(values) + " |")
print(line)
