#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs" / "main" / "smamujoco_halfcheetah_2x3_fair" / "safe_fair_macpo"
TAGS = [f"hc_sf_pareto_{c}" for c in "abcdef"]


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for cfg in LOG_ROOT.glob("seed_*/config.json"):
        eval_path = cfg.parent / "eval_summary.json"
        if not eval_path.exists():
            continue
        with cfg.open() as f:
            config = json.load(f)
        tag = config.get("run_tag", "")
        if tag not in TAGS:
            continue
        with eval_path.open() as f:
            ev = json.load(f)
        rows.append(
            {
                "tag": tag,
                "return": float(ev.get("return_mean", float("nan"))),
                "safety": float(ev.get("safety_cost_mean", float("nan"))),
                "viol": float(ev.get("violation_rate", float("nan"))),
                "fair": float(ev.get("fairness_cost_mean", float("nan"))),
                "jain": float(ev.get("jain_burden_mean", float("nan"))),
                "fair_lr": float(config.get("fairness_dual_lr", float("nan"))),
                "max_lf": float(config.get("max_lambda_fairness", float("nan"))),
                "warmup": int(config.get("fairness_warmup_updates", 0)),
                "budget_init": float(config.get("fairness_budget_scale_init", float("nan"))),
                "budget_ramp": int(config.get("fairness_budget_ramp_updates", 0)),
                "init_lf": float(config.get("init_lambda_fairness", float("nan"))),
                "entropy": float(config.get("entropy_coef", float("nan"))),
                "safety_scale": float(config.get("safety_target_scale", float("nan"))),
            }
        )
    return sorted(rows, key=lambda r: r["tag"])


def main() -> None:
    rows = load_rows()
    if not rows:
        print("No pareto rows found.")
        return
    print(
        "+---------------+----------+--------+------+----------+--------+---------+--------+------------+-----------+---------+----------+"
    )
    print(
        "| Tag           | Return   | Safety | Viol | Fairness | Jain   | fair_lr | max_lf | budget_ini | warmup    | init_lf | entropy  |"
    )
    print(
        "+---------------+----------+--------+------+----------+--------+---------+--------+------------+-----------+---------+----------+"
    )
    for r in rows:
        print(
            f"| {r['tag']:<13} | "
            f"{r['return']:>8.2f} | "
            f"{r['safety']:>6.2f} | "
            f"{r['viol']:>4.2f} | "
            f"{r['fair']:>8.2f} | "
            f"{r['jain']:>6.3f} | "
            f"{r['fair_lr']:>7.3f} | "
            f"{r['max_lf']:>6.2f} | "
            f"{r['budget_init']:>10.2f} | "
            f"{r['warmup']:>9d} | "
            f"{r['init_lf']:>7.2f} | "
            f"{r['entropy']:>8.4f} |"
        )
    print(
        "+---------------+----------+--------+------+----------+--------+---------+--------+------------+-----------+---------+----------+"
    )


if __name__ == "__main__":
    main()
