#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs" / "main"

TAGS = {
    "hc_sf_sweet_a",
    "hc_sf_sweet_b",
    "hc_sf_sweet_c",
    "hc_sf_sweet_d",
    "cs_sf_sweet_a",
    "cs_sf_sweet_b",
    "cs_sf_sweet_c",
    "cs_sf_sweet_d",
}


def load_rows() -> list[dict]:
    rows = []
    for cfg_path in LOG_ROOT.glob("*/*/seed_*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if cfg.get("run_tag") not in TAGS:
            continue
        eval_path = cfg_path.with_name("eval_summary.json")
        if not eval_path.exists():
            continue
        ev = json.loads(eval_path.read_text())
        rows.append(
            {
                "task": ev.get("task"),
                "tag": ev.get("run_tag"),
                "return": float(ev.get("return_mean", 0.0)),
                "safety": float(ev.get("safety_cost_mean", 0.0)),
                "viol": float(ev.get("violation_rate", 0.0)),
                "fair": float(ev.get("fairness_cost_mean", 0.0)),
                "jain": float(ev.get("jain_burden_mean", 0.0)),
                "time": float(ev.get("train_wallclock_sec", 0.0)) / 60.0,
            }
        )
    rows.sort(key=lambda r: (r["task"], r["tag"]))
    return rows


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def print_table(rows: list[dict]) -> None:
    headers = ["Task", "Variant", "Return", "Safety", "Viol", "Fairness", "Jain", "Time"]
    widths = [22, 14, 10, 8, 6, 10, 7, 7]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(vals: list[str]) -> str:
        return "|" + "|".join(f" {v:<{w}} " for v, w in zip(vals, widths)) + "|"

    print(sep)
    print(line(headers))
    print(sep)
    for r in rows:
        print(
            line(
                [
                    r["task"],
                    r["tag"],
                    fmt(r["return"]),
                    fmt(r["safety"]),
                    fmt(r["viol"]),
                    fmt(r["fair"]),
                    fmt(r["jain"], 3),
                    fmt(r["time"], 1),
                ]
            )
        )
    print(sep)


def main() -> None:
    rows = load_rows()
    if not rows:
        print("No completed Safe-Fair tuning rows found yet.")
        return
    print_table(rows)
    for task in sorted({r["task"] for r in rows}):
        feasible = [r for r in rows if r["task"] == task and r["safety"] <= 1.0 and r["viol"] <= 0.05]
        if not feasible:
            print(f"{task}: no safety-feasible tuning variant yet.")
            continue
        best_return = max(feasible, key=lambda r: r["return"])
        best_fair = min(feasible, key=lambda r: r["fair"])
        print(
            f"{task}: best safe return = {best_return['tag']} ({best_return['return']:.2f}); "
            f"best safe fairness = {best_fair['tag']} ({best_fair['fair']:.2f})."
        )


if __name__ == "__main__":
    main()
