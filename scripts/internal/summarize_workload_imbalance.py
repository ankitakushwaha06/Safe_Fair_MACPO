#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "safe_fair_macpo_runs"
LOG_ROOT = RUN_ROOT / "logs"
TABLE_ROOT = RUN_ROOT / "tables"

ENV_LABELS: Dict[str, str] = {
    "smamujoco_halfcheetah_2x3_fair": "HalfCheetah-2x3",
    "smamujoco_halfcheetah_3x2_workload": "HalfCheetah-3x2",
    "smamujoco_halfcheetah_6x1_workload": "HalfCheetah-6x1",
    "smamujoco_ant_2x4_fair": "Ant-2x4",
    "smamujoco_ant_2x4d_workload": "Ant-2x4d",
    "smamujoco_ant_4x2_workload": "Ant-4x2",
    "smamujoco_ant_8x1_workload": "Ant-8x1",
    "smamujoco_manyagent_ant_fair": "ManyAgent-Ant-4x2",
    "smamujoco_walker2d_2x3_workload": "Walker2d-2x3",
}


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if not math.isnan(out) else default
    except Exception:
        return default


def fmt(value: object, digits: int = 2) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def load_rows(run_id: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cfg_path in LOG_ROOT.rglob("config.json"):
        eval_path = cfg_path.parent / "eval_summary.json"
        if not eval_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            ev = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_tag = str(cfg.get("run_tag", ""))
        if not run_tag.startswith(run_id):
            continue
        task_obj = cfg.get("task", {})
        method_obj = cfg.get("method", {})
        task = task_obj.get("name") if isinstance(task_obj, dict) else str(task_obj)
        method = method_obj.get("name") if isinstance(method_obj, dict) else str(method_obj)
        hard = ev.get("hard_violation_rate")
        if hard is None:
            hard = ev.get("violation_rate")
        rows.append(
            {
                "Environment": ENV_LABELS.get(str(task), str(task)),
                "Task": str(task),
                "Method": str(method),
                "Tag": run_tag,
                "Return": as_float(ev.get("return_mean")),
                "Safety": as_float(ev.get("safety_cost_mean")),
                "Hard": as_float(hard),
                "Imbalance": as_float(ev.get("fairness_cost_mean")),
                "Jain": as_float(ev.get("jain_burden_mean")),
                "Gini": as_float(ev.get("gini_burden_mean")),
                "Burden": ev.get("per_agent_burden_mean", []),
            }
        )
    return rows


def lower(row: Dict[str, object], key: str) -> float:
    return as_float(row.get(key), math.inf)


def higher(row: Dict[str, object], key: str) -> float:
    return as_float(row.get(key), -math.inf)


def best_safety_first(rows: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
    rows = list(rows)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            0 if lower(row, "Hard") <= 0.0 else 1,
            lower(row, "Safety"),
            lower(row, "Imbalance"),
            -higher(row, "Return"),
        ),
    )


def best_return_safe(rows: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
    rows = [row for row in rows if lower(row, "Hard") <= 0.0 and lower(row, "Safety") <= 0.0]
    if not rows:
        return None
    return max(rows, key=lambda row: higher(row, "Return"))


def as_output(row: Optional[Dict[str, object]], source: str, env: str) -> Dict[str, str]:
    if row is None:
        return {
            "Environment": env,
            "Source": source,
            "Method": "--",
            "Return": "PENDING",
            "Safety": "PENDING",
            "Hard": "PENDING",
            "Imbalance": "PENDING",
            "Jain": "PENDING",
            "Gini": "PENDING",
            "Tag": "PENDING",
        }
    return {
        "Environment": env,
        "Source": source,
        "Method": str(row.get("Method", "")),
        "Return": fmt(row.get("Return"), 2),
        "Safety": fmt(row.get("Safety"), 2),
        "Hard": fmt(row.get("Hard"), 2),
        "Imbalance": fmt(row.get("Imbalance"), 2),
        "Jain": fmt(row.get("Jain"), 3),
        "Gini": fmt(row.get("Gini"), 3),
        "Tag": str(row.get("Tag", "")),
    }


def build_summary(rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    envs = sorted({str(row["Environment"]) for row in rows})
    for env in envs:
        env_rows = [row for row in rows if row["Environment"] == env]
        sf_rows = [row for row in env_rows if row["Method"] == "safe_fair_macpo"]
        baseline_rows = [row for row in env_rows if row["Method"] != "safe_fair_macpo"]
        out.append(as_output(best_safety_first(sf_rows), "best Safe-Fair safety-first", env))
        out.append(as_output(best_safety_first(baseline_rows), "best baseline safety-first", env))
        out.append(as_output(best_return_safe(env_rows), "best exact-safe return", env))
    return out


def build_candidates(rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["Environment"]),
            0 if lower(row, "Hard") <= 0.0 else 1,
            lower(row, "Safety"),
            lower(row, "Imbalance"),
            -higher(row, "Return"),
        ),
    )
    return [
        {
            "Environment": str(row["Environment"]),
            "Method": str(row["Method"]),
            "Return": fmt(row.get("Return"), 2),
            "Safety": fmt(row.get("Safety"), 2),
            "Hard": fmt(row.get("Hard"), 2),
            "Imbalance": fmt(row.get("Imbalance"), 2),
            "Jain": fmt(row.get("Jain"), 3),
            "Gini": fmt(row.get("Gini"), 3),
            "Tag": str(row.get("Tag", "")),
        }
        for row in rows
    ]


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aligned_text(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "No completed workload-imbalance rows yet.\n"
    headers = list(rows[0].keys())
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))
    lines = ["  ".join(f"{h:<{widths[h]}}" for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    numeric = {"Return", "Safety", "Hard", "Imbalance", "Jain", "Gini"}
    for row in rows:
        cells = []
        for h in headers:
            align = ">" if h in numeric else "<"
            cells.append(f"{row[h]:{align}{widths[h]}}")
        lines.append("  ".join(cells))
    return "\n".join(lines) + "\n"


def write_text(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(aligned_text(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    rows = load_rows(args.run_id)
    summary = build_summary(rows)
    candidates = build_candidates(rows)

    for name, table in [("summary", summary), ("candidates", candidates)]:
        csv_path = TABLE_ROOT / f"{args.run_id}_{name}.csv"
        txt_path = TABLE_ROOT / f"{args.run_id}_{name}.txt"
        write_csv(table, csv_path)
        write_text(table, txt_path)
        write_text(table, TABLE_ROOT / f"workload_imbalance_latest_{name}.txt")
        print(txt_path)
        print(aligned_text(table))


if __name__ == "__main__":
    main()
