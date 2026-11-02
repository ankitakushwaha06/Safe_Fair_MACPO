#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "safe_fair_macpo_runs"
LOG_ROOT = RUN_ROOT / "logs"
TABLE_ROOT = RUN_ROOT / "tables"

TASK_LABELS: Dict[str, str] = {
    "smamujoco_halfcheetah_2x3_fair": "HalfCheetah-2x3",
    "safetygym_point_multigoal1_fair": "Safety-Gym MultiGoal",
    "safetygym_chargeshare_fair": "Safety-Gym ChargeShare",
    "smamujoco_ant_2x4_fair": "Ant-2x4",
}

STANDARD_BASELINES = {"ippo", "mappo", "macpo", "mappo_lagrangian", "dfrl_soto_ggf"}


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if not math.isnan(out) else default
    except Exception:
        return default


def load_rows() -> List[Dict[str, object]]:
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
        task_obj = cfg.get("task", {})
        method_obj = cfg.get("method", {})
        task = task_obj.get("name") if isinstance(task_obj, dict) else str(task_obj)
        method = method_obj.get("name") if isinstance(method_obj, dict) else str(method_obj)
        if task not in TASK_LABELS:
            continue
        try:
            steps = int(cfg.get("total_steps") or 0)
        except Exception:
            steps = 0
        if steps < 500_000:
            continue
        hard = ev.get("hard_violation_rate")
        if hard is None:
            hard = ev.get("violation_rate")
        rows.append(
            {
                "task": task,
                "environment": TASK_LABELS[task],
                "method": method,
                "seed": cfg.get("seed"),
                "steps": steps,
                "run_group": cfg.get("run_group", ""),
                "run_tag": cfg.get("run_tag", ""),
                "variant": str(cfg.get("run_tag", "")).split("_safefair_500k_fairness_tune_")[-1],
                "return": as_float(ev.get("return_mean")),
                "safety": as_float(ev.get("safety_cost_mean")),
                "hard": as_float(hard),
                "fairness": as_float(ev.get("fairness_cost_mean")),
                "jain": as_float(ev.get("jain_burden_mean")),
                "gini": as_float(ev.get("gini_burden_mean")),
                "path": str(cfg_path.parent),
            }
        )
    return rows


def lower(x: object, default: float = math.inf) -> float:
    return as_float(x, default)


def higher(x: object, default: float = -math.inf) -> float:
    return as_float(x, default)


def best_by_fairness(rows: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
    rows = list(rows)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            lower(row.get("fairness")),
            lower(row.get("safety")),
            lower(row.get("hard")),
            -higher(row.get("return")),
        ),
    )


def best_safety_first_fairness(rows: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
    rows = list(rows)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            0 if lower(row.get("hard")) <= 0.0 else 1,
            lower(row.get("fairness")),
            lower(row.get("safety")),
            -higher(row.get("return")),
        ),
    )


def fmt_num(value: object, digits: int = 2) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def build_summary(rows: List[Dict[str, object]], run_id: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for task, label in TASK_LABELS.items():
        task_rows = [r for r in rows if r["task"] == task]
        current_sf = best_safety_first_fairness(
            r
            for r in task_rows
            if r["method"] == "safe_fair_macpo"
            and r["run_group"] == "main"
            and not str(r.get("run_tag", "")).startswith(run_id)
        )
        tuned = best_safety_first_fairness(
            r
            for r in task_rows
            if r["method"] == "safe_fair_macpo" and str(r.get("run_tag", "")).startswith(run_id)
        )
        baseline_any = best_by_fairness(
            r for r in task_rows if r["method"] in STANDARD_BASELINES and r["run_group"] == "main"
        )
        baseline_safe = best_by_fairness(
            r
            for r in task_rows
            if r["method"] in STANDARD_BASELINES
            and r["run_group"] == "main"
            and lower(r.get("safety")) <= 0.0
            and lower(r.get("hard")) <= 0.0
        )
        for source, row in (
            ("best existing Safe-Fair 500k", current_sf),
            ("best tuned Safe-Fair 500k", tuned),
            ("best standard baseline fairness", baseline_any),
            ("best exact-safe baseline fairness", baseline_safe),
        ):
            if row is None:
                out.append(
                    {
                        "Environment": label,
                        "Source": source,
                        "Method": "--",
                        "Tag": "PENDING",
                        "Return": "PENDING",
                        "Safety": "PENDING",
                        "Hard": "PENDING",
                        "Fairness": "PENDING",
                        "Jain": "PENDING",
                        "Gini": "PENDING",
                    }
                )
                continue
            out.append(
                {
                    "Environment": label,
                    "Source": source,
                    "Method": str(row.get("method", "")),
                    "Tag": str(row.get("run_tag", "")),
                    "Return": fmt_num(row.get("return"), 2),
                    "Safety": fmt_num(row.get("safety"), 2),
                    "Hard": fmt_num(row.get("hard"), 2),
                    "Fairness": fmt_num(row.get("fairness"), 2),
                    "Jain": fmt_num(row.get("jain"), 3),
                    "Gini": fmt_num(row.get("gini"), 3),
                }
            )
    return out


def build_candidate_rows(rows: List[Dict[str, object]], run_id: str) -> List[Dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row["method"] == "safe_fair_macpo" and str(row.get("run_tag", "")).startswith(run_id)
    ]
    candidates.sort(
        key=lambda row: (
            str(row.get("environment")),
            0 if lower(row.get("hard")) <= 0.0 else 1,
            lower(row.get("safety")),
            lower(row.get("fairness")),
            -higher(row.get("return")),
        )
    )
    out: List[Dict[str, str]] = []
    for row in candidates:
        out.append(
            {
                "Environment": str(row.get("environment", "")),
                "Tag": str(row.get("run_tag", "")),
                "Return": fmt_num(row.get("return"), 2),
                "Safety": fmt_num(row.get("safety"), 2),
                "Hard": fmt_num(row.get("hard"), 2),
                "Fairness": fmt_num(row.get("fairness"), 2),
                "Jain": fmt_num(row.get("jain"), 3),
                "Gini": fmt_num(row.get("gini"), 3),
            }
        )
    return out


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
        return "No completed tuned rows yet.\n"
    headers = list(rows[0].keys())
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))
    lines = ["  ".join(f"{h:<{widths[h]}}" for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        cells = []
        for h in headers:
            align = ">" if h in {"Return", "Safety", "Hard", "Fairness", "Jain", "Gini"} else "<"
            cells.append(f"{row[h]:{align}{widths[h]}}")
        lines.append("  ".join(cells))
    return "\n".join(lines) + "\n"


def write_text(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(aligned_text(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    run_id = args.run_id.strip()
    if not run_id:
        raise SystemExit("--run-id is required")

    rows = load_rows()
    summary = build_summary(rows, run_id)
    candidates = build_candidate_rows(rows, run_id)

    summary_csv = TABLE_ROOT / f"{run_id}_summary.csv"
    summary_txt = TABLE_ROOT / f"{run_id}_summary.txt"
    candidates_csv = TABLE_ROOT / f"{run_id}_candidates.csv"
    candidates_txt = TABLE_ROOT / f"{run_id}_candidates.txt"
    write_csv(summary, summary_csv)
    write_text(summary, summary_txt)
    write_csv(candidates, candidates_csv)
    write_text(candidates, candidates_txt)
    write_text(summary, TABLE_ROOT / "safefair_500k_fairness_tuning_latest_summary.txt")
    write_text(candidates, TABLE_ROOT / "safefair_500k_fairness_tuning_latest_candidates.txt")

    print(summary_txt)
    print(aligned_text(summary))
    if candidates:
        print(candidates_txt)
        print(aligned_text(candidates))


if __name__ == "__main__":
    main()
