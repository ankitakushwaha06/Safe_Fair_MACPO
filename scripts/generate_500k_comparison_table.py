#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "safe_fair_macpo_runs"
LOG_ROOT = RUN_ROOT / "logs"
TABLE_ROOT = RUN_ROOT / "tables"

TASK_LABELS: Dict[str, str] = {
    "smamujoco_halfcheetah_2x3_fair": "HalfCheetah-2x3",
    "safetygym_point_multigoal1_fair": "Safety-Gym MultiGoal",
    "safetygym_chargeshare_fair": "Safety-Gym ChargeShare",
    "smamujoco_ant_2x4_fair": "Ant-2x4",
    "smamujoco_manyagent_ant_fair": "ManyAgent-Ant",
}

METHOD_LABELS: Dict[str, str] = {
    "ippo": "IPPO",
    "mappo": "MAPPO",
    "macpo": "MACPO",
    "mappo_lagrangian": "MAPPO-Lag",
    "dfrl_soto_ggf": "DFRL/SOTO-GGF",
    "reward_shaped_mappo_fair": "reward-shaped MAPPO-Fair",
    "reward_shaped_macpo_fair": "reward-shaped MACPO-Fair",
    "safe_fair_macpo": "Safe-Fair MACPO",
}

TASKS: Sequence[str] = tuple(TASK_LABELS)
STANDARD_METHODS: Sequence[str] = (
    "ippo",
    "mappo",
    "macpo",
    "mappo_lagrangian",
    "dfrl_soto_ggf",
    "safe_fair_macpo",
)
FULL_METHODS: Sequence[str] = (
    "ippo",
    "mappo",
    "macpo",
    "mappo_lagrangian",
    "dfrl_soto_ggf",
    "reward_shaped_mappo_fair",
    "reward_shaped_macpo_fair",
    "safe_fair_macpo",
)


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_rows(log_root: Path = LOG_ROOT, seed: int = 0) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cfg_path in log_root.rglob("config.json"):
        eval_path = cfg_path.parent / "eval_summary.json"
        if not eval_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            ev = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(cfg.get("run_group", "")) != "main":
            continue
        try:
            steps = int(cfg.get("total_steps") or 0)
            cfg_seed = int(cfg.get("seed"))
        except Exception:
            continue
        if steps < 500_000 or cfg_seed != seed:
            continue
        task_obj = cfg.get("task", {})
        method_obj = cfg.get("method", {})
        task = task_obj.get("name") if isinstance(task_obj, dict) else str(task_obj)
        method = method_obj.get("name") if isinstance(method_obj, dict) else str(method_obj)
        if task not in TASKS or method not in FULL_METHODS:
            continue
        hard = ev.get("hard_violation_rate")
        if hard is None:
            hard = ev.get("violation_rate")
        row = {
            "task": task,
            "method": method,
            "seed": cfg_seed,
            "steps": steps,
            "run_tag": cfg.get("run_tag", ""),
            "return": as_float(ev.get("return_mean")),
            "safety": as_float(ev.get("safety_cost_mean")),
            "hard": as_float(hard),
            "fair": as_float(ev.get("fairness_cost_mean")),
            "jain": as_float(ev.get("jain_burden_mean")),
            "gini": as_float(ev.get("gini_burden_mean")),
            "path": str(cfg_path.parent),
        }
        rows.append(row)
    return rows


def score_candidate(row: Dict[str, object]) -> Tuple[int, float, float, float, float]:
    # Select a useful safety-first 500k row per method: any zero-hard row is
    # preferred over unsafe rows; among hard-safe rows prefer higher return,
    # then lower safety and fairness. If none are hard-safe, choose the least
    # unsafe row, then higher return.
    hard = as_float(row.get("hard"), 1.0)
    safety = as_float(row.get("safety"), math.inf)
    fair = as_float(row.get("fair"), math.inf)
    ret = as_float(row.get("return"), -math.inf)
    hard_safe = 1 if hard <= 0.0 else 0
    if hard_safe:
        return (1, ret, -safety, -fair, -len(str(row.get("run_tag", ""))))
    return (0, -hard, -safety, ret, -fair)


def select_row(rows: Iterable[Dict[str, object]], task: str, method: str) -> Optional[Dict[str, object]]:
    candidates = [row for row in rows if row["task"] == task and row["method"] == method]
    if not candidates:
        return None
    return max(candidates, key=score_candidate)


def fmt(value: object, digits: int = 2) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def build_table(methods: Sequence[str], rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    table: List[Dict[str, str]] = []
    for task in TASKS:
        for method in methods:
            selected = select_row(rows, task, method)
            if selected is None:
                table.append(
                    {
                        "Environment": TASK_LABELS[task],
                        "Method": METHOD_LABELS[method],
                        "Budget": "500k",
                        "Return": "PENDING",
                        "Safety": "PENDING",
                        "Hard": "PENDING",
                        "Fairness": "PENDING",
                        "Jain": "PENDING",
                        "Gini": "PENDING",
                        "Run Tag": "",
                    }
                )
                continue
            table.append(
                {
                    "Environment": TASK_LABELS[task],
                    "Method": METHOD_LABELS[method],
                    "Budget": "500k",
                    "Return": fmt(selected["return"], 2),
                    "Safety": fmt(selected["safety"], 2),
                    "Hard": fmt(selected["hard"], 2),
                    "Fairness": fmt(selected["fair"], 2),
                    "Jain": fmt(selected["jain"], 3),
                    "Gini": fmt(selected["gini"], 3),
                    "Run Tag": str(selected.get("run_tag", "")),
                }
            )
    return table


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(rows: List[Dict[str, str]], path: Path) -> None:
    headers = ["Environment", "Method", "Budget", "Return", "Safety", "Hard", "Fairness", "Jain", "Gini"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))
    lines = []
    lines.append("  ".join(f"{h:<{widths[h]}}" for h in headers))
    lines.append("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        cells = []
        for h in headers:
            align = ">" if h not in {"Environment", "Method", "Budget"} else "<"
            cells.append(f"{row[h]:{align}{widths[h]}}")
        lines.append("  ".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")


def write_latex(rows: List[Dict[str, str]], path: Path) -> None:
    lines = [
        "\\begin{tabular}{@{}llcrrrrrr@{}}",
        "\\toprule",
        "Environment & Method & Budget & Return $\\uparrow$ & Safety $\\downarrow$ & Hard $\\downarrow$ & Fairness $\\downarrow$ & Jain $\\uparrow$ & Gini $\\downarrow$\\\\",
        "\\midrule",
    ]
    last_env = None
    for row in rows:
        env = row["Environment"]
        if last_env is not None and env != last_env:
            lines.append("\\midrule")
        last_env = env
        lines.append(
            " & ".join(
                [
                    latex_escape(row["Environment"]),
                    latex_escape(row["Method"]),
                    row["Budget"],
                    row["Return"],
                    row["Safety"],
                    row["Hard"],
                    row["Fairness"],
                    row["Jain"],
                    row["Gini"],
                ]
            )
            + "\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_all(name: str, rows: List[Dict[str, str]]) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(rows, TABLE_ROOT / f"{name}.csv")
    write_text(rows, TABLE_ROOT / f"{name}.txt")
    write_latex(rows, TABLE_ROOT / f"{name}.tex")


def main() -> None:
    rows = load_rows(seed=0)
    standard = build_table(STANDARD_METHODS, rows)
    full = build_table(FULL_METHODS, rows)
    write_all("comparison_500k_standard_seed0", standard)
    write_all("comparison_500k_full_seed0", full)
    missing_standard = sum(1 for row in standard if row["Return"] == "PENDING")
    missing_full = sum(1 for row in full if row["Return"] == "PENDING")
    print(f"wrote {TABLE_ROOT / 'comparison_500k_standard_seed0.txt'} ({missing_standard} pending)")
    print(f"wrote {TABLE_ROOT / 'comparison_500k_full_seed0.txt'} ({missing_full} pending)")
    print((TABLE_ROOT / "comparison_500k_standard_seed0.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
