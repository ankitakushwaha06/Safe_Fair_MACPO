#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[1] / "safe_fair_macpo_runs"
METRICS = [
    "return_mean",
    "safety_cost_mean",
    "hard_violation_rate",
    "fairness_cost_mean",
    "jain_burden_mean",
    "gini_burden_mean",
    "envy_gap_mean",
    "debt_auc_mean",
    "safety_recovery_rate_mean",
    "safety_recovery_delta_mean",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def matching_manifest_rows(rows: Iterable[Dict[str, str]], run_group: str) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("run_group") == run_group and row.get("task", "").startswith("safefairbench_")
    ]


def collect_eval_rows(run_root: Path, run_group: str) -> List[Dict[str, object]]:
    latest: Dict[tuple, Dict[str, object]] = {}
    group_dir = run_root / "logs" / run_group
    if not group_dir.exists():
        return []
    for path in sorted(group_dir.rglob("eval_summary.json")):
        try:
            row = read_json(path)
        except Exception as exc:
            row = {"task": path.parent.parent.parent.name, "method": path.parent.parent.name, "status": f"read-error:{exc}"}
        row["run_dir"] = str(path.parent)
        row["_mtime"] = path.stat().st_mtime
        key = (str(row.get("task", "")), str(row.get("method", "")), str(row.get("seed", "")))
        if key not in latest or float(row["_mtime"]) > float(latest[key].get("_mtime", 0.0)):
            latest[key] = row
    rows = list(latest.values())
    rows.sort(key=lambda r: (str(r.get("task", "")), str(r.get("method", "")), int(r.get("seed", 0))))
    return rows


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = ["task", "method", "seed", "run_tag", *METRICS]
    rendered = []
    for row in rows:
        rendered.append([fmt(row.get(header)) for header in headers])
    widths = [len(header) for header in headers]
    for row in rendered:
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]
    line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rendered:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SafeFairBench-v0 queue status.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-group", default="safefairbench")
    args = parser.parse_args()

    manifest_rows = matching_manifest_rows(read_csv(args.run_root / "configs" / "experiment_manifest.csv"), args.run_group)
    eval_rows = collect_eval_rows(args.run_root, args.run_group)
    completed_keys = {
        (str(row.get("task")), str(row.get("method")), str(row.get("seed")))
        for row in eval_rows
    }
    expected_keys = {
        (row.get("task", ""), row.get("method", ""), row.get("seed", ""))
        for row in manifest_rows
    }
    remaining = max(0, len(expected_keys - completed_keys)) if expected_keys else None
    completed = len(completed_keys & expected_keys) if expected_keys else len(eval_rows)
    avg_wallclock = 0.0
    wallclocks = [float(row.get("train_wallclock_sec", 0.0) or 0.0) for row in eval_rows]
    wallclocks = [x for x in wallclocks if x > 0.0]
    if wallclocks:
        avg_wallclock = sum(wallclocks) / len(wallclocks)

    print(f"Run group: {args.run_group}")
    print(f"Completed rows: {completed}")
    if remaining is None:
        print("Remaining configured rows: unknown; current manifest has no SafeFairBench rows")
    else:
        print(f"Remaining configured rows: {remaining}")
    if avg_wallclock and remaining:
        eta_min = remaining * avg_wallclock / 60.0
        print(f"ETA from completed-row average: {eta_min:.1f} min")
    if eval_rows:
        print()
        print_table(eval_rows)
    else:
        print("No completed eval summaries yet.")


if __name__ == "__main__":
    main()
