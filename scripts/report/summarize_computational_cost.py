#!/usr/bin/env python3
"""Summarize wall-clock and step cost for frozen publication runs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codes.release_runs import collect_selected_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "release_runs.yaml")
    parser.add_argument("--run-root", type=Path, default=ROOT / "safe_fair_macpo_runs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    output = args.out or args.run_root / "tables" / "computational_cost.csv"
    rows: List[Dict[str, Any]] = []
    for _, config, summary in collect_selected_runs(args.run_root / "logs", args.manifest):
        task = config.get("task", {})
        method = config.get("method", {})
        rows.append(
            {
                "task": task.get("name", "") if isinstance(task, dict) else task,
                "method": method.get("name", "") if isinstance(method, dict) else method,
                "seed": config.get("seed", ""),
                "run_group": config.get("run_group", ""),
                "run_tag": config.get("run_tag", ""),
                "total_steps": config.get("total_steps", summary.get("global_step", "")),
                "wallclock_seconds": summary.get("train_wallclock_sec", ""),
                "device": config.get("device", ""),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "method", "seed", "run_group", "run_tag", "total_steps",
        "wallclock_seconds", "device",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} rows={len(rows)}")


if __name__ == "__main__":
    main()

