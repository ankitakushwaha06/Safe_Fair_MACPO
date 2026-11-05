#!/usr/bin/env python3
"""Generate publication frontier figures from manifest-selected tables."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=ROOT / "safe_fair_macpo_runs")
    args = parser.parse_args()
    table = args.run_root / "tables" / "main_results.csv"
    if not table.exists():
        raise SystemExit(f"missing {table}; run scripts/report/generate_tables.py first")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "codes" / "plot_safe_fair_frontiers.py"),
            "--inputs",
            str(table),
            "--out-dir",
            str(args.run_root / "figures" / "frontiers"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

