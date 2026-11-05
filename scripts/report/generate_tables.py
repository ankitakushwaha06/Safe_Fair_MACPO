#!/usr/bin/env python3
"""Generate publication tables from only the frozen release run manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "release_runs.yaml")
    parser.add_argument("--run-root", type=Path, default=ROOT / "safe_fair_macpo_runs")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "internal" / "regenerate_result_tables.py"),
        "--manifest",
        str(args.manifest),
        "--run-root",
        str(args.run_root),
    ]
    if args.require_complete:
        command.append("--require-complete")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

