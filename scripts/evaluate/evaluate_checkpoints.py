#!/usr/bin/env python3
"""Camera-ready checkpoint evaluation with protocol-explicit defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "evaluation_full.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--stress-modes", nargs="+", default=["standard", "safety_stress", "fairness_stress"])
    parser.add_argument("--out", type=Path, default=ROOT / "safe_fair_macpo_runs" / "tables" / "checkpoint_evaluation.csv")
    args = parser.parse_args()
    command = [
        sys.executable,
        str(ROOT / "codes" / "eval_many_seeds.py"),
        "--manifest", str(args.manifest),
        "--episodes", str(args.episodes),
        "--eval-seeds", *map(str, args.eval_seeds),
        "--stress-modes", *args.stress_modes,
        "--out", str(args.out),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
