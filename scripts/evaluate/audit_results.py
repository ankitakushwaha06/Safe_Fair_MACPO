#!/usr/bin/env python3
"""Recompute burden summaries and audit provenance before reporting."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    for script in ("recompute_burden_summaries.py", "audit_result_integrity.py"):
        subprocess.run([sys.executable, str(ROOT / "codes" / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
