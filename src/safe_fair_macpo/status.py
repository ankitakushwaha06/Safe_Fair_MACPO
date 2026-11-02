from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from .backend import DEFAULT_RUN_ROOT


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_run_root(run_root: Path = DEFAULT_RUN_ROOT) -> Dict[str, object]:
    manifest_rows = load_csv_rows(run_root / "configs" / "experiment_manifest.csv")
    availability_rows = load_csv_rows(run_root / "environment_availability.csv")
    manifest_path = run_root / "reproducibility_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "run_root": str(run_root),
        "configured_runs": len(manifest_rows),
        "available_tasks": sum(1 for row in availability_rows if row.get("available") == "True"),
        "checked_tasks": len(availability_rows),
        "manifest_created_at": manifest.get("created_at", "missing"),
        "device": manifest.get("device", "unknown"),
        "torch": manifest.get("torch", "unknown"),
    }


def format_status(run_root: Path = DEFAULT_RUN_ROOT) -> str:
    """Return a human-readable summary of artifacts in ``run_root``."""
    summary = summarize_run_root(run_root)
    lines = [
        f"- Run root: `{summary['run_root']}`",
        f"- Configured runs in current manifest: `{summary['configured_runs']}`",
        f"- Environment checks passing: `{summary['available_tasks']}` / `{summary['checked_tasks']}`",
        f"- Manifest timestamp: `{summary['manifest_created_at']}`",
        f"- Backend device: `{summary['device']}`",
        f"- Torch version seen by backend: `{summary['torch']}`",
    ]
    return "\n".join(lines)
