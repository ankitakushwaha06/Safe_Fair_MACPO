"""Frozen-run selection shared by camera-ready evaluators and reporters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "release_runs.yaml"


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("PyYAML is required to read the release manifest") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if int(data.get("version", 0)) != 1:
        raise ValueError(f"unsupported release manifest version in {path}")
    return data


def _name(value: Any) -> str:
    return str(value.get("name", "")) if isinstance(value, Mapping) else str(value or "")


def run_identity(config: Mapping[str, Any]) -> Tuple[str, str, int, str, str]:
    return (
        _name(config.get("task")),
        _name(config.get("method")),
        int(config.get("seed", 0)),
        str(config.get("run_group", "")),
        str(config.get("run_tag", "")),
    )


def expand_selections(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    selections: List[Dict[str, Any]] = []
    for item in manifest.get("selected_runs", []):
        seeds: Sequence[int] = item.get("seeds", [item.get("seed", 0)])
        tags: Sequence[str]
        if "run_tags" in item:
            tags = [str(tag) for tag in item["run_tags"]]
            if len(tags) != len(seeds):
                raise ValueError("run_tags and seeds must have the same length")
        else:
            template = str(item["run_tag"])
            tags = [template.format(seed=int(seed)) for seed in seeds]
        for seed, tag in zip(seeds, tags):
            row = dict(item)
            row.pop("seeds", None)
            row.pop("run_tags", None)
            row.update(seed=int(seed), run_tag=tag)
            selections.append(row)
    for matrix in manifest.get("selected_matrices", []):
        for task in matrix["tasks"]:
            for method in matrix["methods"]:
                for seed in matrix.get("seeds", [0]):
                    selections.append(
                        {
                            "task": str(task),
                            "method": str(method),
                            "seed": int(seed),
                            "run_group": str(matrix["run_group"]),
                            "run_tag": str(matrix["run_tag"]).format(
                                task=task, method=method, seed=int(seed)
                            ),
                        }
                    )
    return selections


def selection_identity(item: Mapping[str, Any]) -> Tuple[str, str, int, str, str]:
    return (
        str(item.get("task", "")),
        str(item.get("method", "")),
        int(item.get("seed", 0)),
        str(item.get("run_group", "")),
        str(item.get("run_tag", "")),
    )


def collect_selected_runs(
    log_root: Path, manifest_path: Path = DEFAULT_MANIFEST
) -> List[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    """Return only exact manifest matches; reject duplicate identities."""
    manifest = load_manifest(manifest_path)
    wanted = {selection_identity(item): item for item in expand_selections(manifest)}
    found: Dict[Tuple[str, str, int, str, str], Tuple[Path, Dict[str, Any], Dict[str, Any]]] = {}
    for config_path in sorted(log_root.rglob("config.json")):
        eval_path = config_path.parent / "eval_summary.json"
        if not eval_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        identity = run_identity(config)
        if identity not in wanted:
            continue
        if identity in found:
            first = found[identity][0]
            raise RuntimeError(
                f"duplicate artifact for frozen run {identity}: {first.parent} and {config_path.parent}"
            )
        summary.setdefault("task", identity[0])
        summary.setdefault("method", identity[1])
        summary.setdefault("seed", identity[2])
        summary.setdefault("run_group", identity[3])
        summary.setdefault("run_tag", identity[4])
        summary["_path"] = str(eval_path)
        found[identity] = (config_path, config, summary)
    return [found[key] for key in sorted(found)]


def missing_identities(
    rows: Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any]]],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> List[Tuple[str, str, int, str, str]]:
    wanted = {
        selection_identity(item)
        for item in expand_selections(load_manifest(manifest_path))
    }
    present = {run_identity(config) for _, config, _ in rows}
    return sorted(wanted - present)
