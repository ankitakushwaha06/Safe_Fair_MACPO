#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFETY_GYM_ROOT = REPO_ROOT / "vendor" / "safety-gymnasium"

TARGET_REPLACEMENTS = {
    "color: np.array = COLOR['apple']": "color: np.array = field(default_factory=lambda: COLOR['apple'].copy())",
    "group: np.array = GROUP['apple']": "group: np.array = field(default_factory=lambda: GROUP['apple'])",
    "color: np.array = COLOR['button']": "color: np.array = field(default_factory=lambda: COLOR['button'].copy())",
    "group: np.array = GROUP['button']": "group: np.array = field(default_factory=lambda: GROUP['button'])",
    "color: np.array = COLOR['circle']": "color: np.array = field(default_factory=lambda: COLOR['circle'].copy())",
    "group: np.array = GROUP['circle']": "group: np.array = field(default_factory=lambda: GROUP['circle'])",
    "color: np.array = COLOR['goal']": "color: np.array = field(default_factory=lambda: COLOR['goal'].copy())",
    "group: np.array = GROUP['goal']": "group: np.array = field(default_factory=lambda: GROUP['goal'])",
    "color: np.ndarray = COLOR['goal']": "color: np.ndarray = field(default_factory=lambda: COLOR['goal'].copy())",
    "group: np.ndarray = field(default_factory=lambda: GROUP['goal'].copy())": "group: np.ndarray = field(default_factory=lambda: GROUP['goal'])",
    "group: np.ndarray = GROUP['goal']": "group: np.ndarray = field(default_factory=lambda: GROUP['goal'])",
    "color: np.array = COLOR['hazard']": "color: np.array = field(default_factory=lambda: COLOR['hazard'].copy())",
    "group: np.array = GROUP['hazard']": "group: np.array = field(default_factory=lambda: GROUP['hazard'])",
    "color: np.array = COLOR['orange']": "color: np.array = field(default_factory=lambda: COLOR['orange'].copy())",
    "group: np.array = GROUP['orange']": "group: np.array = field(default_factory=lambda: GROUP['orange'])",
    "color: np.array = COLOR['pillar']": "color: np.array = field(default_factory=lambda: COLOR['pillar'].copy())",
    "group: np.array = GROUP['pillar']": "group: np.array = field(default_factory=lambda: GROUP['pillar'])",
    "color: np.array = COLOR['sigwall']": "color: np.array = field(default_factory=lambda: COLOR['sigwall'].copy())",
    "group: np.array = GROUP['sigwall']": "group: np.array = field(default_factory=lambda: GROUP['sigwall'])",
    "color: np.array = COLOR['wall']": "color: np.array = field(default_factory=lambda: COLOR['wall'].copy())",
    "group: np.array = GROUP['wall']": "group: np.array = field(default_factory=lambda: GROUP['wall'])",
    "color: np.array = COLOR['push_box']": "color: np.array = field(default_factory=lambda: COLOR['push_box'].copy())",
    "group: np.array = GROUP['push_box']": "group: np.array = field(default_factory=lambda: GROUP['push_box'])",
    "color: np.array = COLOR['vase']": "color: np.array = field(default_factory=lambda: COLOR['vase'].copy())",
    "group: np.array = GROUP['vase']": "group: np.array = field(default_factory=lambda: GROUP['vase'])",
    "color: np.array = COLOR['gremlin']": "color: np.array = field(default_factory=lambda: COLOR['gremlin'].copy())",
    "group: np.array = GROUP['gremlin']": "group: np.array = field(default_factory=lambda: GROUP['gremlin'])",
}

FIELD_IMPORT_MARKER = "from dataclasses import dataclass"
FIELD_IMPORT_REPLACEMENT = "from dataclasses import dataclass, field"


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    if FIELD_IMPORT_MARKER in text and "from dataclasses import dataclass, field" not in text:
        text = text.replace(FIELD_IMPORT_MARKER, FIELD_IMPORT_REPLACEMENT, 1)
    for old, new in TARGET_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.replace(".copy())", "))")
    text = text.replace("']))", "'])")
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    if not SAFETY_GYM_ROOT.exists():
        raise SystemExit(f"Safety-Gymnasium clone not found: {SAFETY_GYM_ROOT}")
    changed = 0
    for rel in [
        "safety_gymnasium/assets/geoms/apples.py",
        "safety_gymnasium/assets/geoms/buttons.py",
        "safety_gymnasium/assets/geoms/circle.py",
        "safety_gymnasium/assets/geoms/goal.py",
        "safety_gymnasium/assets/geoms/hazards.py",
        "safety_gymnasium/assets/geoms/oranges.py",
        "safety_gymnasium/assets/geoms/pillars.py",
        "safety_gymnasium/assets/geoms/sigwalls.py",
        "safety_gymnasium/assets/geoms/walls.py",
        "safety_gymnasium/assets/free_geoms/push_box.py",
        "safety_gymnasium/assets/free_geoms/vases.py",
        "safety_gymnasium/assets/mocaps/gremlins.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/apples.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/buttons.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/circle.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/goal.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/hazards.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/oranges.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/pillars.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/sigwalls.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/geoms/walls.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/free_geoms/push_box.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/free_geoms/vases.py",
        "safety_gymnasium/tasks/safe_multi_agent/assets/mocaps/gremlins.py",
    ]:
        changed += int(patch_file(SAFETY_GYM_ROOT / rel))
    print(f"Patched {changed} Safety-Gymnasium files for mutable dataclass defaults.")


if __name__ == "__main__":
    main()
