from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codes.release_runs import collect_selected_runs, expand_selections, load_manifest


class ReleaseRunTests(unittest.TestCase):
    def test_release_manifest_expands_all_declared_ablation_seeds(self) -> None:
        rows = expand_selections(load_manifest())
        ablations = [row for row in rows if row["run_group"] == "safefairbench_ablation"]
        self.assertTrue(ablations)
        self.assertEqual({row["seed"] for row in ablations}, {0, 1, 2, 3, 4})

    def test_selection_uses_exact_tag_not_prefix_or_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "version: 1\nselected_runs:\n"
                "  - {task: task, method: method, seed: 0, run_group: main, run_tag: exact}\n",
                encoding="utf-8",
            )
            for tag in ("exact", "exact_newer"):
                run = root / "logs" / tag
                run.mkdir(parents=True)
                config = {
                    "task": {"name": "task"},
                    "method": {"name": "method"},
                    "seed": 0,
                    "run_group": "main",
                    "run_tag": tag,
                }
                (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
                (run / "eval_summary.json").write_text("{}", encoding="utf-8")
            selected = collect_selected_runs(root / "logs", manifest)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0][1]["run_tag"], "exact")


if __name__ == "__main__":
    unittest.main()

