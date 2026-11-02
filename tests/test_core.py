from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from safe_fair_macpo.backend import backend_env, prepend_env_path
from safe_fair_macpo.experiments import PAPER_EXPERIMENTS, format_experiment_matrix
from safe_fair_macpo.status import format_status, summarize_run_root


class CoreTests(unittest.TestCase):
    def test_prepend_env_path_deduplicates_existing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()

            value = prepend_env_path(f"{second}:{first}", first, second)

            self.assertEqual(value.split(":"), [str(first), str(second)])

    def test_backend_env_is_offline_and_non_training_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            env = backend_env(run_root=run_root, quick=True, run_training=False)

            self.assertEqual(env["SAFE_FAIR_MACPO_ROOT"], str(run_root))
            self.assertEqual(env["SAFE_FAIR_QUICK_TEST"], "1")
            self.assertEqual(env["SAFE_FAIR_RUN_TRAINING"], "0")
            self.assertEqual(env["SAFE_FAIR_CLONE_OFFICIAL_REPOS"], "0")

    def test_experiment_matrix_contains_every_group(self) -> None:
        table = format_experiment_matrix()

        self.assertTrue(table.startswith("| Group |"))
        self.assertTrue(all(f"| {group.code} |" in table for group in PAPER_EXPERIMENTS))

    def test_status_handles_empty_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            summary = summarize_run_root(run_root)

            self.assertEqual(summary["configured_runs"], 0)
            self.assertEqual(summary["checked_tasks"], 0)
            self.assertIn("missing", format_status(run_root))

    def test_status_reads_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            config_dir = run_root / "configs"
            config_dir.mkdir()
            (config_dir / "experiment_manifest.csv").write_text("name\na\nb\n", encoding="utf-8")
            (run_root / "environment_availability.csv").write_text(
                "task,available\na,True\nb,False\n", encoding="utf-8"
            )
            (run_root / "reproducibility_manifest.json").write_text(
                json.dumps({"created_at": "now", "device": "cpu", "torch": "test"}),
                encoding="utf-8",
            )

            summary = summarize_run_root(run_root)

            self.assertEqual(summary["configured_runs"], 2)
            self.assertEqual(summary["available_tasks"], 1)
            self.assertEqual(summary["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
