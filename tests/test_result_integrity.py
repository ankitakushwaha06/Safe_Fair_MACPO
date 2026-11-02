from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from codes.audit_result_integrity import display_path, enrich_table_rows, normalize_table


class ResultIntegrityTests(unittest.TestCase):
    def test_external_table_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "results.csv"
            table.write_text("task,method,return_mean\nsmoke,ippo,1.5\n", encoding="utf-8")

            normalized = normalize_table(table)

            self.assertEqual(len(normalized), 1)
            self.assertEqual(normalized.iloc[0]["source_table"], str(table.resolve()))
            enriched = enrich_table_rows(normalized, pd.DataFrame())
            self.assertEqual(len(enriched), 1)

    def test_repository_path_is_displayed_relative_to_root(self) -> None:
        source = Path(__file__).resolve()
        displayed = display_path(source)

        self.assertFalse(Path(displayed).is_absolute())
        self.assertTrue(displayed.endswith("tests/test_result_integrity.py"))


if __name__ == "__main__":
    unittest.main()
