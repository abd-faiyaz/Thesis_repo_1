"""Tests for dataset split helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.common import DatasetRow, temporal_year_split, year_from_apk_path


def _row(year: str, label: int, name: str) -> DatasetRow:
    path = Path(f"/data/{year}/benign/{name}.apk")
    return DatasetRow(apk_path=path, label=label, apk_id=name)


class TestTemporalYearSplit(unittest.TestCase):
    def test_year_from_path(self) -> None:
        self.assertEqual(year_from_apk_path(Path("/mnt/ds/2021/malware/x.apk")), "2021")
        self.assertIsNone(year_from_apk_path(Path("/no/year/here.apk")))

    def test_split_by_year(self) -> None:
        rows = [
            _row("2020", 0, "a"),
            _row("2021", 1, "b"),
            _row("2022", 0, "c"),
            _row("2023", 1, "d"),
        ]
        train, val = temporal_year_split(
            rows, train_years=[2020, 2021], val_years=[2022, 2023]
        )
        self.assertEqual({r.apk_id for r in train}, {"a", "b"})
        self.assertEqual({r.apk_id for r in val}, {"c", "d"})


if __name__ == "__main__":
    unittest.main()
