"""Tests for temporal year and random index splits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.splits import temporal_split_indices, year_from_apk_path


class TestTemporalYearSplit(unittest.TestCase):
    def test_year_from_path(self) -> None:
        self.assertEqual(year_from_apk_path(Path("/mnt/ds/2021/malware/x.apk")), "2021")
        self.assertIsNone(year_from_apk_path(Path("/no/year/here.apk")))

    def test_split_by_year(self) -> None:
        paths = [
            "/data/2020/benign/a.apk",
            "/data/2021/malware/b.apk",
            "/data/2022/benign/c.apk",
            "/data/2023/malware/d.apk",
        ]
        train_idx, val_idx = temporal_split_indices(
            paths, train_years=[2020, 2021], val_years=[2022, 2023]
        )
        self.assertEqual(train_idx.tolist(), [0, 1])
        self.assertEqual(val_idx.tolist(), [2, 3])

    def test_rejects_overlap(self) -> None:
        with self.assertRaises(ValueError):
            temporal_split_indices(
                ["/data/2020/a.apk"],
                train_years=[2020],
                val_years=[2020],
            )

    def test_rejects_unassigned_years(self) -> None:
        with self.assertRaises(ValueError):
            temporal_split_indices(
                ["/data/2019/a.apk"],
                train_years=[2020],
                val_years=[2022],
            )


class TestTemporalSplitOnBundle(unittest.TestCase):
    def test_resolve_train_val_indices(self) -> None:
        from src.data.dataloaders import resolve_train_val_indices
        from src.data.store import ProcessedBundle

        paths = [f"/ds/2020/benign/{i}.apk" for i in range(3)] + [
            f"/ds/2023/malware/{i}.apk" for i in range(2)
        ]
        bundle = ProcessedBundle(
            features=torch.zeros(len(paths), 4),
            labels=torch.zeros(len(paths)),
            paths=paths,
            feature_dim=4,
            source_path=Path("x.pt"),
        )
        train_idx, val_idx = resolve_train_val_indices(
            bundle,
            split_mode="temporal_year",
            train_years=[2020],
            val_years=[2023],
        )
        self.assertEqual(train_idx.numel(), 3)
        self.assertEqual(val_idx.numel(), 2)


if __name__ == "__main__":
    unittest.main()
