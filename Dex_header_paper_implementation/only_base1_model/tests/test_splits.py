"""Tests for temporal year and random index splits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.splits import temporal_three_way_split_indices, year_from_apk_path


class TestTemporalYearSplit(unittest.TestCase):
    def test_year_from_path(self) -> None:
        self.assertEqual(year_from_apk_path(Path("/mnt/ds/2021/malware/x.apk")), "2021")
        self.assertIsNone(year_from_apk_path(Path("/no/year/here.apk")))

    def test_three_way_split(self) -> None:
        paths = [
            "/data/2020/benign/a.apk",
            "/data/2020/malware/b.apk",
            "/data/2021/benign/c.apk",
            "/data/2021/malware/d.apk",
            "/data/2022/benign/e.apk",
            "/data/2023/malware/f.apk",
        ]
        labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float)
        train_idx, val_idx, test_idx = temporal_three_way_split_indices(
            paths,
            labels,
            train_years=[2020, 2021],
            test_years=[2022, 2023],
            val_fraction=0.5,
            seed=42,
        )
        self.assertEqual(set(train_idx.tolist()), {0, 1, 2, 3})
        self.assertEqual(sorted(val_idx.tolist() + test_idx.tolist()), [4, 5])
        self.assertEqual(len(val_idx) + len(test_idx), 2)

    def test_rejects_overlap(self) -> None:
        with self.assertRaises(ValueError):
            temporal_three_way_split_indices(
                ["/data/2020/a.apk"],
                torch.tensor([0.0]),
                train_years=[2020],
                test_years=[2020],
            )

    def test_rejects_unassigned_years(self) -> None:
        with self.assertRaises(ValueError):
            temporal_three_way_split_indices(
                ["/data/2019/a.apk"],
                torch.tensor([0.0]),
                train_years=[2020],
                test_years=[2022],
            )


class TestTemporalSplitOnBundle(unittest.TestCase):
    def test_resolve_split_indices(self) -> None:
        from src.data.dataloaders import resolve_split_indices
        from src.data.store import ProcessedBundle

        paths = [f"/ds/2020/benign/{i}.apk" for i in range(3)] + [
            f"/ds/2023/malware/{i}.apk" for i in range(2)
        ]
        bundle = ProcessedBundle(
            features=torch.zeros(len(paths), 4),
            labels=torch.tensor([0, 0, 0, 1, 1], dtype=torch.float),
            paths=paths,
            feature_dim=4,
            source_path=Path("x.pt"),
        )
        train_idx, val_idx, test_idx = resolve_split_indices(
            bundle,
            split_mode="temporal_holdout",
            train_years=[2020],
            test_years=[2023],
            val_fraction=0.5,
            seed=42,
        )
        self.assertEqual(train_idx.numel(), 3)
        self.assertEqual(val_idx.numel() + test_idx.numel(), 2)


if __name__ == "__main__":
    unittest.main()
