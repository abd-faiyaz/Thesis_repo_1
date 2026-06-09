"""Tests for unified temporal holdout splits."""

from __future__ import annotations

import unittest

import torch

from shared_splits.temporal import (
    crosscheck_temporal_holdout,
    temporal_holdout_partition,
    temporal_holdout_split_indices,
    year_from_apk_path,
)


class TemporalHoldoutSplitTest(unittest.TestCase):
    def test_year_from_path(self) -> None:
        self.assertEqual(year_from_apk_path("/mnt/ds/2021/malware/x.apk"), "2021")
        self.assertIsNone(year_from_apk_path("/no/year/here.apk"))

    def test_holdout_partition(self) -> None:
        paths = [
            "/data/2020/benign/a.apk",
            "/data/2020/malware/b.apk",
            "/data/2021/benign/c.apk",
            "/data/2021/malware/d.apk",
            "/data/2022/benign/e.apk",
            "/data/2023/malware/f.apk",
        ]
        train, val, test, other = temporal_holdout_partition(
            paths,
            [0, 1, 0, 1, 0, 1],
            get_year=lambda p: year_from_apk_path(p),
            val_fraction_of_holdout=0.5,
            seed=42,
        )
        self.assertEqual(set(train), set(paths[:4]))
        self.assertEqual(set(val) | set(test), set(paths[4:]))
        self.assertEqual(set(val) & set(test), set())
        self.assertEqual(other, [])

    def test_split_indices(self) -> None:
        paths = [
            "/data/2020/benign/a.apk",
            "/data/2020/malware/b.apk",
            "/data/2022/benign/c.apk",
            "/data/2023/malware/d.apk",
        ]
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.float)
        train_idx, val_idx, test_idx = temporal_holdout_split_indices(
            paths,
            labels,
            train_years=[2020],
            holdout_years=[2022, 2023],
            val_fraction_of_holdout=0.5,
            seed=42,
        )
        self.assertEqual(train_idx.tolist(), [0, 1])
        self.assertEqual(sorted(val_idx.tolist() + test_idx.tolist()), [2, 3])
        self.assertEqual(len(val_idx) + len(test_idx), 2)

    def test_crosscheck(self) -> None:
        rows = [
            {"split": "train", "year": 2020, "path": "a"},
            {"split": "val", "year": 2022, "path": "b"},
            {"split": "test", "year": 2021, "path": "c"},
        ]
        errors = crosscheck_temporal_holdout(
            rows,
            get_split=lambda r: r["split"],
            get_year=lambda r: r["year"],
            get_path=lambda r: r["path"],
        )
        self.assertGreaterEqual(len(errors), 1)
        self.assertTrue(any("test split contains train year" in msg for msg in errors))


if __name__ == "__main__":
    unittest.main()
