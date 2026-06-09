"""Tests for shared split integration (train/val/test from Shared_pipeline_Files)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline_integration import partition_rows_from_shared_paths
from src.preprocessing.common import DatasetRow


def _row(rel: str, label: int) -> DatasetRow:
    path = Path(f"/data/{rel}")
    return DatasetRow(apk_path=path, label=label, apk_id=rel.replace("/", "_"))


class TestSharedSplitPartition(unittest.TestCase):
    def test_includes_test_rows_in_index(self) -> None:
        apk_root = Path("/data")
        rows = [
            _row("2020/benign/a.apk", 0),
            _row("2022/benign/b.apk", 0),
            _row("2023/malware/c.apk", 1),
        ]
        train_rows, val_rows, test_rows, indexed = partition_rows_from_shared_paths(
            rows,
            ["2020/benign/a.apk"],
            ["2022/benign/b.apk"],
            apk_root,
            test_paths=["2023/malware/c.apk"],
        )
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(len(val_rows), 1)
        self.assertIsNotNone(test_rows)
        assert test_rows is not None
        self.assertEqual(len(test_rows), 1)
        self.assertEqual({r.apk_id for r in indexed}, {"2020_benign_a.apk", "2022_benign_b.apk", "2023_malware_c.apk"})


if __name__ == "__main__":
    unittest.main()
