"""Integration tests for multi-Dex preprocessing and metadata."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import DEX_HEADER_SIZE
from src.features.dex_header import FEATURE_DIM
from src.preprocessing.preprocess_apks import preprocess


def _build_synthetic_dex(seed: int = 0) -> bytes:
    header = bytearray(DEX_HEADER_SIZE)
    header[0:8] = b"dex\n035\x00"
    struct.pack_into("<I", header, 8, 0xDEADBEEF + seed)
    header[12:32] = bytes((i + seed) % 256 for i in range(20))
    struct.pack_into("<I", header, 32, DEX_HEADER_SIZE)
    struct.pack_into("<I", header, 36, DEX_HEADER_SIZE)
    return bytes(header)


class TestMultidexPreprocess(unittest.TestCase):
    def test_preprocess_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benign").mkdir()
            (root / "malware").mkdir()

            with zipfile.ZipFile(root / "benign" / "multi.apk", "w") as zf:
                zf.writestr("classes.dex", _build_synthetic_dex(0))
                zf.writestr("classes2.dex", _build_synthetic_dex(1))
            with zipfile.ZipFile(root / "malware" / "single.apk", "w") as zf:
                zf.writestr("classes.dex", _build_synthetic_dex(2))

            processed_dir = root / "processed"
            norm_path = root / "normalization.json"
            summary = preprocess(
                apk_root=root,
                processed_dir=processed_dir,
                failed_log=root / "failed.log",
                normalization_stats_path=norm_path,
                multidex={
                    "mode": "sum",
                    "dex_pattern": r"^classes(\d*)\.dex$",
                    "max_dex": 3,
                },
                cache_version=2,
                output_format="pt",
                aggregate_filename="features.pt",
                label_mode="parent_folder",
                labels_csv=None,
                benign_names={"benign"},
                malicious_names={"malware"},
            )

            self.assertEqual(summary["successful"], 2)
            self.assertEqual(summary["feature_dim"], FEATURE_DIM)
            self.assertEqual(summary["multidex_mode"], "sum")
            self.assertEqual(summary["cache_version"], 2)
            self.assertEqual(summary["dex_file_counts"], {"1": 1, "2": 1})

            norm = json.loads(norm_path.read_text())
            self.assertEqual(norm["multidex_mode"], "sum")
            self.assertEqual(norm["cache_version"], 2)
            self.assertEqual(norm["dex_file_counts"], {"1": 1, "2": 1})

            bundle = torch.load(processed_dir / "features.pt", weights_only=False)
            self.assertEqual(bundle["features"].shape, (2, FEATURE_DIM))
            self.assertEqual(bundle["multidex_mode"], "sum")
            self.assertEqual(bundle["cache_version"], 2)
            self.assertEqual(bundle["dex_file_counts"], {"1": 1, "2": 1})


if __name__ == "__main__":
    unittest.main()
