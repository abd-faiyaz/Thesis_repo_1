"""Phase 7: multi-dex discovery and aggregation (already default in Phase 2)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DEX_HEADER_FEATURE_DIM, DEX_HEADER_SIZE
from src.features.apk_extract import extract_apk_raw_header
from src.features.dex_header import extract_header_features
from src.features.multidex import aggregate_header_vectors, multidex_settings


def _build_synthetic_dex(seed: int = 0) -> bytes:
    import struct

    header = bytearray(DEX_HEADER_SIZE)
    header[0:8] = b"dex\n035\x00"
    struct.pack_into("<I", header, 8, 0xDEADBEEF + seed)
    header[12:32] = bytes((i + seed) % 256 for i in range(20))
    struct.pack_into("<I", header, 32, DEX_HEADER_SIZE)
    struct.pack_into("<I", header, 36, DEX_HEADER_SIZE)
    struct.pack_into("<I", header, 40, 0x12345678)
    struct.pack_into("<I", header, 52, 0x100)
    struct.pack_into("<I", header, 56, 10)
    struct.pack_into("<I", header, 60, 0x200)
    return bytes(header)


def _write_apk(path: Path, dex_entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in dex_entries.items():
            zf.writestr(name, data)


class TestMultidexPhase7Policy(unittest.TestCase):
    def test_config_default_is_sum(self) -> None:
        cfg = load_config()
        md = multidex_settings(cfg.preprocessing)
        self.assertEqual(md["mode"], "sum")

    def test_sum_matches_paper_blind_spot_fix(self) -> None:
        """classes2.dex-only signal must affect aggregated H when summed."""
        with tempfile.TemporaryDirectory() as tmp:
            apk = Path(tmp) / "multi.apk"
            d0 = _build_synthetic_dex(0)
            d1 = _build_synthetic_dex(1)
            _write_apk(apk, {"classes.dex": d0, "classes2.dex": d1})
            header = extract_apk_raw_header(apk, mode="sum")
            manual = extract_header_features(d0) + extract_header_features(d1)
            np.testing.assert_allclose(header, manual)
            self.assertEqual(header.shape, (DEX_HEADER_FEATURE_DIM,))

    def test_primary_only_differs_from_sum(self) -> None:
        v0 = extract_header_features(_build_synthetic_dex(0))
        v1 = extract_header_features(_build_synthetic_dex(1))
        primary = aggregate_header_vectors([v0, v1], "primary_only")
        summed = aggregate_header_vectors([v0, v1], "sum")
        self.assertFalse(np.allclose(primary, summed))


if __name__ == "__main__":
    unittest.main()
