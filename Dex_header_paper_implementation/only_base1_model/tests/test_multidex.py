"""Unit tests for multi-Dex discovery and aggregation."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import DEX_HEADER_SIZE
from src.features.dex_header import FEATURE_DIM, extract_header_features
from src.features.multidex import (
    MultidexError,
    aggregate_header_vectors,
    dex_suffix_sort_key,
)
from src.preprocessing.apk_extract import (
    ApkExtractError,
    extract_apk_header_extraction,
    extract_apk_raw_header,
    list_dex_entries,
    read_all_dex_from_apk,
)


def _build_synthetic_dex(seed: int = 0) -> bytes:
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


class TestMultidex(unittest.TestCase):
    def test_dex_suffix_sort_key(self) -> None:
        self.assertLess(
            dex_suffix_sort_key("classes.dex"),
            dex_suffix_sort_key("classes2.dex"),
        )
        self.assertLess(
            dex_suffix_sort_key("classes2.dex"),
            dex_suffix_sort_key("classes10.dex"),
        )

    def test_list_dex_entries_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            apk = Path(tmp) / "multi.apk"
            _write_apk(
                apk,
                {
                    "classes3.dex": _build_synthetic_dex(3),
                    "classes.dex": _build_synthetic_dex(1),
                    "classes2.dex": _build_synthetic_dex(2),
                },
            )
            with zipfile.ZipFile(apk, "r") as zf:
                names = list_dex_entries(zf)
            basenames = [Path(n).name for n in names]
            self.assertEqual(basenames, ["classes.dex", "classes2.dex", "classes3.dex"])

    def test_list_dex_entries_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            apk = Path(tmp) / "nested.apk"
            _write_apk(apk, {"foo/classes2.dex": _build_synthetic_dex(1)})
            with zipfile.ZipFile(apk, "r") as zf:
                names = list_dex_entries(zf)
            self.assertEqual(len(names), 1)
            self.assertEqual(Path(names[0]).name, "classes2.dex")

    def test_sum_equals_manual(self) -> None:
        v0 = extract_header_features(_build_synthetic_dex(0))
        v1 = extract_header_features(_build_synthetic_dex(1))
        result = aggregate_header_vectors([v0, v1], "sum")
        np.testing.assert_allclose(result, v0 + v1)

    def test_single_dex_sum_is_identity(self) -> None:
        vec = extract_header_features(_build_synthetic_dex(5))
        summed = aggregate_header_vectors([vec], "sum")
        np.testing.assert_allclose(summed, vec)

    def test_empty_vectors_raises(self) -> None:
        with self.assertRaises(MultidexError):
            aggregate_header_vectors([], "sum")

    def test_extract_apk_raw_header_multi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            apk = Path(tmp) / "multi.apk"
            d0 = _build_synthetic_dex(0)
            d1 = _build_synthetic_dex(1)
            _write_apk(apk, {"classes.dex": d0, "classes2.dex": d1})
            header = extract_apk_raw_header(apk, mode="sum")
            manual = extract_header_features(d0) + extract_header_features(d1)
            np.testing.assert_allclose(header, manual)
            self.assertEqual(header.shape, (FEATURE_DIM,))

    def test_extract_apk_header_extraction_reports_dex_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            apk = Path(tmp) / "multi.apk"
            _write_apk(
                apk,
                {
                    "classes.dex": _build_synthetic_dex(0),
                    "classes2.dex": _build_synthetic_dex(1),
                    "classes3.dex": _build_synthetic_dex(2),
                },
            )
            result = extract_apk_header_extraction(apk, mode="sum")
            self.assertEqual(result.num_dex_files, 3)
            self.assertEqual(result.vector.shape, (FEATURE_DIM,))

    def test_extract_apk_raw_header_single(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            apk = Path(tmp) / "single.apk"
            d0 = _build_synthetic_dex(0)
            _write_apk(apk, {"classes.dex": d0})
            header = extract_apk_raw_header(apk, mode="sum")
            np.testing.assert_allclose(header, extract_header_features(d0))

    def test_no_dex_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            apk = Path(tmp) / "empty.apk"
            with zipfile.ZipFile(apk, "w") as zf:
                zf.writestr("readme.txt", b"no dex")
            with self.assertRaises(ApkExtractError):
                read_all_dex_from_apk(apk)

    def test_mean_mode(self) -> None:
        v0 = extract_header_features(_build_synthetic_dex(0))
        v1 = extract_header_features(_build_synthetic_dex(1))
        mean = aggregate_header_vectors([v0, v1], "mean")
        np.testing.assert_allclose(mean, (v0 + v1) / 2.0)

    def test_primary_only_mode(self) -> None:
        v0 = extract_header_features(_build_synthetic_dex(0))
        v1 = extract_header_features(_build_synthetic_dex(1))
        primary = aggregate_header_vectors([v0, v1], "primary_only")
        np.testing.assert_allclose(primary, v0)

    def test_concat_mode(self) -> None:
        v0 = extract_header_features(_build_synthetic_dex(0))
        v1 = extract_header_features(_build_synthetic_dex(1))
        concat = aggregate_header_vectors([v0, v1], "concat", max_dex=3)
        self.assertEqual(concat.shape, (FEATURE_DIM * 3,))
        np.testing.assert_allclose(concat[:FEATURE_DIM], v0)
        np.testing.assert_allclose(concat[FEATURE_DIM : 2 * FEATURE_DIM], v1)
        self.assertTrue(np.all(concat[2 * FEATURE_DIM :] == 0))


if __name__ == "__main__":
    unittest.main()
