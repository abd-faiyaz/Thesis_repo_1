"""Unit tests for Dex header parsing (no APK required)."""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import DEX_HEADER_FEATURE_DIM, DEX_HEADER_SIZE, DEX_MAGIC_LEN
from src.features.dex_header import (
    DexHeaderError,
    extract_header_features,
    extract_headers_from_dex_list,
    parse_dex_header_fields,
    validate_magic,
)
from src.features.multidex import aggregate_header_vectors
from src.features.normalization import fit_minmax, transform_minmax


def _build_synthetic_dex() -> bytes:
    header = bytearray(DEX_HEADER_SIZE)
    header[0:8] = b"dex\n035\x00"
    struct.pack_into("<I", header, 8, 0xDEADBEEF)
    header[12:32] = bytes(range(20))
    struct.pack_into("<I", header, 32, DEX_HEADER_SIZE)
    struct.pack_into("<I", header, 36, DEX_HEADER_SIZE)
    struct.pack_into("<I", header, 40, 0x12345678)
    struct.pack_into("<I", header, 52, 0x100)
    struct.pack_into("<I", header, 56, 10)
    struct.pack_into("<I", header, 60, 0x200)
    return bytes(header)


class TestDexHeader(unittest.TestCase):
    def setUp(self) -> None:
        self.dex = _build_synthetic_dex()

    def test_magic_validation(self) -> None:
        self.assertTrue(validate_magic(self.dex))
        self.assertFalse(validate_magic(b"notadex"))

    def test_feature_vector_shape(self) -> None:
        vec = extract_header_features(self.dex)
        self.assertEqual(vec.shape, (DEX_HEADER_FEATURE_DIM,))
        self.assertEqual(DEX_HEADER_FEATURE_DIM, DEX_HEADER_SIZE - DEX_MAGIC_LEN)

    def test_invalid_dex_raises(self) -> None:
        with self.assertRaises(DexHeaderError):
            extract_header_features(b"dex\n035\x00" + b"\x00" * 10)

    def test_minmax_pipeline(self) -> None:
        a = extract_header_features(self.dex)
        b = a.copy()
        b[0] = min(1.0, b[0] + 0.1)
        matrix = np.stack([a, b])
        mins, maxs = fit_minmax(matrix)
        normed = transform_minmax(matrix, mins, maxs)
        self.assertEqual(normed.shape, (2, DEX_HEADER_FEATURE_DIM))

    def test_sum_aggregation(self) -> None:
        vectors = extract_headers_from_dex_list([self.dex, self.dex])
        summed = aggregate_header_vectors(vectors, "sum")
        np.testing.assert_allclose(summed, vectors[0] * 2.0)


if __name__ == "__main__":
    unittest.main()
