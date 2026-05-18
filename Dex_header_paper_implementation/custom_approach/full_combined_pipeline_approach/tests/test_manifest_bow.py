"""Unit tests for manifest BoW helpers (no APK required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.manifest_bow import build_lexicon_from_counts, build_multihot_vector


class TestManifestBoW(unittest.TestCase):
    def test_lexicon_and_multihot(self) -> None:
        counts = {
            "android.permission.INTERNET": 10,
            "android.permission.CAMERA": 5,
            "android.intent.action.MAIN": 8,
            "rare.permission": 1,
        }
        token_to_index, unk_index = build_lexicon_from_counts(
            counts, lexicon_size=10, min_token_freq=2
        )
        self.assertEqual(unk_index, 3)
        vector_size = unk_index + 1

        vec = build_multihot_vector(
            ["android.permission.INTERNET", "unknown.token"],
            token_to_index,
            vector_size=vector_size,
            unk_index=unk_index,
        )
        self.assertEqual(vec.shape, (vector_size,))
        self.assertEqual(vec[token_to_index["android.permission.INTERNET"]], 1.0)
        self.assertEqual(vec[unk_index], 1.0)
        self.assertEqual(vec.sum(), 2.0)


if __name__ == "__main__":
    unittest.main()
