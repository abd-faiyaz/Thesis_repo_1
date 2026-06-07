"""Tests for permission token normalization and binary vectors."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.permission_vector import (
    build_binary_vector,
    load_vocab,
    normalize_permission,
    save_vocab,
)


class TestPermissionNormalization(unittest.TestCase):
    def test_android_permission_prefix(self) -> None:
        self.assertEqual(
            normalize_permission("android.permission.SEND_SMS"),
            "permissions::send_sms",
        )

    def test_dots_become_underscores(self) -> None:
        self.assertEqual(
            normalize_permission("android.permission.READ_PHONE_STATE"),
            "permissions::read_phone_state",
        )


class TestVocabAndVectors(unittest.TestCase):
    def test_save_load_vocab_roundtrip(self) -> None:
        perms = ["permissions::internet", "permissions::send_sms"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            save_vocab(path, perms)
            loaded, token_to_index = load_vocab(path)
            self.assertEqual(loaded, perms)
            self.assertEqual(token_to_index["permissions::send_sms"], 1)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["M"], 2)
            self.assertEqual(data["token_normalization"], "vigidroid")

    def test_build_binary_vector_respects_vocab(self) -> None:
        perms = ["permissions::internet", "permissions::send_sms", "permissions::camera"]
        token_to_index = {p: i for i, p in enumerate(perms)}
        vec = build_binary_vector(
            ["permissions::send_sms", "permissions::unknown", "permissions::camera"],
            token_to_index,
            vector_size=len(perms),
        )
        np.testing.assert_array_equal(vec, np.array([0.0, 1.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
