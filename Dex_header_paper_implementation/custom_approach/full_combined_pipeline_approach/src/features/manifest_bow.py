"""Manifest permission and intent tokens → multi-hot BoW vector."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from pyaxmlparser import APK

from src.constants import DEFAULT_LEXICON_SIZE

_PYAXML_LOGGERS = (
    "pyaxmlparser",
    "pyaxmlparser.axmlparser",
    "pyaxmlparser.axmlprinter",
    "pyaxmlparser.core",
)


class ManifestBoWError(ValueError):
    """Manifest could not be parsed or encoded."""


@contextmanager
def _quiet_pyaxmlparser():
    """Suppress pyaxmlparser warnings for packed/obfuscated manifests during batch runs."""
    saved: list[tuple[logging.Logger, int]] = []
    for name in _PYAXML_LOGGERS:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in saved:
            logger.setLevel(level)


def extract_manifest_tokens(apk_path: Path) -> list[str]:
    """
    Collect permission names and intent action/category strings from the manifest.
    Uses pyaxmlparser (Python 3–compatible; replaces legacy axmlparserpy).
    """
    try:
        with _quiet_pyaxmlparser():
            apk = APK(str(apk_path))
    except Exception as exc:
        raise ManifestBoWError(f"Failed to parse APK manifest: {apk_path}") from exc

    tokens: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        v = value.strip()
        if v and v not in seen:
            seen.add(v)
            tokens.append(v)

    with _quiet_pyaxmlparser():
        for perm in apk.get_permissions():
            add(perm)

        for tag_name in ("action", "category"):
            for tag in apk.find_tags(tag_name):
                add(apk.get_value_from_tag(tag, "name"))

    if not tokens:
        raise ManifestBoWError(f"No manifest tokens found: {apk_path}")
    return tokens


def build_multihot_vector(
    tokens: Iterable[str],
    token_to_index: Mapping[str, int],
    *,
    vector_size: int,
    unk_index: int,
) -> np.ndarray:
    vec = np.zeros(vector_size, dtype=np.float32)
    for token in tokens:
        idx = token_to_index.get(token, unk_index)
        if 0 <= idx < vector_size:
            vec[idx] = 1.0
    return vec


def save_vocab(
    path: Path,
    token_to_index: dict[str, int],
    *,
    lexicon_size: int,
    unk_index: int,
    min_token_freq: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "lexicon_size": lexicon_size,
        "unk_index": unk_index,
        "vector_size": lexicon_size + 1,
        "min_token_freq": min_token_freq,
        "token_to_index": token_to_index,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_vocab(path: Path) -> tuple[dict[str, int], int, int]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    token_to_index = {str(k): int(v) for k, v in data["token_to_index"].items()}
    unk_index = int(data["unk_index"])
    vector_size = int(data.get("vector_size", len(token_to_index) + 1))
    return token_to_index, unk_index, vector_size


def build_lexicon_from_counts(
    token_counts: Mapping[str, int],
    *,
    lexicon_size: int = DEFAULT_LEXICON_SIZE,
    min_token_freq: int = 2,
) -> tuple[dict[str, int], int]:
    """Top-N frequent tokens (freq >= min_token_freq); UNK at index N."""
    filtered = [(t, c) for t, c in token_counts.items() if c >= min_token_freq]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    top = filtered[:lexicon_size]
    token_to_index = {token: i for i, (token, _) in enumerate(top)}
    unk_index = len(token_to_index)
    return token_to_index, unk_index
