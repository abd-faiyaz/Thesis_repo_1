"""Read all classes*.dex from APK archives in memory."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.dex_header import DexHeaderError, extract_headers_from_dex_list
from src.features.multidex import (
    DEFAULT_DEX_PATTERN,
    DEFAULT_MULTIDEX_MODE,
    aggregate_header_vectors,
    dex_suffix_sort_key,
)


class ApkExtractError(Exception):
    """APK could not be opened or does not contain the target Dex entry."""


@dataclass(frozen=True)
class ApkHeaderExtraction:
    vector: np.ndarray
    num_dex_files: int


def _dex_basename(zip_entry_name: str) -> str:
    return Path(zip_entry_name.replace("\\", "/")).name


def list_dex_entries(
    zf: zipfile.ZipFile,
    *,
    pattern: str = DEFAULT_DEX_PATTERN,
) -> list[str]:
    compiled = re.compile(pattern)
    matches: list[str] = []
    for name in zf.namelist():
        if compiled.match(_dex_basename(name)):
            matches.append(name)
    matches.sort(key=lambda n: dex_suffix_sort_key(_dex_basename(n)))
    return matches


def read_all_dex_from_apk(
    apk_path: Path,
    *,
    pattern: str = DEFAULT_DEX_PATTERN,
) -> list[tuple[str, bytes]]:
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            entries = list_dex_entries(zf, pattern=pattern)
            if not entries:
                raise ApkExtractError(
                    f"No Dex files matching {pattern!r} in APK: {apk_path}"
                )
            return [(name, zf.read(name)) for name in entries]
    except (zipfile.BadZipFile, OSError) as exc:
        raise ApkExtractError(f"Failed to open APK: {apk_path}") from exc


def extract_apk_raw_header(
    apk_path: Path,
    *,
    mode: str = DEFAULT_MULTIDEX_MODE,
    pattern: str = DEFAULT_DEX_PATTERN,
    max_dex: int = 3,
) -> np.ndarray:
    dex_list = read_all_dex_from_apk(apk_path, pattern=pattern)
    try:
        vectors = extract_headers_from_dex_list([data for _, data in dex_list])
    except DexHeaderError as exc:
        raise ApkExtractError(str(exc)) from exc
    return aggregate_header_vectors(vectors, mode, max_dex=max_dex)
