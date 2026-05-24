"""Read all classes*.dex from APK archives in memory."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.dex_header import (
    DexHeaderError,
    extract_headers_from_dex_list,
)
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
    """Aggregated header vector plus per-APK Dex discovery metadata."""

    vector: np.ndarray
    num_dex_files: int


def _dex_basename(zip_entry_name: str) -> str:
    return Path(zip_entry_name.replace("\\", "/")).name


def _read_zip_entry(zf: zipfile.ZipFile, entry_name: str) -> bytes:
    names = zf.namelist()
    if entry_name in names:
        return zf.read(entry_name)

    suffix = f"/{entry_name}"
    for name in names:
        if name.endswith(suffix) or name.lower() == entry_name.lower():
            return zf.read(name)

    raise ApkExtractError(f"'{entry_name}' not found in APK")


def list_dex_entries(
    zf: zipfile.ZipFile,
    *,
    pattern: str = DEFAULT_DEX_PATTERN,
) -> list[str]:
    """Return ZIP entry paths for all Dex files matching pattern on basename."""
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
    """Open APK once and read every matched Dex entry."""
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


def extract_apk_header_extraction(
    apk_path: Path,
    *,
    mode: str = DEFAULT_MULTIDEX_MODE,
    pattern: str = DEFAULT_DEX_PATTERN,
    max_dex: int = 3,
) -> ApkHeaderExtraction:
    """Discover all classes*.dex, parse headers, aggregate into one feature vector."""
    dex_list = read_all_dex_from_apk(apk_path, pattern=pattern)
    try:
        vectors = extract_headers_from_dex_list([data for _, data in dex_list])
    except DexHeaderError as exc:
        raise ApkExtractError(str(exc)) from exc
    vector = aggregate_header_vectors(vectors, mode, max_dex=max_dex)
    return ApkHeaderExtraction(vector=vector, num_dex_files=len(dex_list))


def extract_apk_raw_header(
    apk_path: Path,
    *,
    mode: str = DEFAULT_MULTIDEX_MODE,
    pattern: str = DEFAULT_DEX_PATTERN,
    max_dex: int = 3,
) -> np.ndarray:
    """Discover all classes*.dex, parse headers, aggregate into one feature vector."""
    return extract_apk_header_extraction(
        apk_path,
        mode=mode,
        pattern=pattern,
        max_dex=max_dex,
    ).vector


def read_classes_dex(apk_path: Path, entry_name: str = "classes.dex") -> bytes:
    """Read a single Dex entry (legacy / ablation helper)."""
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            return _read_zip_entry(zf, entry_name)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ApkExtractError(f"Failed to open APK: {apk_path}") from exc
