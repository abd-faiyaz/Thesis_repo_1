"""Read classes.dex from APK archives in memory."""

from __future__ import annotations

import zipfile
from pathlib import Path


class ApkExtractError(Exception):
    """APK could not be opened or does not contain the target Dex entry."""


def read_classes_dex(apk_path: Path, entry_name: str = "classes.dex") -> bytes:
    """
    Open APK as ZIP and return raw classes.dex bytes.
    Matches primary Dex only (no classes2.dex).
    """
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            names = zf.namelist()
            if entry_name in names:
                return zf.read(entry_name)

            suffix = f"/{entry_name}"
            for name in names:
                if name.endswith(suffix) or name.lower() == entry_name.lower():
                    return zf.read(name)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ApkExtractError(f"Failed to open APK: {apk_path}") from exc

    raise ApkExtractError(f"'{entry_name}' not found in {apk_path}")
