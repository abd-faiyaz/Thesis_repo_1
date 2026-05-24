#!/usr/bin/env python3
"""Verify dependencies, config, and multi-Dex module wiring."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
]


def check_imports() -> list[str]:
    errors: list[str] = []
    for module, pip_name in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            errors.append(f"  - {pip_name} (import {module})")
    return errors


def check_package() -> list[str]:
    errors: list[str] = []
    try:
        from src.config import ensure_artifact_dirs, load_config, multidex_settings_from_config
        from src.features.multidex import aggregate_header_vectors, multidex_settings
        from src.preprocessing.apk_extract import extract_apk_raw_header

        cfg = load_config()
        ensure_artifact_dirs(cfg)
        md = multidex_settings_from_config(cfg)
        if md["mode"] != "sum":
            errors.append(f"  - expected multidex.mode=sum, got {md['mode']!r}")

        cache_version = int(cfg.preprocessing.get("cache_version", 0))
        if cache_version < 2:
            errors.append(f"  - expected preprocessing.cache_version>=2, got {cache_version}")

        _ = aggregate_header_vectors
        _ = extract_apk_raw_header
        _ = multidex_settings
    except Exception as exc:
        errors.append(f"  - src package / config: {exc}")
    return errors


def main() -> int:
    print("Base Model 1 (MLP(H)) — environment check\n")
    print(f"Project root: {ROOT}\n")

    import_errors = check_imports()
    if import_errors:
        print("Missing dependencies (install with pip -r requirements.txt):")
        print("\n".join(import_errors))
        print()
    else:
        print("All pip dependencies OK.\n")

    pkg_errors = check_package()
    if pkg_errors:
        print("Package / config issues:")
        print("\n".join(pkg_errors))
        print()
    else:
        print("Package layout and config loader OK.")
        from src.config import load_config, multidex_settings_from_config

        cfg = load_config()
        md = multidex_settings_from_config(cfg)
        print(f"  processed_dir:   {cfg.paths.processed_dir}")
        print(f"  checkpoint_dir:  {cfg.paths.checkpoint_dir}")
        print(f"  multidex.mode:   {md['mode']}")
        print(f"  dex_pattern:     {md['dex_pattern']}")
        print(f"  cache_version:   {cfg.preprocessing.get('cache_version', 'unset')}")
        print()

    try:
        import zipfile  # noqa: F401 — stdlib, listed in plan

        print("stdlib zipfile OK.\n")
    except ImportError:
        import_errors.append("  - zipfile (stdlib)")

    if import_errors or pkg_errors:
        return 1
    print(
        "Setup verified. Run multi-Dex tests:\n"
        "  PYTHONPATH=. python -m unittest tests.test_multidex tests.test_multidex_preprocess tests.test_dex_header -v"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
