#!/usr/bin/env python3
"""Phase 1: verify dependencies, config, and artifact directories."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
    ("pyaxmlparser", "pyaxmlparser"),
]

VENV_HINT = (
    "/run/media/abd-faiyaz/Files/thesis_vigidroid/thesis_venv/bin/python"
)


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
        from src.config import ensure_artifact_dirs, load_config, validate_model_dims
        from src.features.multidex import multidex_settings

        cfg = load_config()
        ensure_artifact_dirs(cfg)
        md = multidex_settings(cfg.preprocessing)
        if md["mode"] != "sum":
            errors.append(f"  - expected multidex.mode=sum, got {md['mode']!r}")

        dim_errors = validate_model_dims(cfg)
        errors.extend(f"  - {msg}" for msg in dim_errors)
    except Exception as exc:
        errors.append(f"  - src package / config: {exc}")
    return errors


def main() -> int:
    print("Full Combined Pipeline (Pattern A) — Phase 1 environment check\n")
    print(f"Project root: {ROOT}")
    print(f"Python:       {sys.executable}")
    if VENV_HINT not in sys.executable:
        print(f"Hint:         use thesis venv: {VENV_HINT}\n")
    else:
        print()

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
        from src.config import load_config
        from src.features.multidex import multidex_settings

        cfg = load_config()
        md = multidex_settings(cfg.preprocessing)
        model = cfg.model
        print(f"  apk_root:              {cfg.paths.apk_root}")
        print(f"  processed_dir:         {cfg.paths.processed_dir}")
        print(f"  shards_train:          {cfg.paths.shards_train_dir}")
        print(f"  checkpoint_dir:        {cfg.paths.checkpoint_dir}")
        print(f"  multidex.mode:         {md['mode']}")
        print(f"  combined_input_len:    {model.get('combined_input_len')}")
        print(f"  combined_padded_len:   {model.get('combined_padded_len')}")
        print()

    try:
        import zipfile  # noqa: F401

        print("stdlib zipfile OK.\n")
    except ImportError:
        import_errors.append("  - zipfile (stdlib)")

    if import_errors or pkg_errors:
        return 1
    print("Phase 1 setup verified. (Run Phase 2 tests: python -m unittest tests.test_multidex -v)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
