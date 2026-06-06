#!/usr/bin/env python3
"""Verify MLDP pipeline environment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("mlxtend", "mlxtend"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("torch", "torch"),
    ("pyaxmlparser", "pyaxmlparser"),
    ("pandas", "pandas"),
]


def main() -> int:
    print("MLDP pruned permission classifier — environment check\n")
    errors: list[str] = []
    for module, pip_name in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            errors.append(f"  - {pip_name}")

    if errors:
        print("Missing dependencies:\n" + "\n".join(errors))
        return 1

    from src.config import ensure_artifact_dirs, load_config
    from src.features.permission_vector import normalize_permission

    cfg = load_config()
    ensure_artifact_dirs(cfg)
    sample = normalize_permission("android.permission.READ_SMS")
    if sample != "permissions::read_sms":
        print(f"Bad normalization: {sample}")
        return 1

    print("Dependencies OK.")
    print(f"  apk_root: {cfg.paths.apk_root}")
    print(f"  model_id: {cfg.pipeline.get('model_id')}")
    print(f"  max |S|: {cfg.mldp.get('max_permissions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
