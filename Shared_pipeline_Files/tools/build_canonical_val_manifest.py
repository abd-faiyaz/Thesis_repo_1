#!/usr/bin/env python3
"""Build canonical_val.txt (SHA-256 list) from a pipeline val manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_apk_ids_from_manifest(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(entry["apk_id"]).lower() for entry in payload]
    if isinstance(payload, dict) and "entries" in payload:
        return [str(entry["apk_id"]).lower() for entry in payload["entries"]]
    raise ValueError(f"Unsupported manifest shape: {manifest_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write Shared_pipeline_Files/data/splits/canonical_val.txt"
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="e.g. permission_extractor/artifacts/processed/manifest_val.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "Shared_pipeline_Files/data/splits/canonical_val.txt",
    )
    args = parser.parse_args(argv)

    apk_ids = sorted(set(load_apk_ids_from_manifest(args.source_manifest.resolve())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Canonical validation APK set (2022-2023 holdout val partition).\n"
        "# One full SHA-256 per line; used for cross-model val_scores alignment.\n"
    )
    args.out.write_text(header + "\n".join(apk_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(apk_ids)} APK ids → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
