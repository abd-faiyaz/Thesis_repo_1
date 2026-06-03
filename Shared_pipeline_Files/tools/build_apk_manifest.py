#!/usr/bin/env python3
"""Build apk_index.csv from a corpus root (manifest with paths, no APK copy)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

BENIGN_NAMES = {"benign", "goodware", "clean", "good", "0"}
MALWARE_NAMES = {"malware", "malicious", "virus", "bad", "1"}
YEAR_RE = re.compile(r"(20[12][0-9])")


def load_config(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def infer_label(apk_path: Path, apk_root: Path) -> str | None:
    rel = apk_path.relative_to(apk_root)
    for part in rel.parts[:-1]:
        key = part.lower()
        if key in BENIGN_NAMES:
            return "benign"
        if key in MALWARE_NAMES:
            return "malware"
    return None


def infer_year(apk_path: Path) -> str:
    for part in apk_path.parts:
        m = YEAR_RE.search(part)
        if m:
            return m.group(1)
    return ""


def iter_apks(apk_root: Path) -> list[Path]:
    return sorted(p for p in apk_root.rglob("*.apk") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build apk_index.csv from corpus root.")
    parser.add_argument("--config", type=Path, default=Path("Shared_pipeline_Files/data/dataset_paths.yaml"))
    parser.add_argument("--apk-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Max APKs (0 = all)")
    parser.add_argument("--skip-hash", action="store_true", help="Leave sha256 empty (faster smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apk_root = (args.apk_root or Path(cfg["apk_root"])).resolve()
    if not apk_root.is_dir():
        print(f"apk_root not found: {apk_root}", file=sys.stderr)
        return 1

    out = args.output or Path(cfg.get("manifest_csv", "Shared_pipeline_Files/data/manifests/apk_index.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    apks = iter_apks(apk_root)
    if args.limit:
        apks = apks[: args.limit]

    for i, apk in enumerate(apks, 1):
        label = infer_label(apk, apk_root)
        if label is None:
            print(f"skip (no label): {apk}", file=sys.stderr)
            continue
        rel = apk.relative_to(apk_root).as_posix()
        digest = "" if args.skip_hash else sha256_file(apk)
        rows.append(
            {
                "apk_path": rel,
                "sha256": digest,
                "label": label,
                "year": infer_year(apk),
                "split": "",
            }
        )
        if i % 500 == 0:
            print(f"indexed {i} APKs...", file=sys.stderr)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["apk_path", "sha256", "label", "year", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
