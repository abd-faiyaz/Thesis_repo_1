#!/usr/bin/env python3
"""Build device eval manifest for all 2022+2023 APKs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DATASET_ROOT = Path("/mnt/Files/thesis_full_dataset")
YEARS = ("2022", "2023")
LABELS = ("benign", "malware")
OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "device_eval_manifest.csv"
SUMMARY_PATH = OUT_DIR / "device_eval_manifest_summary.json"
PUSH_SCRIPT_PATH = OUT_DIR / "push_device_eval_apks.sh"
PHONE_DEST = "/sdcard/Download/Scanable"
ID_PREFIX = "scan"


def size_bucket(size_bytes: int) -> str:
    mb = size_bytes / (1024 * 1024)
    if mb < 1:
        return "00_<1MB"
    if mb < 5:
        return "01_1-5MB"
    if mb < 10:
        return "02_5-10MB"
    if mb < 20:
        return "03_10-20MB"
    if mb < 50:
        return "04_20-50MB"
    if mb < 100:
        return "05_50-100MB"
    return "06_100-200MB"


def collect_rows() -> list[dict[str, object]]:
    entries: list[tuple[str, str, Path]] = []
    for year in YEARS:
        for label in LABELS:
            folder = DATASET_ROOT / year / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing dataset folder: {folder}")
            for apk in sorted(folder.glob("*.apk")):
                entries.append((year, label, apk))

    rows: list[dict[str, object]] = []
    for idx, (year, label, apk) in enumerate(entries, start=1):
        size_bytes = apk.stat().st_size
        apk_id = f"{ID_PREFIX}_{idx:04d}"
        sha256 = apk.stem.lower()
        rows.append(
            {
                "apk_id": apk_id,
                "phone_filename": f"{apk_id}_{label}.apk",
                "source_path": str(apk),
                "year": year,
                "label": label,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "size_bucket": size_bucket(size_bytes),
                "original_name": apk.name,
                "sha256": sha256,
                "pushed": "no",
                "scanned": "no",
                "notes": "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "apk_id",
        "phone_filename",
        "source_path",
        "year",
        "label",
        "size_bytes",
        "size_mb",
        "size_bucket",
        "original_name",
        "sha256",
        "pushed",
        "scanned",
        "notes",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]]) -> None:
    benign = sum(1 for row in rows if row["label"] == "benign")
    malware = sum(1 for row in rows if row["label"] == "malware")
    total_bytes = sum(int(row["size_bytes"]) for row in rows)

    by_bucket: Counter[str] = Counter()
    by_bucket_label: dict[str, dict[str, int]] = defaultdict(lambda: {"benign": 0, "malware": 0})
    for row in rows:
        bucket = str(row["size_bucket"])
        label = str(row["label"])
        by_bucket[bucket] += 1
        by_bucket_label[bucket][label] += 1

    summary = {
        "target_total": len(rows),
        "actual_total": len(rows),
        "benign": benign,
        "malware": malware,
        "total_size_gb": round(total_bytes / (1024**3), 2),
        "years": list(YEARS),
        "phone_dest": PHONE_DEST,
        "id_prefix": ID_PREFIX,
        "by_size_bucket": dict(sorted(by_bucket.items())),
        "by_size_bucket_label": {
            bucket: by_bucket_label[bucket] for bucket in sorted(by_bucket_label)
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_push_script(rows: list[dict[str, object]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# Push all 2022-2023 thesis device-eval APKs ({len(rows)} total) to {PHONE_DEST}.",
        "# Skips files already present on the phone so the run can be resumed after adb errors.",
        "# Requires: adb, USB debugging, phone connected.",
        f'DEST="{PHONE_DEST}"',
        f"MANIFEST='{CSV_PATH}'",
        'adb shell "mkdir -p \\"$DEST\\""',
        f'total={len(rows)}',
        'pushed=0',
        'skipped=0',
        'failed=0',
        'echo "Pushing up to $total APKs to $DEST (skipping existing) ..."',
        "# Use process substitution so adb push cannot consume the manifest on stdin.",
        'while IFS=, read -r apk_id phone_filename source_path _rest; do',
        '  dest="$DEST/$phone_filename"',
        '  if adb shell "[ -f \\"$dest\\" ]" </dev/null >/dev/null 2>&1; then',
        '    skipped=$((skipped + 1))',
        '    continue',
        "  fi",
        '  if adb push "$source_path" "$dest" </dev/null; then',
        '    pushed=$((pushed + 1))',
        "  else",
        '    echo "adb push failed for $phone_filename" >&2',
        '    failed=$((failed + 1))',
        '    exit 1',
        "  fi",
        'done < <(tail -n +2 "$MANIFEST")',
        "",
        'echo "Done. APKs are in $DEST"',
        'adb shell ls -1 "$DEST"/*.apk | wc -l',
        "",
    ]
    PUSH_SCRIPT_PATH.write_text("\n".join(lines), encoding="utf-8")
    PUSH_SCRIPT_PATH.chmod(0o755)


def main() -> None:
    rows = collect_rows()
    write_csv(rows)
    write_summary(rows)
    write_push_script(rows)
    print(f"Wrote {len(rows)} rows to {CSV_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")
    print(f"Wrote push script to {PUSH_SCRIPT_PATH}")


if __name__ == "__main__":
    main()
