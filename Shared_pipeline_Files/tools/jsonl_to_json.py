#!/usr/bin/env python3
"""Fold append-only all_scan_metrics.jsonl into legacy {device, scans[]} JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_records(jsonl_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scans: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON: {exc}") from exc
            record_type = record.get("record_type")
            if record_type == "session":
                sessions.append(record)
            elif record_type == "scan" or "stages" in record:
                scans.append(record)
    return scans, sessions


def merge(jsonl_path: Path, out_path: Path) -> tuple[int, int]:
    scans, sessions = load_records(jsonl_path)
    device: dict[str, Any] = {}
    if scans:
        device = scans[0].get("device", {})
    elif sessions:
        device = sessions[0].get("device", {})
    payload: dict[str, Any] = {"device": device, "scans": scans}
    if sessions:
        payload["sessions"] = sessions
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(scans), len(sessions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        help="Path to all_scan_metrics.jsonl (default: sibling of script results/device/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output all_scan_metrics.json path",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir.parent / "results" / "device"
    jsonl_path = args.jsonl or (default_dir / "all_scan_metrics.jsonl")
    out_path = args.output or (default_dir / "all_scan_metrics.json")

    if not jsonl_path.is_file():
        print(f"JSONL not found: {jsonl_path}", file=sys.stderr)
        return 1

    scan_count, session_count = merge(jsonl_path, out_path)
    print(f"Wrote {scan_count} scan(s), {session_count} session(s) → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
