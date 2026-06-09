#!/usr/bin/env python3
"""Filter pulled device JSONL to ablation-only or cascade-only records."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from device_metrics_lib import filter_jsonl_by_mode, write_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="all_scan_metrics.jsonl to filter in place")
    parser.add_argument(
        "--mode",
        choices=("ablation", "cascade"),
        required=True,
        help="Keep scans with cascade_enabled false (ablation) or true (cascade)",
    )
    args = parser.parse_args(argv)

    path = args.jsonl.resolve()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    cascade_enabled = args.mode == "cascade"
    scans, sessions = filter_jsonl_by_mode(path, cascade_enabled=cascade_enabled)
    write_jsonl(path, scans, sessions)
    print(f"Filtered {path} → {len(scans)} scan(s), {len(sessions)} session(s) ({args.mode})")

    merge_script = _TOOLS / "jsonl_to_json.py"
    out_json = path.parent / f"{path.stem}.json"
    subprocess.run(
        [sys.executable, str(merge_script), str(path), "-o", str(out_json)],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
