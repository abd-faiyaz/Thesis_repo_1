"""P2a — extract full permission transactions per APK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import ensure_artifact_dirs, load_config
from src.features.permission_vector import extract_permission_tokens
from src.preprocessing.common import read_dataset_index

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build permission transaction JSON per APK.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "dev_test", "temporal_holdout"],
    )
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    rows = read_dataset_index(cfg.paths.dataset_index)
    failed_log = cfg.paths.failed_apks_log

    for split in args.splits:
        split_rows = [r for r in rows if r.split == split]
        out_dir = cfg.paths.transactions_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for row in tqdm(split_rows, desc=f"transactions:{split}"):
            out_path = out_dir / f"{row.apk_id}.json"
            if out_path.is_file():
                ok += 1
                continue
            try:
                tokens = extract_permission_tokens(row.apk_path)
            except Exception as exc:
                with failed_log.open("a", encoding="utf-8") as f:
                    f.write(f"{row.apk_path}\t{exc}\n")
                continue
            payload = {
                "apk_id": row.apk_id,
                "permissions": tokens,
                "label": row.label,
            }
            out_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            ok += 1
        print(f"{split}: {ok} transactions in {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
