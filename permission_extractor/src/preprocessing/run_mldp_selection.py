"""P2b — run MLDP selection on train transactions only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ensure_artifact_dirs, load_config
from src.mldp.pipeline import load_transactions_for_split, run_mldp_selection

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLDP permission selection (train only).")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    transactions, labels, _ = load_transactions_for_split(cfg.paths.transactions_dir, "train")
    if not transactions:
        raise SystemExit("No train transactions; run build_transactions.py first")

    selected = run_mldp_selection(cfg, train_transactions=transactions, train_labels=labels)
    print(f"Selected |S|={len(selected)} permissions → {cfg.paths.selected_permissions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
