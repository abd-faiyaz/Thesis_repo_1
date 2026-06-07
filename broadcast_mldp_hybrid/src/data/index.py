"""Load P1 apk_index.csv rows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.config import PipelineConfig


@dataclass(frozen=True)
class ApkIndexRow:
    apk_path: Path
    sha256: str
    label: int
    year: int | None
    split: str


def load_apk_index(path: Path | None = None, cfg: PipelineConfig | None = None) -> list[ApkIndexRow]:
    index_path = path or (cfg.paths.dataset_index if cfg else None)
    if index_path is None:
        raise ValueError("index path required")
    if not index_path.is_file():
        raise FileNotFoundError(f"apk index not found: {index_path}")

    rows: list[ApkIndexRow] = []
    with index_path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            year_raw = raw.get("year", "").strip()
            rows.append(
                ApkIndexRow(
                    apk_path=Path(raw["apk_path"]).resolve(),
                    sha256=raw["sha256"].strip().lower(),
                    label=int(raw["label"]),
                    year=int(year_raw) if year_raw else None,
                    split=raw["split"].strip(),
                )
            )
    return rows


def rows_for_split(rows: list[ApkIndexRow], split: str) -> list[ApkIndexRow]:
    return [r for r in rows if r.split == split]
