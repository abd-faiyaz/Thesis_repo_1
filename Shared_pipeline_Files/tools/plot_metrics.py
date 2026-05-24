#!/usr/bin/env python3
"""Plot device scan metrics JSON (Phase 2) and offline eval JSON (Phase 10)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


def load_json_files(directory: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
    return records


def collect_stage_metrics(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    """domain -> metric_name -> values"""
    out: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        for stage in rec.get("stages", []):
            domain = stage.get("domain", "unknown")
            bucket = out.setdefault(domain, {})
            for key in ("parse_ms", "vectorize_ms", "inference_ms", "score"):
                if key in stage and stage[key] is not None:
                    bucket.setdefault(key, []).append(float(stage[key]))
    return out


def plot_device_metrics(records: list[dict], out_dir: Path) -> list[Path]:
    if plt is None:
        raise SystemExit("matplotlib required: pip install matplotlib")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stage_data = collect_stage_metrics(records)

    if not stage_data:
        print("No stage data to plot.", file=sys.stderr)
        return written

    # Latency bar chart (mean inference_ms per domain)
    domains = sorted(stage_data.keys())
    means = [sum(stage_data[d].get("inference_ms", [0])) / max(len(stage_data[d].get("inference_ms", [])), 1) for d in domains]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(domains, means, color="steelblue")
    ax.set_ylabel("Mean inference_ms")
    ax.set_title("On-device inference latency by domain")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    p1 = out_dir / "device_inference_latency.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(p1)

    # Score distribution (boxplot)
    score_series = [stage_data[d].get("score", []) for d in domains]
    if any(score_series):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(score_series, tick_labels=domains)
        ax.set_ylabel("Malware score")
        ax.set_title("Score distribution by domain")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        p2 = out_dir / "device_score_distribution.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        written.append(p2)

    # Totals: wall_ms and mem_delta across scans
    wall = [float(r["totals"]["wall_ms"]) for r in records if r.get("totals", {}).get("wall_ms") is not None]
    mem = [float(r["totals"]["mem_delta_bytes"]) for r in records if r.get("totals", {}).get("mem_delta_bytes") is not None]
    if wall or mem:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if wall:
            axes[0].hist(wall, bins=min(20, max(len(wall), 1)), color="coral")
            axes[0].set_xlabel("wall_ms")
            axes[0].set_title("Total scan wall time")
        if mem:
            axes[1].hist(mem, bins=min(20, max(len(mem), 1)), color="seagreen")
            axes[1].set_xlabel("mem_delta_bytes")
            axes[1].set_title("Native heap delta per scan")
        fig.tight_layout()
        p3 = out_dir / "device_totals_hist.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        written.append(p3)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot metrics JSON from device or offline runs.")
    parser.add_argument(
        "--device-dir",
        type=Path,
        default=Path("Shared_pipeline_Files/results/device"),
        help="Directory of pulled device scan JSON files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Shared_pipeline_Files/results/figures"),
        help="Where to write PNG figures",
    )
    args = parser.parse_args()

    if not args.device_dir.is_dir():
        print(f"Device dir not found: {args.device_dir}", file=sys.stderr)
        return 1

    records = load_json_files(args.device_dir)
    if not records:
        print(f"No JSON files in {args.device_dir}. Run a scan and pull metrics first.", file=sys.stderr)
        return 1

    paths = plot_device_metrics(records, args.out_dir)
    for p in paths:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
