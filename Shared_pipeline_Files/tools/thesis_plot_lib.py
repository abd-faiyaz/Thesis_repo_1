#!/usr/bin/env python3
"""Shared loaders and matplotlib figures for thesis plot scripts (Phase 6)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from plot_registry_lib import repo_root  # noqa: E402

plt = None
np = None


def _ensure_matplotlib() -> None:
    global plt, np
    if plt is not None:
        return
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as _plt
        import numpy as _np

        mpl.use("Agg")
        plt = _plt
        np = _np
    except ImportError as exc:
        raise SystemExit(
            "matplotlib required for thesis plots: pip install matplotlib numpy "
            "(or use thesis_venv)"
        ) from exc

# Stacked breakdown colours (parse / vectorize / inference).
COLOR_PARSE = "#4C72B0"
COLOR_VECTORIZE = "#DD8452"
COLOR_INFERENCE = "#C44E52"

DEFAULT_TABLE = (
    repo_root() / "Shared_pipeline_Files/results/figures/plot_metrics_table.json"
)
DEFAULT_OUT = repo_root() / "Shared_pipeline_Files/results/figures/templates"
DEFAULT_EXTENDED_ABSTRACT_OUT = (
    repo_root() / "extended_abstract/plots_and_table/Generated"
)

TOP5_COUNT = 5

# Feasibility ranks 1–3 and 5, with rank-4 Dual swapped for MLDP-pruned (extended abstract).
EXTENDED_ABSTRACT_TOP5_MODEL_IDS: tuple[str, ...] = (
    "mlp_header",
    "mldp_dexheader_cascade",
    "dexheader_broadcast_fusion",
    "mldp_pruned_permission",
    "broadcast_mldp_hybrid",
)

# Family-specific stacked-segment colours (reference-inspired palettes).
FAMILY_PALETTES: dict[str, dict[str, str]] = {
    "dex_header": {
        "parse": "#70AD47",
        "vectorize": "#7030A0",
        "inference": "#2E5597",
    },
    "permission": {
        "parse": "#FF2D95",
        "vectorize": "#ED7D31",
        "inference": "#C00000",
    },
    "fusion": {
        "parse": "#00B0F0",
        "vectorize": "#FFC000",
        "inference": "#5B9BD5",
    },
}

MODEL_FEATURE_FAMILY: dict[str, str] = {
    "mlp_header": "dex_header",
    "broadcast_mldp_hybrid": "permission",
    "mldp_pruned_permission": "permission",
    "mldp_dexheader_cascade": "fusion",
    "dexheader_broadcast_fusion": "fusion",
    "dual_branch_dex_manifest": "fusion",
}


def load_table(path: Path | None = None) -> dict[str, Any]:
    table_path = (path or DEFAULT_TABLE).resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"plot_metrics_table not found: {table_path}")
    return json.loads(table_path.read_text(encoding="utf-8"))


def model_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    return list(table.get("models", []))


def model_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [str(r.get("method") or r.get("model_id", "")) for r in rows]


def top5_model_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Extended-abstract top 5 (feasibility order, Dual → MLDP-pruned)."""
    by_id = {str(r["model_id"]): r for r in model_rows(table)}
    out: list[dict[str, Any]] = []
    for mid in EXTENDED_ABSTRACT_TOP5_MODEL_IDS:
        row = by_id.get(mid)
        if row is not None:
            out.append(row)
    return out[:TOP5_COUNT]


def _family_palette(model_id: str) -> dict[str, str]:
    family = MODEL_FEATURE_FAMILY.get(model_id, "fusion")
    return FAMILY_PALETTES[family]


def wilson_accuracy_ci(
    accuracy: float, n_samples: int, *, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for binomial accuracy (proportion scale)."""
    if n_samples <= 0:
        p = float(accuracy)
        return p, p
    p = float(accuracy)
    n = float(n_samples)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (
        z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def pareto_frontier(
    xs: list[float], ys: list[float]
) -> list[int]:
    """Indices of non-dominated points (minimize x, maximize y)."""
    indices = list(range(len(xs)))
    frontier: list[int] = []
    for i in indices:
        dominated = False
        for j in indices:
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (
                xs[j] < xs[i] or ys[j] > ys[i]
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return sorted(frontier, key=lambda k: xs[k])


def _save(fig, out_path: Path) -> Path:
    _ensure_matplotlib()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_apk_size_vs_detection_time(
    table: dict[str, Any], out_dir: Path
) -> Path:
    _ensure_matplotlib()
    series = table.get("per_apk_series", [])
    fig, ax = plt.subplots(figsize=(8, 5))
    specs = [
        ("manifest_xgb", "XGBoost", "#4C72B0", "o"),
        ("bytecnn", "1D-CNN", "#DD8452", "s"),
    ]
    for model_id, label, color, marker in specs:
        xs, ys = [], []
        for entry in series:
            metrics = (entry.get("by_model") or {}).get(model_id)
            if not metrics:
                continue
            x = entry.get("apk_size_mb")
            y = metrics.get("stage_total_ms")
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))
        if xs:
            ax.scatter(xs, ys, label=label, color=color, marker=marker, alpha=0.75, s=40)
    ax.set_xlabel("APK size (MB)")
    ax.set_ylabel("Detection time (ms)")
    ax.set_title("APK Size vs Detection Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "apk_size_vs_detection_time.png")


def plot_inference_breakdown_stacked(
    table: dict[str, Any], out_dir: Path
) -> Path:
    _ensure_matplotlib()
    buckets = table.get("size_bucket_medians", {})
    bucket_order = [str(b) for b in table.get("size_bucket_edges_mb", [])] + ["100+"]
    bucket_order = [b for b in bucket_order if b in buckets]

    compare = [
        ("manifest_xgb", "XGBoost"),
        ("bytecnn", "1D-CNN"),
    ]
    n_buckets = len(bucket_order)
    width = 0.35
    x = np.arange(n_buckets)
    fig, ax = plt.subplots(figsize=(10, 5))

    for offset, (model_id, label) in enumerate(compare):
        parse_v, vec_v, inf_v = [], [], []
        for bucket in bucket_order:
            stats = (buckets.get(bucket) or {}).get(model_id) or {}
            parse_v.append(float(stats.get("parse_ms") or 0))
            vec_v.append(float(stats.get("vectorize_ms") or 0))
            inf_v.append(float(stats.get("inference_ms") or 0))
        pos = x + (offset - 0.5) * width
        ax.bar(pos, parse_v, width, label=f"{label} parse" if offset == 0 else "_nolegend_", color=COLOR_PARSE)
        ax.bar(pos, vec_v, width, bottom=parse_v, label=f"{label} vectorize" if offset == 0 else "_nolegend_", color=COLOR_VECTORIZE)
        bottoms = [parse_v[i] + vec_v[i] for i in range(n_buckets)]
        bars = ax.bar(
            pos,
            inf_v,
            width,
            bottom=bottoms,
            label=f"{label} inference" if offset == 0 else "_nolegend_",
            color=COLOR_INFERENCE,
        )
        for bar, total in zip(bars, bottoms):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                total + height + 0.5,
                f"{total + height:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.bar(pos, [0], width, label=label, color="none", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"≤{b} MB" if b != "100+" else ">100 MB" for b in bucket_order])
    ax.set_ylabel("Median time (ms)")
    ax.set_title("Inference Time Breakdown vs APK Size")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_dir / "inference_breakdown_stacked.png")


def plot_accuracy_vs_ram(table: dict[str, Any], out_dir: Path) -> Path:
    _ensure_matplotlib()
    rows = model_rows(table)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        scan_a = row.get("device_scan_a") or {}
        offline = row.get("offline") or {}
        x = scan_a.get("mem_mb")
        y = offline.get("f1")
        if x is None or y is None:
            continue
        ax.scatter(float(x), float(y) * 100.0, s=60)
        ax.annotate(
            row.get("method", row.get("model_id", "")),
            (float(x), float(y) * 100.0),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xlabel("Memory (MB, device median)")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Accuracy–Memory Trade-off")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "plot3_accuracy_vs_ram.png")


def plot_accuracy_vs_latency(table: dict[str, Any], out_dir: Path) -> Path:
    _ensure_matplotlib()
    rows = model_rows(table)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        scan_a = row.get("device_scan_a") or {}
        offline = row.get("offline") or {}
        x = scan_a.get("inference_ms")
        y = offline.get("f1")
        if x is None or y is None:
            continue
        ax.scatter(float(x), float(y) * 100.0, s=60)
        ax.annotate(
            row.get("method", row.get("model_id", "")),
            (float(x), float(y) * 100.0),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xlabel("Inference time (ms, device median)")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Accuracy–Latency Trade-off")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "plot4_accuracy_vs_latency.png")


def plot_inference_time_vs_apk_size(
    table: dict[str, Any], out_dir: Path
) -> Path:
    """Stacked inference-time breakdown for top-5 feasibility models."""
    _ensure_matplotlib()
    from matplotlib.patches import Patch

    rows = top5_model_rows(table)
    labels = model_labels(rows)
    n = len(rows)
    x = np.arange(n)
    width = 0.62

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(top=0.82)

    ymax = 0.0
    for i, row in enumerate(rows):
        model_id = str(row["model_id"])
        scan_a = row.get("device_scan_a") or {}
        parse_v = float(scan_a.get("parse_ms") or 0)
        vec_v = float(scan_a.get("vectorize_ms") or 0)
        inf_v = float(scan_a.get("inference_ms") or 0)
        palette = _family_palette(model_id)

        ax.bar(i, parse_v, width, color=palette["parse"])
        ax.bar(i, vec_v, width, bottom=parse_v, color=palette["vectorize"])
        bottom = parse_v + vec_v
        ax.bar(i, inf_v, width, bottom=bottom, color=palette["inference"])
        total = bottom + inf_v
        ymax = max(ymax, total)
        label = f"{total:.0f}" if total >= 10 else f"{total:.1f}"
        ax.text(
            i,
            total + ymax * 0.02 + 0.05,
            f"{label} ms",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Model")
    ax.set_title(
        "Inference Time Breakdown by Model",
        fontsize=12,
        fontweight="bold",
        pad=28,
    )
    ax.set_ylim(0, ymax * 1.18 + 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    family_names = {
        "dex_header": "Dex-header",
        "permission": "Permission",
        "fusion": "Fusion",
    }
    legend_handles: list[Patch] = []
    for family, palette in FAMILY_PALETTES.items():
        legend_handles.append(
            Patch(facecolor=palette["parse"], label=f"{family_names[family]} — Feature Extraction")
        )
        legend_handles.append(
            Patch(
                facecolor=palette["vectorize"],
                label=f"{family_names[family]} — Vectorization (Feature Processing)",
            )
        )
        legend_handles.append(
            Patch(
                facecolor=palette["inference"],
                label=f"{family_names[family]} — Model Inference",
            )
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=1)

    fig.text(
        0.5,
        0.94,
        (
            r"$T_{\mathrm{total}} = T_{\mathrm{feature\,extraction}}"
            r"+ T_{\mathrm{vectorization}} + T_{\mathrm{inference}}$"
        ),
        ha="center",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "gray",
            "linestyle": "--",
        },
    )
    return _save(fig, out_dir / "inferenceTime_vs_apkSize.png")


def plot_model_vs_resources(table: dict[str, Any], out_dir: Path) -> Path:
    """Top-5 models: accuracy (Wilson CI) vs on-device RAM / CPU / battery."""
    _ensure_matplotlib()
    rows = top5_model_rows(table)
    labels = model_labels(rows)
    x = np.arange(len(rows))

    acc_pct: list[float] = []
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for row in rows:
        offline = row.get("offline") or {}
        acc = float(offline.get("accuracy") or 0)
        n_samples = int(offline.get("n_samples") or 0)
        lo, hi = wilson_accuracy_ci(acc, n_samples)
        acc_pct.append(acc * 100.0)
        yerr_lo.append(acc * 100.0 - lo * 100.0)
        yerr_hi.append(hi * 100.0 - acc * 100.0)

    mem = [float((r.get("device_scan_a") or {}).get("mem_mb") or 0) for r in rows]
    cpu_ms = [float((r.get("device_scan_a") or {}).get("cpu_ms") or 0) for r in rows]
    battery = [
        float((r.get("device_scan_a") or {}).get("battery_pct_delta") or 0) for r in rows
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.errorbar(
        x,
        acc_pct,
        yerr=[yerr_lo, yerr_hi],
        fmt="o-",
        color="#4C72B0",
        ecolor="#4C72B0",
        capsize=4,
        linewidth=1.2,
        markersize=7,
        label="Accuracy (%)",
        zorder=4,
    )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_title(
        "Experimental Visualization: Model Performance vs Resource Cost",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.3)

    acc_min = min(acc_pct) - max(yerr_lo) - 1
    acc_max = max(acc_pct) + max(yerr_hi) + 1
    ax1.set_ylim(max(0, acc_min), min(100, acc_max))

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        cpu_ms,
        "s-",
        color="#B026FF",
        markersize=6,
        linewidth=1.2,
        label="CPU (%)",
        zorder=3,
    )
    ax2.plot(
        x,
        mem,
        "^-",
        color="#ED7D31",
        markersize=7,
        linewidth=1.2,
        label="RAM (MB)",
        zorder=3,
    )
    ax2.plot(
        x,
        battery,
        "D-",
        color="#70AD47",
        markersize=6,
        linewidth=1.2,
        label="Battery (%)",
        zorder=3,
    )
    ax2.set_ylabel("Resource Usage")

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8)
    return _save(fig, out_dir / "sample_model_vs_accuracy_resources.png")


def plot_performance_tradeoff(table: dict[str, Any], out_dir: Path) -> Path:
    _ensure_matplotlib()
    rows = model_rows(table)
    points: list[tuple[float, float, str, dict[str, Any]]] = []
    for row in rows:
        derived = row.get("derived") or {}
        cost = derived.get("cost_score")
        f1p = derived.get("f1_pct")
        if cost is None or f1p is None:
            continue
        lab = str(row.get("method", row.get("model_id", "")))
        points.append((float(cost), float(f1p), lab, row.get("device_scan_a") or {}))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, s=70, color="#4C72B0", zorder=3)
    for x, y, lab, scan_a in points:
        inf = scan_a.get("inference_ms")
        mem_gb = scan_a.get("mem_gb")
        ax.annotate(
            f"{lab}\nF1={y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
        if inf is not None and mem_gb is not None:
            ax.annotate(
                f"t={inf:.0f}ms, mem={mem_gb:.3f}GB",
                (x, y),
                textcoords="offset points",
                xytext=(5, -12),
                fontsize=6,
                color="gray",
            )

    frontier = pareto_frontier(xs, ys)
    if len(frontier) >= 2:
        fx = [xs[i] for i in frontier]
        fy = [ys[i] for i in frontier]
        order = sorted(range(len(fx)), key=lambda k: fx[k])
        ax.plot(
            [fx[i] for i in order],
            [fy[i] for i in order],
            "r--",
            linewidth=1.5,
            label="Pareto frontier",
            zorder=2,
        )

    ax.set_xlabel("Cost score (lower is better)")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Performance vs Resource Cost")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, out_dir / "performance_res_usage_tradeoff_plot.jpeg")


def plot_cascade_exit_tiers(table: dict[str, Any], out_dir: Path) -> Path | None:
    _ensure_matplotlib()
    scan_b = table.get("device_scan_b")
    if not scan_b:
        return None
    hist = scan_b.get("exit_tier_histogram") or {}
    if not hist:
        return None
    tiers = sorted(hist.keys(), key=lambda k: int(k) if k.isdigit() else 99)
    counts = [int(hist[t]) for t in tiers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tiers, counts, color="#4C72B0")
    ax.set_xlabel("Cascade exit tier")
    ax.set_ylabel("APK count")
    ax.set_title("Cascade Exit Tier Distribution")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.05, str(c), ha="center", fontsize=9)
    return _save(fig, out_dir / "plot_cascade_exit_tiers.png")


ALL_PLOTS = [
    ("apk_size_vs_detection_time", plot_apk_size_vs_detection_time),
    ("inference_breakdown_stacked", plot_inference_breakdown_stacked),
    ("inferenceTime_vs_apkSize", plot_inference_time_vs_apk_size),
    ("plot3_accuracy_vs_ram", plot_accuracy_vs_ram),
    ("plot4_accuracy_vs_latency", plot_accuracy_vs_latency),
    ("sample_model_vs_accuracy_resources", plot_model_vs_resources),
    ("performance_res_usage_tradeoff", plot_performance_tradeoff),
    ("plot_cascade_exit_tiers", plot_cascade_exit_tiers),
]

EXTENDED_ABSTRACT_PLOTS = [
    ("inferenceTime_vs_apkSize", plot_inference_time_vs_apk_size),
    ("sample_model_vs_accuracy_resources", plot_model_vs_resources),
]


def write_extended_abstract_plots(
    table: dict[str, Any],
    out_dir: Path | None = None,
) -> list[Path]:
    """Write extended-abstract figures to plots_and_table/Generated."""
    target = (out_dir or DEFAULT_EXTENDED_ABSTRACT_OUT).resolve()
    written: list[Path] = []
    for _name, fn in EXTENDED_ABSTRACT_PLOTS:
        written.append(fn(table, target))
    ext_manifest = {
        "figures": [p.name for p in written],
        "out_dir": str(target),
    }
    manifest_path = target / "figure_index.json"
    manifest_path.write_text(json.dumps(ext_manifest, indent=2) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def run_all_plots(
    table: dict[str, Any], out_dir: Path
) -> list[Path]:
    written: list[Path] = []
    for _name, fn in ALL_PLOTS:
        result = fn(table, out_dir)
        if result is not None:
            written.append(result)
    written.extend(write_extended_abstract_plots(table))
    manifest = {
        "figures": [str(p.name) for p in written if p.name != "figure_index.json"],
        "out_dir": str(out_dir),
        "extended_abstract_dir": str(DEFAULT_EXTENDED_ABSTRACT_OUT),
    }
    manifest_path = out_dir / "figure_index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest_path not in written:
        written.append(manifest_path)
    return written
