#!/usr/bin/env python3
"""Generate plotting sufficiency report from plot_metrics_table + outputs (Phase 8)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from aggregate_plot_metrics import missing_required_fields  # noqa: E402
from plot_registry_lib import repo_root  # noqa: E402
from thesis_plot_lib import DEFAULT_OUT, DEFAULT_TABLE  # noqa: E402

WAIVED: dict[str, str] = {}

REQUIRED_FIGURES = [
    "apk_size_vs_detection_time.png",
    "inference_breakdown_stacked.png",
    "inferenceTime_vs_apkSize.png",
    "plot3_accuracy_vs_ram.png",
    "plot4_accuracy_vs_latency.png",
    "sample_model_vs_accuracy_resources.png",
    "performance_res_usage_tradeoff_plot.jpeg",
]

OPTIONAL_FIGURES = ["plot_cascade_exit_tiers.png"]


def _status(ok: bool, waived: bool = False) -> str:
    if waived:
        return "Waived"
    return "Yes" if ok else "No"


def build_report(
    *,
    table_path: Path,
    figures_dir: Path,
    csv_path: Path | None,
) -> tuple[str, list[str]]:
    table = json.loads(table_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    blockers: list[str] = []

    agg_errors = missing_required_fields(table)
    rows.append(
        {
            "deliverable": "plot_metrics_table.json (ranked models complete)",
            "status": _status(not agg_errors),
            "notes": "; ".join(agg_errors[:3]) if agg_errors else f"{len(table.get('models', []))} models",
        }
    )
    if agg_errors:
        blockers.extend(agg_errors[:5])

    for fig in REQUIRED_FIGURES:
        exists = (figures_dir / fig).is_file()
        rows.append(
            {
                "deliverable": fig,
                "status": _status(exists),
                "notes": str(figures_dir / fig) if exists else "run run_all_thesis_plots.sh",
            }
        )
        if not exists:
            blockers.append(f"missing figure: {fig}")

    for fig in OPTIONAL_FIGURES:
        exists = (figures_dir / fig).is_file()
        has_scan_b = bool(table.get("device_scan_b"))
        rows.append(
            {
                "deliverable": fig,
                "status": _status(exists or not has_scan_b),
                "notes": "Scan B bonus chart" if has_scan_b else "no Scan B data yet",
            }
        )

    csv_ok = csv_path is not None and csv_path.is_file()
    rows.append(
        {
            "deliverable": "Extended-abstract CSV",
            "status": _status(csv_ok),
            "notes": str(csv_path) if csv_ok else "run build_extended_abstract_csv.py",
        }
    )

    for item, note in WAIVED.items():
        rows.append({"deliverable": item, "status": "Waived", "notes": note})

    scan_a = table.get("n_scans_scan_a", 0)
    scan_b = table.get("n_scans_scan_b", 0)
    rows.append(
        {
            "deliverable": "Device Scan A samples",
            "status": _status(scan_a > 0),
            "notes": f"n={scan_a}",
        }
    )
    rows.append(
        {
            "deliverable": "Device Scan B samples",
            "status": _status(scan_b > 0),
            "notes": f"n={scan_b}",
        }
    )

    lines = [
        "# Plotting sufficiency report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Table: `{table_path}`",
        f"Figures: `{figures_dir}`",
        "",
        "| Deliverable | Status | Notes |",
        "|-------------|--------|-------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['deliverable']} | {row['status']} | {row['notes']} |"
        )

    all_required_yes = all(
        r["status"] == "Yes"
        for r in rows
        if r["deliverable"] not in WAIVED
        and r["deliverable"] != "Device Scan B samples"
        and not r["deliverable"].startswith("plot_cascade")
    )
    lines.extend(
        [
            "",
            f"**Overall (required deliverables):** {'Yes' if all_required_yes and not blockers else 'No — see blockers'}",
            "",
        ]
    )
    if blockers:
        lines.append("## Blockers")
        lines.append("")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    return "\n".join(lines) + "\n", blockers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = repo_root()
    table_path = args.table.resolve()
    if not table_path.is_file():
        print(f"Missing {table_path}", file=sys.stderr)
        return 1

    csv_path = args.csv or (
        root / "Illustrations_templates/On-Device ML-Experiments - Sheet1-generated.csv"
    )
    out = args.out or (
        root / "Shared_pipeline_Files/results/figures/plotting_sufficiency_report.md"
    )

    report, blockers = build_report(
        table_path=table_path,
        figures_dir=args.figures_dir.resolve(),
        csv_path=csv_path.resolve() if csv_path else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
