#!/usr/bin/env python3
"""Load device scan JSON/JSONL and compute Scan A summary statistics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from plot_registry_lib import load_registry, repo_root

# Thesis plot template bucket upper bounds (MB).
SIZE_BUCKET_EDGES_MB = [1, 5, 10, 25, 50, 100]

COST_SCORE_FORMULA = (
    "Per ranked model: z(stage_total_ms) + z(cpu_ms) + z(mem_mb) using Scan A medians "
    "across plot_order models; min–max the sum to [0.1, 100] (higher = more costly)."
)


def load_device_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load scans and session records from JSONL or legacy aggregate JSON."""
    path = path.resolve()
    if path.suffix == ".jsonl":
        scans: list[dict[str, Any]] = []
        sessions: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if record.get("record_type") == "session":
                    sessions.append(record)
                elif record.get("record_type") == "scan" or "stages" in record:
                    scans.append(record)
        return scans, sessions

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, []
    return list(payload.get("scans", [])), list(payload.get("sessions", []))


def load_scan_records(path: Path) -> list[dict[str, Any]]:
    """Load scans from JSONL or legacy {device, scans[]} JSON."""
    scans, _sessions = load_device_records(path)
    return scans


def sessions_by_id(sessions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for session in sessions:
        sid = session.get("session_id")
        if sid:
            out[str(sid)] = session
    return out


def effective_cascade_enabled(
    scan: dict[str, Any], session_index: dict[str, dict[str, Any]]
) -> bool | None:
    if "cascade_enabled" in scan:
        return bool(scan["cascade_enabled"])
    session_id = scan.get("session_id")
    if session_id and session_id in session_index:
        session = session_index[session_id]
        if "cascade_enabled" in session:
            return bool(session["cascade_enabled"])
    return None


def filter_scans_by_cascade_mode(
    scans: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    *,
    cascade_enabled: bool,
) -> list[dict[str, Any]]:
    session_index = sessions_by_id(sessions)
    out: list[dict[str, Any]] = []
    for scan in scans:
        if scan.get("dedup_skipped"):
            continue
        mode = effective_cascade_enabled(scan, session_index)
        if mode is cascade_enabled:
            out.append(scan)
    return out


def filter_jsonl_by_mode(
    jsonl_path: Path, *, cascade_enabled: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scans, sessions = load_device_records(jsonl_path)
    kept_scans = filter_scans_by_cascade_mode(
        scans, sessions, cascade_enabled=cascade_enabled
    )
    kept_session_ids = {s.get("session_id") for s in kept_scans if s.get("session_id")}
    kept_sessions = [
        s
        for s in sessions
        if s.get("session_id") in kept_session_ids or s.get("cascade_enabled") is cascade_enabled
    ]
    return kept_scans, kept_sessions


def write_jsonl(
    path: Path,
    scans: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session, separators=(",", ":")) + "\n")
        for scan in scans:
            handle.write(json.dumps(scan, separators=(",", ":")) + "\n")


def apk_keys(scan: dict[str, Any]) -> set[str]:
    apk = scan.get("apk") or {}
    keys: set[str] = set()
    if apk.get("sha256"):
        keys.add(str(apk["sha256"]))
    if apk.get("name"):
        keys.add(str(apk["name"]))
    return keys


def stage_total_ms(stage: dict[str, Any]) -> float | None:
    parts = [
        stage.get("parse_ms"),
        stage.get("vectorize_ms"),
        stage.get("inference_ms"),
    ]
    if any(p is None for p in parts):
        return None
    return float(parts[0]) + float(parts[1]) + float(parts[2])


def mem_mb(stage: dict[str, Any]) -> float | None:
    raw = stage.get("mem_delta_bytes")
    if raw is None:
        return None
    return float(raw) / (1024.0 * 1024.0)


def device_stage_for_registry_model(model_id: str) -> str:
    if model_id == "mldp_dexheader_cascade":
        return "mldp_dexheader_cascade_mode_a"
    return model_id


def registry_model_from_stage_id(stage_model_id: str | None) -> str | None:
    if not stage_model_id:
        return None
    if stage_model_id == "mldp_dexheader_cascade_mode_a":
        return "mldp_dexheader_cascade"
    if stage_model_id == "mldp_dexheader_cascade_mode_b":
        return "mldp_dexheader_cascade"
    return stage_model_id


def session_battery_pct(session: dict[str, Any]) -> float | None:
    """Session-level battery drain as % of full charge."""
    battery = session.get("battery") or {}
    pct_delta = battery.get("capacity_pct_delta")
    if pct_delta is not None and float(pct_delta) != 0.0:
        return float(pct_delta)
    used = battery.get("charge_counter_uah_used")
    start_uah = battery.get("charge_counter_uah_start")
    cap_start = battery.get("capacity_pct_start")
    if used is not None and start_uah and float(start_uah) > 0 and cap_start is not None:
        return float(used) / float(start_uah) * float(cap_start)
    return None


def compute_session_battery_per_model(
    scans: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    *,
    root: Path | None = None,
) -> dict[str, float | None]:
    """Allocate Scan A session battery drain to models by aggregate stage time share."""
    root = root or repo_root()
    model_ids = plot_order_model_ids(root)
    session_index = sessions_by_id(sessions)
    active = [s for s in scans if not s.get("dedup_skipped")]

    by_session: dict[str, list[dict[str, Any]]] = {}
    for scan in active:
        sid = scan.get("session_id")
        key = str(sid) if sid else "__no_session__"
        by_session.setdefault(key, []).append(scan)

    per_model_values: dict[str, list[float]] = {mid: [] for mid in model_ids}

    for sid, session_scans in by_session.items():
        session = session_index.get(sid) if sid != "__no_session__" else None
        session_bat = session_battery_pct(session) if session else None
        if session_bat is None:
            continue

        model_ms = {mid: 0.0 for mid in model_ids}
        total_ms = 0.0
        for scan in session_scans:
            for stage in scan.get("stages") or []:
                if stage.get("status") != "ok":
                    continue
                registry_mid = registry_model_from_stage_id(stage.get("model_id"))
                if registry_mid not in model_ms:
                    continue
                st = stage_total_ms(stage)
                if st is None:
                    continue
                model_ms[registry_mid] += float(st)
                total_ms += float(st)

        if total_ms <= 0:
            continue
        for mid, ms in model_ms.items():
            if ms > 0:
                per_model_values[mid].append(session_bat * (ms / total_ms))

    return {mid: _median(vals) for mid, vals in per_model_values.items()}


def _min_max_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def cascade_tier_weight_map(root: Path | None = None) -> dict[str, float]:
    root = root or repo_root()
    spec_path = root / "Shared_pipeline_Files/data/cascade_tier_spec.json"
    registry = load_registry(root)
    feas = registry.get("feasibility", {})
    tier_cfg = feas.get("tier_weights", {})
    default_tiers = {"1": 1.0, "2": 0.85, "3": 0.65, "4": 0.5}
    weights: dict[str, float] = {}
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        for tier_entry in spec.get("tiers", []):
            tier = tier_entry.get("tier")
            w = float(tier_cfg.get(str(tier), default_tiers.get(str(tier), 0.25)))
            for mid in tier_entry.get("models", []):
                registry_mid = registry_model_from_stage_id(str(mid))
                if registry_mid:
                    weights[registry_mid] = w
    weights.setdefault(
        "mlp_header",
        float(tier_cfg.get("mlp_header_fallback", 0.8)),
    )
    not_in = float(tier_cfg.get("not_in_cascade", 0.25))
    for mid in plot_order_model_ids(root):
        weights.setdefault(mid, not_in)
    return weights


def feasibility_composite_ranking(
    model_rows: list[dict[str, Any]], *, root: Path | None = None
) -> list[tuple[str, float]]:
    """Return (model_id, composite_score) sorted best-first."""
    root = root or repo_root()
    registry = load_registry(root)
    feas = registry.get("feasibility", {})
    w_q = float(feas.get("quality_weight", 0.45))
    w_t = float(feas.get("tier_weight", 0.3))
    w_c = float(feas.get("cost_weight", 0.25))
    tier_map = cascade_tier_weight_map(root)

    qualities: list[float] = []
    costs: list[float] = []
    scored: list[tuple[str, float, float, float]] = []

    for row in model_rows:
        mid = row["model_id"]
        offline = row.get("offline") or {}
        scan_a = row.get("device_scan_a") or {}
        acc = offline.get("accuracy")
        f1 = offline.get("f1")
        auc = offline.get("roc_auc")
        if acc is None or f1 is None or auc is None:
            continue
        quality = (float(acc) + float(f1) + float(auc)) / 3.0
        st = scan_a.get("stage_total_ms")
        mem = scan_a.get("mem_mb")
        cost = float(st or 999.0) + float(mem or 0.0) * 10.0
        tier = tier_map.get(mid, 0.25)
        qualities.append(quality)
        costs.append(cost)
        scored.append((mid, quality, cost, tier))

    if not scored:
        return []

    q_norm = _min_max_norm(qualities)
    c_norm = _min_max_norm(costs)
    composite: list[tuple[str, float]] = []
    for i, (mid, _quality, _cost, tier) in enumerate(scored):
        comp = w_q * q_norm[i] + w_t * tier + w_c * (1.0 - c_norm[i])
        composite.append((mid, comp))

    composite.sort(key=lambda x: x[1], reverse=True)
    return composite


def feasibility_ranked_model_ids(
    model_rows: list[dict[str, Any]], *, root: Path | None = None
) -> list[str]:
    """Model ids ordered by feasibility composite score (best first)."""
    return [mid for mid, _ in feasibility_composite_ranking(model_rows, root=root)]


def compute_feasibility_ranks(
    model_rows: list[dict[str, Any]], *, root: Path | None = None
) -> dict[str, str]:
    """Rank models → high / medium / low feasibility labels."""
    root = root or repo_root()
    registry = load_registry(root)
    feas = registry.get("feasibility", {})
    labels = feas.get("labels", {})
    high_l = labels.get("high", "high")
    med_l = labels.get("medium", "medium")
    low_l = labels.get("low", "low")
    top_high = int(feas.get("top_high", 3))
    middle_medium = int(feas.get("middle_medium", 4))

    composite = feasibility_composite_ranking(model_rows, root=root)
    if not composite:
        return {}

    out: dict[str, str] = {}
    for rank, (mid, _) in enumerate(composite):
        if rank < top_high:
            out[mid] = high_l
        elif rank < top_high + middle_medium:
            out[mid] = med_l
        else:
            out[mid] = low_l
    return out


def expected_ablation_stage_ids(root: Path | None = None) -> list[str]:
    registry = load_registry(root)
    return list(registry.get("device_stage_ids", []))


def plot_order_model_ids(root: Path | None = None) -> list[str]:
    registry = load_registry(root)
    return list(registry.get("plot_order", []))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def collect_stage_samples(
    scans: list[dict[str, Any]], *, stage_model_id: str
) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = {
        "parse_ms": [],
        "vectorize_ms": [],
        "inference_ms": [],
        "cpu_ms": [],
        "stage_total_ms": [],
        "mem_mb": [],
        "battery_pct_delta": [],
    }
    for scan in scans:
        if scan.get("dedup_skipped"):
            continue
        for stage in scan.get("stages") or []:
            if stage.get("model_id") != stage_model_id:
                continue
            if stage.get("status") != "ok":
                continue
            for key in ("parse_ms", "vectorize_ms", "inference_ms", "cpu_ms"):
                val = stage.get(key)
                if val is not None:
                    buckets[key].append(float(val))
            total = stage_total_ms(stage)
            if total is not None:
                buckets["stage_total_ms"].append(total)
            mb = mem_mb(stage)
            if mb is not None:
                buckets["mem_mb"].append(mb)
            bat = stage.get("battery_pct_delta")
            if bat is not None:
                buckets["battery_pct_delta"].append(float(bat))
    return buckets


def summarize_scan_a_model(
    scans: list[dict[str, Any]], *, registry_model_id: str
) -> dict[str, Any]:
    stage_id = device_stage_for_registry_model(registry_model_id)
    buckets = collect_stage_samples(scans, stage_model_id=stage_id)
    scan_a = {
        "device_stage_id": stage_id,
        "n_stage_samples": len(buckets["stage_total_ms"]),
        "parse_ms": _median(buckets["parse_ms"]),
        "vectorize_ms": _median(buckets["vectorize_ms"]),
        "inference_ms": _median(buckets["inference_ms"]),
        "cpu_ms": _median(buckets["cpu_ms"]),
        "stage_total_ms": _median(buckets["stage_total_ms"]),
        "mem_mb": _median(buckets["mem_mb"]),
        "battery_pct_delta": _median(buckets["battery_pct_delta"]),
    }
    return {"model_id": registry_model_id, "device_scan_a": scan_a}


def device_feasibility(
    stage_total: float | None, mem: float | None, registry: dict[str, Any]
) -> str:
    """Legacy threshold rule — prefer compute_feasibility_ranks for CSV/plots."""
    _ = stage_total, mem
    labels = registry.get("feasibility", {}).get("labels", {})
    return labels.get("medium", "medium")


def summarize_scan_b(scans: list[dict[str, Any]]) -> dict[str, Any]:
    active = [s for s in scans if not s.get("dedup_skipped")]
    walls: list[float] = []
    tiers: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    reasons: dict[str, int] = {}
    models_run_counts: list[int] = []
    models_skipped_counts: list[int] = []
    early_exit = 0

    for scan in active:
        totals = scan.get("totals") or {}
        wall = totals.get("wall_ms")
        if wall is not None:
            walls.append(float(wall))
        cascade = scan.get("cascade") or {}
        tier = cascade.get("exit_tier")
        if isinstance(tier, int) and tier in tiers:
            tiers[tier] += 1
            if tier < 4:
                early_exit += 1
        reason = cascade.get("exit_reason")
        if isinstance(reason, str):
            reasons[reason] = reasons.get(reason, 0) + 1
        models_run = cascade.get("models_run") or []
        models_skipped = cascade.get("models_skipped") or []
        if isinstance(models_run, list):
            models_run_counts.append(len(models_run))
        if isinstance(models_skipped, list):
            models_skipped_counts.append(len(models_skipped))

    n = len(active)
    return {
        "n_scans": n,
        "median_wall_ms": _median(walls),
        "mean_wall_ms": float(statistics.mean(walls)) if walls else None,
        "exit_tier_histogram": {str(k): v for k, v in sorted(tiers.items())},
        "exit_tier_rates": {
            str(k): (v / n if n else 0.0) for k, v in sorted(tiers.items())
        },
        "early_exit_rate": (early_exit / n if n else 0.0),
        "exit_reason_counts": reasons,
        "median_models_run": _median([float(x) for x in models_run_counts]),
        "median_models_skipped": _median([float(x) for x in models_skipped_counts]),
    }


def merge_plot_metrics_scan_b(
    table_path: Path,
    scan_b_summary: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    if table_path.is_file():
        table = json.loads(table_path.read_text(encoding="utf-8"))
    else:
        table = {"version": 1, "models": []}

    table["device_scan_b"] = scan_b_summary
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return table


def apk_size_mb(size_bytes: int | float | None) -> float | None:
    if size_bytes is None:
        return None
    return float(size_bytes) / (1024.0 * 1024.0)


def size_bucket_label(size_bytes: int | float | None) -> str | None:
    mb = apk_size_mb(size_bytes)
    if mb is None:
        return None
    for edge in SIZE_BUCKET_EDGES_MB:
        if mb <= edge:
            return str(edge)
    return "100+"


def cpu_pct(cpu_ms: float | None, stage_total: float | None) -> float | None:
    if cpu_ms is None or stage_total is None or stage_total <= 0:
        return None
    return float(cpu_ms) / float(stage_total) * 100.0


def _stage_metrics_for_model(scan: dict[str, Any], stage_model_id: str) -> dict[str, Any] | None:
    for stage in scan.get("stages") or []:
        if stage.get("model_id") != stage_model_id:
            continue
        if stage.get("status") != "ok":
            return None
        total = stage_total_ms(stage)
        mb = mem_mb(stage)
        cpu = stage.get("cpu_ms")
        cpu_f = float(cpu) if cpu is not None else None
        return {
            "parse_ms": stage.get("parse_ms"),
            "vectorize_ms": stage.get("vectorize_ms"),
            "inference_ms": stage.get("inference_ms"),
            "cpu_ms": cpu_f,
            "stage_total_ms": total,
            "mem_mb": mb,
            "mem_gb": (mb / 1024.0) if mb is not None else None,
            "battery_pct_delta": stage.get("battery_pct_delta"),
            "cpu_pct": cpu_pct(cpu_f, total),
        }
    return None


def build_per_apk_series(
    scans: list[dict[str, Any]], root: Path | None = None
) -> list[dict[str, Any]]:
    root = root or repo_root()
    model_ids = plot_order_model_ids(root)
    active = [s for s in scans if not s.get("dedup_skipped")]
    series: list[dict[str, Any]] = []
    for scan in active:
        apk = scan.get("apk") or {}
        size_bytes = apk.get("size_bytes")
        entry: dict[str, Any] = {
            "scan_id": scan.get("scan_id"),
            "apk_name": apk.get("name"),
            "apk_sha256": apk.get("sha256"),
            "apk_size_bytes": size_bytes,
            "apk_size_mb": apk_size_mb(size_bytes),
            "size_bucket": size_bucket_label(size_bytes),
            "totals_wall_ms": (scan.get("totals") or {}).get("wall_ms"),
            "by_model": {},
        }
        for model_id in model_ids:
            stage_id = device_stage_for_registry_model(model_id)
            metrics = _stage_metrics_for_model(scan, stage_id)
            if metrics is not None:
                entry["by_model"][model_id] = metrics
        series.append(entry)
    return series


def build_size_bucket_medians(
    scans: list[dict[str, Any]], root: Path | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """Median stage timings per (size_bucket, model_id) from Scan A."""
    root = root or repo_root()
    model_ids = plot_order_model_ids(root)
    buckets: dict[str, dict[str, dict[str, list[float]]]] = {
        str(edge): {mid: {} for mid in model_ids} for edge in SIZE_BUCKET_EDGES_MB
    }
    buckets["100+"] = {mid: {} for mid in model_ids}

    def _append(
        store: dict[str, list[float]], key: str, value: float | None
    ) -> None:
        if value is None:
            return
        store.setdefault(key, []).append(float(value))

    active = [s for s in scans if not s.get("dedup_skipped")]
    for scan in active:
        apk = scan.get("apk") or {}
        bucket = size_bucket_label(apk.get("size_bytes"))
        if bucket is None or bucket not in buckets:
            continue
        for model_id in model_ids:
            stage_id = device_stage_for_registry_model(model_id)
            metrics = _stage_metrics_for_model(scan, stage_id)
            if metrics is None:
                continue
            slot = buckets[bucket][model_id]
            for key in (
                "parse_ms",
                "vectorize_ms",
                "inference_ms",
                "cpu_ms",
                "stage_total_ms",
                "mem_mb",
            ):
                val = metrics.get(key)
                if val is not None:
                    _append(slot, key, float(val))

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for bucket, per_model in buckets.items():
        out[bucket] = {}
        for model_id, samples in per_model.items():
            if not samples.get("stage_total_ms"):
                continue
            out[bucket][model_id] = {
                "n": len(samples["stage_total_ms"]),
                "parse_ms": _median(samples.get("parse_ms", [])),
                "vectorize_ms": _median(samples.get("vectorize_ms", [])),
                "inference_ms": _median(samples.get("inference_ms", [])),
                "cpu_ms": _median(samples.get("cpu_ms", [])),
                "stage_total_ms": _median(samples.get("stage_total_ms", [])),
                "mem_mb": _median(samples.get("mem_mb", [])),
            }
    return out


def _zscores(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return [0.0 for _ in values]
    return [(v - mean) / stdev for v in values]


def _min_max_scale(values: list[float], lo: float = 0.1, hi: float = 100.0) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [(lo + hi) / 2.0]
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [(lo + hi) / 2.0 for _ in values]
    return [lo + (v - vmin) * (hi - lo) / (vmax - vmin) for v in values]


def compute_cost_scores(model_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return model_id → cost_score in [0.1, 100] from Scan A resource medians."""
    ids: list[str] = []
    stage_totals: list[float] = []
    cpus: list[float] = []
    mems: list[float] = []
    for row in model_rows:
        scan_a = row.get("device_scan_a") or {}
        st = scan_a.get("stage_total_ms")
        cpu = scan_a.get("cpu_ms")
        mem = scan_a.get("mem_mb")
        if st is None or cpu is None or mem is None:
            continue
        ids.append(row["model_id"])
        stage_totals.append(float(st))
        cpus.append(float(cpu))
        mems.append(float(mem))

    if not ids:
        return {}

    z_stage = _zscores(stage_totals)
    z_cpu = _zscores(cpus)
    z_mem = _zscores(mems)
    raw = [z_stage[i] + z_cpu[i] + z_mem[i] for i in range(len(ids))]
    scaled = _min_max_scale(raw)
    return {ids[i]: float(scaled[i]) for i in range(len(ids))}


def enrich_scan_a_metrics(scan_a: dict[str, Any]) -> dict[str, Any]:
    out = dict(scan_a)
    mb = out.get("mem_mb")
    if mb is not None:
        out["mem_gb"] = float(mb) / 1024.0
    out["cpu_pct"] = cpu_pct(out.get("cpu_ms"), out.get("stage_total_ms"))
    return out


SCAN_A_JSONL = "scan_a_all_models.jsonl"
SCAN_A_JSON = "scan_a_all_models.json"
SCAN_B_JSONL = "scan_b_cascade.jsonl"
SCAN_B_JSON = "scan_b_cascade.json"
LEGACY_JSONL = "all_scan_metrics.jsonl"
LEGACY_JSON = "all_scan_metrics.json"


def resolve_device_metrics_path(
    directory: Path, *, prefer_cascade: bool | None = None
) -> Path | None:
    """Resolve metrics file in a pull directory (mode-specific or legacy combined)."""
    directory = directory.resolve()
    if prefer_cascade is True:
        candidates = (SCAN_B_JSONL, SCAN_B_JSON, LEGACY_JSONL, LEGACY_JSON)
    elif prefer_cascade is False:
        candidates = (SCAN_A_JSONL, SCAN_A_JSON, LEGACY_JSONL, LEGACY_JSON)
    else:
        name = directory.name.lower()
        if "scan_b" in name or "cascade" in name:
            candidates = (SCAN_B_JSONL, SCAN_B_JSON, LEGACY_JSONL, LEGACY_JSON)
        elif "scan_a" in name or "ablation" in name or "all_models" in name:
            candidates = (SCAN_A_JSONL, SCAN_A_JSON, LEGACY_JSONL, LEGACY_JSON)
        else:
            candidates = (
                SCAN_A_JSONL,
                SCAN_A_JSON,
                SCAN_B_JSONL,
                SCAN_B_JSON,
                LEGACY_JSONL,
                LEGACY_JSON,
            )
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    return None


def jsonl_filename_for_mode(cascade_enabled: bool) -> str:
    return SCAN_B_JSONL if cascade_enabled else SCAN_A_JSONL


def legacy_json_filename_for_mode(cascade_enabled: bool) -> str:
    return SCAN_B_JSON if cascade_enabled else SCAN_A_JSON


def build_plot_metrics_table_scan_a(
    scans: list[dict[str, Any]],
    sessions: list[dict[str, Any]] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    registry = load_registry(root)
    session_battery = (
        compute_session_battery_per_model(scans, sessions or [], root=root)
        if sessions
        else {}
    )
    models: list[dict[str, Any]] = []
    for model_id in plot_order_model_ids(root):
        row = summarize_scan_a_model(scans, registry_model_id=model_id)
        row["device_scan_a"] = enrich_scan_a_metrics(row["device_scan_a"])
        scan_a = row["device_scan_a"]
        session_bat = session_battery.get(model_id)
        if session_bat is not None:
            scan_a["battery_pct_delta"] = session_bat
            scan_a["battery_source"] = "session_time_share"
        elif scan_a.get("battery_pct_delta") == 0.0:
            scan_a["battery_pct_delta"] = None
        row["derived"] = {}
        models.append(row)

    cost_scores = compute_cost_scores(models)
    feas_ranks = compute_feasibility_ranks(models, root=root)
    for row in models:
        mid = row["model_id"]
        derived = row.setdefault("derived", {})
        if mid in cost_scores:
            derived["cost_score"] = cost_scores[mid]
        if mid in feas_ranks:
            derived["device_feasibility"] = feas_ranks[mid]

    active = [s for s in scans if not s.get("dedup_skipped")]
    return {
        "version": 1,
        "source": "scan_a",
        "n_scans": len(active),
        "models": models,
    }
