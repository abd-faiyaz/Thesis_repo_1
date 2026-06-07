"""P8 — PyTorch vs ONNX Runtime parity on export parity_samples."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.models import DeployedMlpHeaderRef
from src.models.export_wrapper import MalwareProbExport, Stage1ProbExport
from src.training.export_onnx import load_mode_a_export_model, load_stage1_export_model

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_parity_samples(bundle_dir: Path) -> list[dict[str, Any]]:
    index_path = bundle_dir / "parity_samples" / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing parity index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index.get("samples") or []
    samples: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["sample_id"])
        sample_dir = bundle_dir / "parity_samples" / sid
        expected_path = sample_dir / "expected_prob.json"
        if not expected_path.is_file():
            raise FileNotFoundError(f"Missing expected_prob.json for {sid}")

        samples.append(
            {
                "sample_id": sid,
                "index": int(row.get("index", -1)),
                "label": int(row.get("label", -1)),
                "x_s": np.load(sample_dir / "x_S.npy").astype(np.float32),
                "h": np.load(sample_dir / "H.npy").astype(np.float32),
                "x": np.load(sample_dir / "x.npy").astype(np.float32),
                "expected": json.loads(expected_path.read_text(encoding="utf-8")),
            }
        )
    if not samples:
        raise ValueError(f"No parity samples found in {index_path}")
    return samples


@torch.no_grad()
def run_pytorch_mode_a(model: MalwareProbExport, x: np.ndarray) -> float:
    tensor = torch.from_numpy(x.reshape(1, -1))
    return float(model(tensor).view(-1)[0].item())


@torch.no_grad()
def run_pytorch_stage1(model: Stage1ProbExport, x_s: np.ndarray) -> float:
    tensor = torch.from_numpy(x_s.reshape(1, -1))
    return float(model(tensor).view(-1)[0].item())


def run_onnx_session(session, features: np.ndarray, *, output_name: str | None = None) -> float:
    inp_name = session.get_inputs()[0].name
    out_name = output_name or session.get_outputs()[0].name
    x = features.reshape(1, -1).astype(np.float32)
    out = session.run([out_name], {inp_name: x})[0]
    return float(np.asarray(out, dtype=np.float64).ravel()[0])


def _summarize_channel(
    *,
    channel: str,
    pytorch_scores: np.ndarray,
    onnx_scores: np.ndarray,
    expected_scores: np.ndarray,
    sample_ids: list[str],
    tolerance: float,
) -> dict[str, Any]:
    pt_onnx = np.abs(pytorch_scores - onnx_scores)
    pt_expected = np.abs(pytorch_scores - expected_scores)
    onnx_expected = np.abs(onnx_scores - expected_scores)
    max_delta = float(pt_onnx.max())
    per_sample: list[dict[str, Any]] = []
    for i, sid in enumerate(sample_ids):
        per_sample.append(
            {
                "sample_id": sid,
                "pytorch": float(pytorch_scores[i]),
                "onnx": float(onnx_scores[i]),
                "export_expected": float(expected_scores[i]),
                "delta_pytorch_onnx": float(pt_onnx[i]),
                "delta_pytorch_expected": float(pt_expected[i]),
                "delta_onnx_expected": float(onnx_expected[i]),
            }
        )
    return {
        "channel": channel,
        "passed": max_delta <= tolerance,
        "max_delta": max_delta,
        "mean_delta": float(pt_onnx.mean()),
        "pytorch_vs_onnx": {
            "max_delta": max_delta,
            "mean_delta": float(pt_onnx.mean()),
        },
        "pytorch_vs_export_expected": {
            "max_delta": float(pt_expected.max()),
            "mean_delta": float(pt_expected.mean()),
        },
        "onnx_vs_export_expected": {
            "max_delta": float(onnx_expected.max()),
            "mean_delta": float(onnx_expected.mean()),
        },
        "per_sample": per_sample,
    }


def _summarize_stage2(
    *,
    ref_scores: np.ndarray,
    onnx_scores: np.ndarray,
    expected_scores: np.ndarray,
    sample_ids: list[str],
    tolerance: float,
) -> dict[str, Any]:
    ref_onnx = np.abs(ref_scores - onnx_scores)
    ref_expected = np.abs(ref_scores - expected_scores)
    onnx_expected = np.abs(onnx_scores - expected_scores)
    max_delta = float(ref_onnx.max())
    per_sample: list[dict[str, Any]] = []
    for i, sid in enumerate(sample_ids):
        per_sample.append(
            {
                "sample_id": sid,
                "deployed_onnx_ref": float(ref_scores[i]),
                "bundle_onnx": float(onnx_scores[i]),
                "export_expected": float(expected_scores[i]),
                "delta_ref_onnx": float(ref_onnx[i]),
                "delta_ref_expected": float(ref_expected[i]),
                "delta_onnx_expected": float(onnx_expected[i]),
            }
        )
    return {
        "channel": "stage2_prob",
        "passed": max_delta <= tolerance,
        "max_delta": max_delta,
        "mean_delta": float(ref_onnx.mean()),
        "note": "Stage 2 is a byte-copy of deployed mlp_header; reference uses the source deployed bundle via DeployedMlpHeaderRef.",
        "deployed_onnx_vs_bundle_onnx": {
            "max_delta": max_delta,
            "mean_delta": float(ref_onnx.mean()),
        },
        "deployed_onnx_vs_export_expected": {
            "max_delta": float(ref_expected.max()),
            "mean_delta": float(ref_expected.mean()),
        },
        "bundle_onnx_vs_export_expected": {
            "max_delta": float(onnx_expected.max()),
            "mean_delta": float(onnx_expected.mean()),
        },
        "per_sample": per_sample,
    }


def run_parity_check(
    cfg: PipelineConfig,
    *,
    bundle_dir: Path | None = None,
    mode_a_checkpoint: Path | None = None,
    stage1_checkpoint: Path | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    ensure_artifact_dirs(cfg)
    bundle = (bundle_dir or cfg.paths.export).resolve()
    tol = float(tolerance if tolerance is not None else cfg.export.get("parity_max_delta", 1e-4))

    mode_a_onnx = bundle / "mode_a" / "model.onnx"
    stage1_onnx = bundle / "mode_b" / "stage1_mldp.onnx"
    stage2_onnx = bundle / "mode_b" / "stage2_mlp_header.onnx"
    for path in (mode_a_onnx, stage1_onnx, stage2_onnx):
        if not path.is_file():
            raise FileNotFoundError(f"Missing ONNX in export bundle: {path}")

    mode_a_ckpt = (mode_a_checkpoint or (cfg.paths.checkpoints / "mode_a_best.pt")).resolve()
    stage1_ckpt = (stage1_checkpoint or (cfg.paths.checkpoints / "stage1_best.pt")).resolve()
    if not mode_a_ckpt.is_file():
        raise FileNotFoundError(f"Mode A checkpoint not found: {mode_a_ckpt}")
    if not stage1_ckpt.is_file():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt}")

    samples = load_parity_samples(bundle)
    sample_ids = [str(s["sample_id"]) for s in samples]

    import onnxruntime as ort

    mode_a_session = ort.InferenceSession(
        str(mode_a_onnx), providers=["CPUExecutionProvider"]
    )
    stage1_session = ort.InferenceSession(
        str(stage1_onnx), providers=["CPUExecutionProvider"]
    )
    stage2_session = ort.InferenceSession(
        str(stage2_onnx), providers=["CPUExecutionProvider"]
    )
    stage2_ref = DeployedMlpHeaderRef.from_config(cfg)

    mode_a_model, _ = load_mode_a_export_model(cfg, mode_a_ckpt)
    stage1_model, _ = load_stage1_export_model(cfg, stage1_ckpt)
    mode_a_model.eval()
    stage1_model.eval()

    mode_a_pytorch: list[float] = []
    mode_a_onnx_scores: list[float] = []
    mode_a_expected: list[float] = []
    stage1_pytorch: list[float] = []
    stage1_onnx_scores: list[float] = []
    stage1_expected: list[float] = []
    stage2_ref_scores: list[float] = []
    stage2_onnx_scores: list[float] = []
    stage2_expected: list[float] = []

    stage2_output_name = stage2_session.get_outputs()[0].name

    for sample in samples:
        x = sample["x"]
        x_s = sample["x_s"]
        h = sample["h"]
        expected = sample["expected"]

        mode_a_pytorch.append(run_pytorch_mode_a(mode_a_model, x))
        mode_a_onnx_scores.append(
            run_onnx_session(mode_a_session, x, output_name="malware_prob")
        )
        mode_a_expected.append(float(expected["mode_a_malware_prob"]))

        stage1_pytorch.append(run_pytorch_stage1(stage1_model, x_s))
        stage1_onnx_scores.append(
            run_onnx_session(stage1_session, x_s, output_name="stage1_prob")
        )
        stage1_expected.append(float(expected["stage1_prob"]))

        stage2_ref_scores.append(float(stage2_ref.score(h)[0]))
        stage2_onnx_scores.append(
            run_onnx_session(stage2_session, h, output_name=stage2_output_name)
        )
        stage2_expected.append(float(expected["stage2_prob"]))

    mode_a_report = _summarize_channel(
        channel="mode_a_malware_prob",
        pytorch_scores=np.asarray(mode_a_pytorch, dtype=np.float64),
        onnx_scores=np.asarray(mode_a_onnx_scores, dtype=np.float64),
        expected_scores=np.asarray(mode_a_expected, dtype=np.float64),
        sample_ids=sample_ids,
        tolerance=tol,
    )
    stage1_report = _summarize_channel(
        channel="stage1_prob",
        pytorch_scores=np.asarray(stage1_pytorch, dtype=np.float64),
        onnx_scores=np.asarray(stage1_onnx_scores, dtype=np.float64),
        expected_scores=np.asarray(stage1_expected, dtype=np.float64),
        sample_ids=sample_ids,
        tolerance=tol,
    )
    stage2_report = _summarize_stage2(
        ref_scores=np.asarray(stage2_ref_scores, dtype=np.float64),
        onnx_scores=np.asarray(stage2_onnx_scores, dtype=np.float64),
        expected_scores=np.asarray(stage2_expected, dtype=np.float64),
        sample_ids=sample_ids,
        tolerance=tol,
    )

    max_delta = max(
        mode_a_report["max_delta"],
        stage1_report["max_delta"],
        stage2_report["max_delta"],
    )
    passed = (
        mode_a_report["passed"]
        and stage1_report["passed"]
        and stage2_report["passed"]
    )

    mode_a_manifest_path = bundle / "mode_a" / "export_manifest.json"
    mode_b_manifest_path = bundle / "mode_b" / "export_manifest.json"
    manifest = {}
    if mode_a_manifest_path.is_file():
        manifest["mode_a"] = json.loads(mode_a_manifest_path.read_text(encoding="utf-8"))
    if mode_b_manifest_path.is_file():
        manifest["mode_b"] = json.loads(mode_b_manifest_path.read_text(encoding="utf-8"))

    return {
        "timestamp": _utc_now(),
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "bundle_dir": str(bundle),
        "mode_a_checkpoint": str(mode_a_ckpt),
        "stage1_checkpoint": str(stage1_ckpt),
        "onnx_files": {
            "mode_a": str(mode_a_onnx),
            "stage1": str(stage1_onnx),
            "stage2": str(stage2_onnx),
        },
        "tolerance": tol,
        "passed": passed,
        "max_delta": max_delta,
        "max_delta_by_output": {
            "mode_a_malware_prob": mode_a_report["max_delta"],
            "stage1_prob": stage1_report["max_delta"],
            "stage2_prob": stage2_report["max_delta"],
        },
        "n_samples": len(samples),
        "mode_a": mode_a_report,
        "stage1": stage1_report,
        "stage2": stage2_report,
        "export_manifest": manifest,
        "failure_hints": [
            "Verify float32 row-major tensors (x_S, H, x).",
            "Mode A / Stage 1: sigmoid must be inside the ONNX graph (MalwareProbExport / Stage1ProbExport).",
            "Stage 2: H must use deployed normalization_header.json (not recomputed per APK).",
            "Re-export after checkpoint changes: bash scripts/run_export.sh",
        ],
    }


def write_parity_report(cfg: PipelineConfig, report: dict[str, Any]) -> Path:
    out_path = cfg.paths.metrics / "parity_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTorch vs ONNX parity check for mldp_dexheader_cascade (P8)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Export bundle directory (default: artifacts/export/mldp_dexheader_cascade)",
    )
    parser.add_argument(
        "--mode-a-checkpoint",
        type=Path,
        default=None,
        help="Default: artifacts/checkpoints/mode_a_best.pt",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help="Default: artifacts/checkpoints/stage1_best.pt",
    )
    parser.add_argument("--tolerance", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    report = run_parity_check(
        cfg,
        bundle_dir=args.bundle,
        mode_a_checkpoint=args.mode_a_checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        tolerance=args.tolerance,
    )
    report_path = write_parity_report(cfg, report)

    status = "PASS" if report["passed"] else "FAIL"
    print(f"P8 parity {status}: overall max_delta={report['max_delta']:.2e} (tolerance {report['tolerance']})")
    for key in ("mode_a", "stage1", "stage2"):
        section = report[key]
        print(
            f"  {section['channel']}: max_delta={section['max_delta']:.2e} "
            f"({'PASS' if section['passed'] else 'FAIL'})"
        )
    print(f"  samples: {report['n_samples']}")
    print(f"  report → {report_path}")

    if not report["passed"]:
        for key in ("mode_a", "stage1", "stage2"):
            section = report[key]
            if section["passed"]:
                continue
            if key == "stage2":
                worst = max(section["per_sample"], key=lambda row: row["delta_ref_onnx"])
                print(
                    f"  worst {key}: {worst['sample_id']} "
                    f"ref={worst['deployed_onnx_ref']:.6f} "
                    f"onnx={worst['bundle_onnx']:.6f} "
                    f"delta={worst['delta_ref_onnx']:.2e}"
                )
            else:
                worst = max(section["per_sample"], key=lambda row: row["delta_pytorch_onnx"])
                print(
                    f"  worst {key}: {worst['sample_id']} "
                    f"pytorch={worst['pytorch']:.6f} "
                    f"onnx={worst['onnx']:.6f} "
                    f"delta={worst['delta_pytorch_onnx']:.2e}"
                )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
