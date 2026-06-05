"""P7/P8: ONNX export bundle and PyTorch vs ONNX parity for Pattern A."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config import PipelineConfig, load_config
from src.data.dataset import CombinedPipelineDataset
from src.features.multidex import multidex_settings
from src.models.combined_net import CombinedNet, build_combined_net_from_config
from src.pipeline_integration import get_pipeline_settings
from src.training.checkpoint import load_checkpoint, load_model_from_checkpoint

ONNX_OPSET = 14
DEFAULT_TOLERANCE = 1e-4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class OnnxInferenceWrapper(nn.Module):
    """Export graph: two float inputs → sigmoid malware probability."""

    def __init__(self, model: CombinedNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, header: torch.Tensor, bow: torch.Tensor) -> torch.Tensor:
        return self.model.predict_proba(header, bow)


def load_wrapped_model(
    checkpoint_path: Path,
    *,
    config_path: Path | None = None,
) -> tuple[OnnxInferenceWrapper, PipelineConfig, dict[str, Any]]:
    cfg = load_config(config_path)
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_combined_net_from_config(cfg)
    load_model_from_checkpoint(ckpt, model)
    wrapper = OnnxInferenceWrapper(model)
    wrapper.eval()
    meta = {
        "header_dim": int(cfg.model.get("header_dim", 104)),
        "bow_dim": int(cfg.model.get("bow_dim", 4381)),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    return wrapper, cfg, meta


def export_onnx_model(
    wrapper: OnnxInferenceWrapper,
    onnx_path: Path,
    *,
    header_dim: int,
    bow_dim: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_header = torch.zeros(1, header_dim, dtype=torch.float32)
    dummy_bow = torch.zeros(1, bow_dim, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy_header, dummy_bow),
        str(onnx_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=["header", "bow"],
        output_names=["malware_probability"],
        dynamic_axes={
            "header": {0: "batch_size"},
            "bow": {0: "batch_size"},
            "malware_probability": {0: "batch_size"},
        },
    )
    data_sidecar = onnx_path.with_suffix(".onnx.data")
    if data_sidecar.is_file():
        data_sidecar.unlink()


def build_export_manifest(
    cfg: PipelineConfig,
    meta: dict[str, Any],
    *,
    settings_model_id: str,
    settings_domain: str,
    onnx_path: Path,
    normalization_rel: str,
    vocab_rel: str,
) -> dict[str, Any]:
    md = multidex_settings(cfg.preprocessing)
    pre = cfg.preprocessing
    return {
        "model_id": settings_model_id,
        "domain": settings_domain,
        "exported_at": _utc_now(),
        "opset_version": ONNX_OPSET,
        "checkpoint": meta["checkpoint"],
        "header_dim": meta["header_dim"],
        "bow_dim": meta["bow_dim"],
        "preprocessing_version": int(pre.get("cache_version", 2)),
        "multidex_mode": md["mode"],
        "multidex_max_dex": md["max_dex"],
        "normalization": normalization_rel,
        "vocab": vocab_rel,
        "inputs": [
            {
                "name": "header",
                "shape": [1, meta["header_dim"]],
                "dtype": "float32",
                "description": "Min-max normalized Dex header (post-multidex aggregation)",
            },
            {
                "name": "bow",
                "shape": [1, meta["bow_dim"]],
                "dtype": "float32",
                "description": "Manifest BoW multihot vector (lexicon order)",
            },
        ],
        "outputs": [
            {
                "name": "malware_probability",
                "shape": [1, 1],
                "dtype": "float32",
                "description": "Sigmoid malware probability in [0, 1]",
            }
        ],
        "onnx_file": onnx_path.name,
        "android_assets_target": f"vigidroid/app/src/main/assets/models/{settings_model_id}/",
    }


def build_thresholds(cfg: PipelineConfig) -> dict[str, Any]:
    threshold = float(cfg.evaluation.get("threshold", 0.5))
    return {
        "malware_threshold": threshold,
        "benign_threshold": 1.0 - threshold,
        "description": "Predict malware when malware_probability >= malware_threshold",
    }


def write_parity_samples(
    wrapper: OnnxInferenceWrapper,
    out_dir: Path,
    *,
    manifest_path: Path,
    num_samples: int = 8,
    seed: int = 42,
) -> Path:
    dataset = CombinedPipelineDataset.from_manifest(manifest_path)
    n = len(dataset)
    num_samples = min(num_samples, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_samples, replace=False)

    headers: list[np.ndarray] = []
    bows: list[np.ndarray] = []
    labels: list[int] = []
    expected: list[float] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            header, bow, label = dataset[int(idx)]
            h = header.float().unsqueeze(0)
            b = bow.float().unsqueeze(0)
            score = float(wrapper(h, b).item())
            headers.append(h.numpy().astype(np.float32).ravel())
            bows.append(b.numpy().astype(np.float32).ravel())
            labels.append(int(label.item()))
            expected.append(score)
            sample_ids.append(f"sample_{i:03d}")

    samples_dir = out_dir / "parity_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        samples_dir / "sample_vectors.npz",
        indices=indices.astype(np.int64),
        headers=np.stack(headers, axis=0),
        bows=np.stack(bows, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        expected_scores=np.asarray(expected, dtype=np.float64),
        sample_ids=np.asarray(sample_ids),
    )
    index = [
        {
            "sample_id": sid,
            "index": int(indices[i]),
            "label": labels[i],
            "expected_malware_probability": expected[i],
            "header_dim": len(headers[i]),
            "bow_dim": len(bows[i]),
        }
        for i, sid in enumerate(sample_ids)
    ]
    write_json(
        samples_dir / "index.json",
        {"num_samples": num_samples, "seed": seed, "samples": index},
    )
    return samples_dir


def verify_onnx(
    onnx_path: Path,
    *,
    header_dim: int,
    bow_dim: int,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime required: pip install onnxruntime") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_names = {inp.name: inp.name for inp in session.get_inputs()}
    out_name = session.get_outputs()[0].name
    header_name = inp_names.get("header", session.get_inputs()[0].name)
    bow_name = inp_names.get("bow", session.get_inputs()[1].name)
    h = np.zeros((1, header_dim), dtype=np.float32)
    b = np.zeros((1, bow_dim), dtype=np.float32)
    out = session.run([out_name], {header_name: h, bow_name: b})[0]
    return {
        "input_names": [header_name, bow_name],
        "output_name": out_name,
        "test_output_shape": list(out.shape),
        "test_output_value": float(out.ravel()[0]),
    }


def export_bundle(
    *,
    checkpoint: Path,
    out_dir: Path,
    config_path: Path | None = None,
    num_parity_samples: int = 8,
    skip_verify: bool = False,
) -> Path:
    wrapper, cfg, meta = load_wrapped_model(checkpoint, config_path=config_path)
    settings = get_pipeline_settings(cfg)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "model.onnx"
    export_onnx_model(
        wrapper,
        onnx_path,
        header_dim=meta["header_dim"],
        bow_dim=meta["bow_dim"],
    )

    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.paths.normalization_stats, features_dir / "normalization_header.json")
    shutil.copy2(cfg.paths.vocab, features_dir / "vocab.json")

    write_json(out_dir / "thresholds.json", build_thresholds(cfg))
    manifest = build_export_manifest(
        cfg,
        meta,
        settings_model_id=settings.model_id,
        settings_domain=settings.domain,
        onnx_path=onnx_path,
        normalization_rel="features/normalization_header.json",
        vocab_rel="features/vocab.json",
    )
    write_json(out_dir / "export_manifest.json", manifest)

    write_parity_samples(
        wrapper,
        out_dir,
        manifest_path=cfg.paths.manifest_train,
        num_samples=num_parity_samples,
    )

    if not skip_verify:
        verify_info = verify_onnx(
            onnx_path,
            header_dim=meta["header_dim"],
            bow_dim=meta["bow_dim"],
        )
        manifest["onnx_runtime_check"] = verify_info
        write_json(out_dir / "export_manifest.json", manifest)

    return out_dir


def load_parity_vectors(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    npz_path = bundle_dir / "parity_samples" / "sample_vectors.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")
    data = np.load(npz_path)
    headers = np.asarray(data["headers"], dtype=np.float64)
    bows = np.asarray(data["bows"], dtype=np.float64)
    expected = np.asarray(data["expected_scores"], dtype=np.float64)
    sample_ids = [str(s) for s in data["sample_ids"].tolist()]
    return headers, bows, expected, sample_ids


def run_onnx_scores(
    session: Any,
    headers: np.ndarray,
    bows: np.ndarray,
) -> np.ndarray:
    out_name = session.get_outputs()[0].name
    scores: list[float] = []
    for i in range(headers.shape[0]):
        h = headers[i : i + 1].astype(np.float32)
        b = bows[i : i + 1].astype(np.float32)
        out = session.run([out_name], {"header": h, "bow": b})[0]
        scores.append(float(np.asarray(out).ravel()[0]))
    return np.asarray(scores, dtype=np.float64)


@torch.no_grad()
def run_pytorch_scores(
    wrapper: OnnxInferenceWrapper,
    headers: np.ndarray,
    bows: np.ndarray,
) -> np.ndarray:
    scores: list[float] = []
    for i in range(headers.shape[0]):
        h = torch.from_numpy(headers[i : i + 1]).float()
        b = torch.from_numpy(bows[i : i + 1]).float()
        scores.append(float(wrapper(h, b).item()))
    return np.asarray(scores, dtype=np.float64)


def run_parity_check(
    bundle_dir: Path,
    *,
    checkpoint: Path,
    tolerance: float = DEFAULT_TOLERANCE,
    config_path: Path | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "export_manifest.json"
    onnx_path = bundle_dir / "model.onnx"
    if not manifest_path.is_file() or not onnx_path.is_file():
        raise FileNotFoundError(f"Incomplete export bundle: {bundle_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    headers, bows, export_expected, sample_ids = load_parity_vectors(bundle_dir)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_scores = run_onnx_scores(session, headers, bows)
    wrapper, _, _ = load_wrapped_model(checkpoint, config_path=config_path)
    pytorch_scores = run_pytorch_scores(wrapper, headers, bows)

    pt_onnx_diff = np.abs(pytorch_scores - onnx_scores)
    onnx_export_diff = np.abs(onnx_scores - export_expected)

    per_sample = [
        {
            "sample_id": sid,
            "pytorch": float(pytorch_scores[i]),
            "onnx": float(onnx_scores[i]),
            "export_expected": float(export_expected[i]),
            "abs_diff_pytorch_onnx": float(pt_onnx_diff[i]),
            "abs_diff_onnx_export": float(onnx_export_diff[i]),
        }
        for i, sid in enumerate(sample_ids)
    ]

    max_pt_onnx = float(pt_onnx_diff.max())
    report: dict[str, Any] = {
        "timestamp": _utc_now(),
        "model_id": manifest.get("model_id", "pattern_a_combined"),
        "bundle_dir": str(bundle_dir),
        "checkpoint": str(checkpoint.resolve()),
        "onnx_file": str(onnx_path),
        "tolerance": tolerance,
        "passed": max_pt_onnx < tolerance,
        "n_samples": int(headers.shape[0]),
        "pytorch_vs_onnx": {
            "max_abs_diff": max_pt_onnx,
            "mean_abs_diff": float(pt_onnx_diff.mean()),
            "per_sample": per_sample,
        },
        "onnx_vs_export_expected": {
            "max_abs_diff": float(onnx_export_diff.max()),
            "mean_abs_diff": float(onnx_export_diff.mean()),
        },
    }
    return report


def write_parity_outputs(
    report: dict[str, Any],
    bundle_dir: Path,
    *,
    local_parity_dir: Path,
) -> Path:
    local_parity_dir.mkdir(parents=True, exist_ok=True)
    report_path = local_parity_dir / "parity_report.json"
    write_json(report_path, report)
    npz_src = bundle_dir / "parity_samples" / "sample_vectors.npz"
    shutil.copy2(npz_src, local_parity_dir / "sample_vectors.npz")
    return report_path
