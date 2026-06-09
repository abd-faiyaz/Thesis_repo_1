#!/usr/bin/env python3
"""Evaluate legacy ByteCNN and XGBoost on shared thesis splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from common import (  # noqa: E402
    compute_metrics,
    export_offline_json,
    label_from_rel_path,
    load_dataset_paths,
    read_split_paths,
    repo_root,
    resolve_apk_path,
    write_test_results,
)
from xgb_features import build_xgb_vector, load_feature_index  # noqa: E402

BYTECNN_MODEL_ID = "bytecnn"
BYTECNN_DOMAIN = "bytecnn"
MANIFEST_XGB_MODEL_ID = "manifest_xgb"
MANIFEST_XGB_DOMAIN = "manifest_xgb"
DEFAULT_THRESHOLD = 0.5
BYTE_LENGTH = 1024


def _load_tail_bytes(apk_path: Path, length: int = BYTE_LENGTH) -> np.ndarray:
    data = apk_path.read_bytes()
    if len(data) >= length:
        segment = data[-length:]
    else:
        segment = data.rjust(length, b"\0")
    return np.frombuffer(segment, dtype=np.uint8).astype(np.int64)


def _softmax_malware_prob(logits: np.ndarray) -> float:
    exp0 = float(np.exp(logits[0]))
    exp1 = float(np.exp(logits[1]))
    return exp1 / (exp0 + exp1)


def evaluate_bytecnn(
    *,
    apk_paths: list[Path],
    labels: np.ndarray,
    model_path: Path,
    threshold: float,
    use_onnx: bool,
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    if use_onnx:
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        for apk_path in tqdm(apk_paths, desc="bytecnn"):
            x = _load_tail_bytes(apk_path)[None, :]
            logits = session.run(None, {input_name: x})[0][0]
            scores.append(_softmax_malware_prob(logits))
    else:
        sys.path.insert(0, str(repo_root() / "1dcnn/src"))
        from model.bytecnn import ByteCNN  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ByteCNN().to(device)
        state = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            for apk_path in tqdm(apk_paths, desc="bytecnn"):
                x = torch.from_numpy(_load_tail_bytes(apk_path)).unsqueeze(0).to(device)
                logits = model(x).squeeze(0).cpu().numpy()
                scores.append(_softmax_malware_prob(logits))

    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)
    return y_pred, y_score


def evaluate_manifest_xgb(
    *,
    apk_paths: list[Path],
    labels: np.ndarray,
    model_path: Path,
    features_path: Path,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    feature_index = load_feature_index(features_path)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    scores: list[float] = []
    for apk_path in tqdm(apk_paths, desc="manifest_xgb"):
        vector = np.asarray(build_xgb_vector(apk_path, feature_index), dtype=np.float32)
        outputs = session.run(None, {input_name: vector[None, :]})
        probabilities = outputs[1][0]
        scores.append(float(probabilities[1]))
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)
    return y_pred, y_score


def _default_paths(root: Path, model: str) -> dict[str, Path]:
    if model == BYTECNN_MODEL_ID:
        onnx = root / "1dcnn/bytecnn_basemodel_2020.onnx"
        pth = root / "1dcnn/bytecnn_basemodel_2020.pth"
        model_path = onnx if onnx.is_file() else pth
        return {
            "model_path": model_path,
            "artifacts_dir": root / "legacy_models/artifacts/bytecnn/metrics",
            "use_onnx": model_path.suffix == ".onnx",
        }
    features = root / "vigidroid/app/src/main/assets/mh1m_2500_rp_features.json.gzip"
    onnx = root / "vigidroid/app/src/main/assets/mh1m_2500_rp_XGBoost.onnx"
    return {
        "model_path": onnx,
        "features_path": features,
        "artifacts_dir": root / "legacy_models/artifacts/manifest_xgb/metrics",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate legacy ByteCNN or XGBoost on thesis splits.")
    parser.add_argument(
        "--model",
        choices=[BYTECNN_MODEL_ID, MANIFEST_XGB_MODEL_ID],
        required=True,
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--apk-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N APKs (smoke test).")
    parser.add_argument("--no-offline-export", action="store_true")
    args = parser.parse_args(argv)

    root = repo_root()
    cfg = load_dataset_paths(root)
    apk_root = Path(args.apk_root or cfg["apk_root"])
    split_key = f"{args.split}_split"
    split_file = Path(args.split_file or cfg[split_key])
    defaults = _default_paths(root, args.model)

    rel_paths = read_split_paths(split_file)
    if args.limit > 0:
        rel_paths = rel_paths[: args.limit]

    apk_paths: list[Path] = []
    labels: list[int] = []
    missing: list[str] = []
    for rel in rel_paths:
        apk_path = resolve_apk_path(apk_root, rel)
        if not apk_path.is_file():
            missing.append(rel)
            continue
        apk_paths.append(apk_path)
        labels.append(label_from_rel_path(rel))

    if missing:
        print(f"Warning: {len(missing)} APK(s) missing under {apk_root}", file=sys.stderr)
    if not apk_paths:
        print("No APKs found for evaluation.", file=sys.stderr)
        return 1

    y_true = np.asarray(labels, dtype=int)
    model_path = Path(args.model_path or defaults["model_path"])
    if args.model == BYTECNN_MODEL_ID:
        y_pred, y_score = evaluate_bytecnn(
            apk_paths=apk_paths,
            labels=y_true,
            model_path=model_path,
            threshold=args.threshold,
            use_onnx=bool(defaults.get("use_onnx", model_path.suffix == ".onnx")),
        )
        model_id = BYTECNN_MODEL_ID
        domain = BYTECNN_DOMAIN
    else:
        features_path = Path(args.features_path or defaults["features_path"])
        y_pred, y_score = evaluate_manifest_xgb(
            apk_paths=apk_paths,
            labels=y_true,
            model_path=model_path,
            features_path=features_path,
            threshold=args.threshold,
        )
        model_id = MANIFEST_XGB_MODEL_ID
        domain = MANIFEST_XGB_DOMAIN

    metrics = compute_metrics(y_true, y_pred, y_score)
    artifacts_dir = defaults["artifacts_dir"]
    out_name = "test_results.json" if args.split == "test" else f"{args.split}_results.json"
    scores_name = "test_scores.json" if args.split == "test" else f"{args.split}_scores.json"
    results_path = write_test_results(
        out_path=artifacts_dir / out_name,
        model_id=model_id,
        domain=domain,
        split=args.split,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        threshold=args.threshold,
        checkpoint_path=str(model_path),
        n_samples=len(apk_paths),
        extra={
            "apk_root": str(apk_root),
            "split_file": str(split_file),
            "missing_apks": len(missing),
        },
    )

    from shared_calibration.val_scores import (  # type: ignore
        apk_ids_from_paths,
        build_split_scores_payload,
        write_split_scores,
    )

    apk_ids = apk_ids_from_paths([str(path) for path in apk_paths])
    scores_payload = build_split_scores_payload(
        model_id=model_id,
        split=args.split,
        apk_ids=apk_ids,
        labels=y_true,
        scores=y_score,
        threshold=args.threshold,
    )
    scores_path = artifacts_dir / scores_name
    write_split_scores(scores_path, scores_payload)

    print(f"  split={args.split} n={len(apk_paths)} missing={len(missing)}")
    print(
        f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
        f"auc={metrics['roc_auc']}"
    )
    print(f"  metrics → {results_path}")
    print(f"  scores  → {scores_path}")

    if not args.no_offline_export and args.split == "test":
        offline_path = export_offline_json(
            model_id=model_id,
            domain=domain,
            split=args.split,
            metrics=metrics,
            n_samples=len(apk_paths),
            threshold=args.threshold,
            y_true=y_true,
            y_pred=y_pred,
            checkpoint_path=str(model_path),
            root=root,
        )
        print(f"  offline → {offline_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
