"""Train LinearSVC and Tiny MLP; pick best on val F1."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score

from src.config import ensure_artifact_dirs, load_config
from src.constants import DOMAIN_ID, MODEL_ID
from src.data.dataset import stack_split_arrays
from src.models.linear_svm import malware_probabilities, train_linear_svc
from src.models.tiny_mlp import LinearSigmoidModule, TinyMlpModule, train_tiny_mlp

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _val_f1_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    preds = (probs >= threshold).astype(np.int64)
    return float(f1_score(y_true, preds, zero_division=0))


def train_and_select(cfg) -> Path:
    threshold = float(cfg.evaluation.get("threshold", 0.5))
    processed = cfg.paths.processed
    X_train, y_train = stack_split_arrays(processed, "train")
    X_val, y_val = stack_split_arrays(processed, "val")

    candidates = cfg.model.get("candidates", ["linear_svc", "tiny_mlp"])
    svc_cfg = cfg.model.get("linear_svc", {})
    mlp_cfg = cfg.model.get("tiny_mlp", {})

    results: dict[str, dict] = {}
    best_name = ""
    best_f1 = -1.0

    svc_model = None
    mlp_model = None

    if "linear_svc" in candidates:
        svc_model = train_linear_svc(
            X_train,
            y_train,
            C=float(svc_cfg.get("C", 1.0)),
            class_weight=svc_cfg.get("class_weight", "balanced"),
        )
        val_probs = malware_probabilities(svc_model, X_val)
        val_f1 = _val_f1_from_probs(y_val, val_probs, threshold)
        results["linear_svc"] = {"val_f1": val_f1}
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = "linear_svc"

    if "tiny_mlp" in candidates:
        mlp_model = train_tiny_mlp(
            X_train,
            y_train,
            hidden_dim=int(mlp_cfg.get("hidden_dim", 32)),
            learning_rate=float(mlp_cfg.get("learning_rate", 0.01)),
            epochs=int(mlp_cfg.get("epochs", 100)),
            batch_size=int(mlp_cfg.get("batch_size", 256)),
        )
        with torch.no_grad():
            val_probs = mlp_model(torch.from_numpy(X_val.astype(np.float32))).numpy().reshape(-1)
        val_f1 = _val_f1_from_probs(y_val, val_probs, threshold)
        results["tiny_mlp"] = {"val_f1": val_f1}
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = "tiny_mlp"

    if not best_name:
        raise RuntimeError("No candidate models trained")

    checkpoint_dir = cfg.paths.checkpoints
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if best_name == "linear_svc":
        assert svc_model is not None
        joblib.dump(svc_model, checkpoint_dir / "model.joblib")
        export_module = LinearSigmoidModule.from_linear_svc(svc_model, X_train.shape[1])
    else:
        assert mlp_model is not None
        export_module = mlp_model

    ckpt = {
        "model_state_dict": export_module.state_dict(),
        "model_type": best_name,
        "feature_dim": int(X_train.shape[1]),
        "hidden_dim": int(mlp_cfg.get("hidden_dim", 32)) if best_name == "tiny_mlp" else None,
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "val_f1": best_f1,
        "candidate_scores": results,
        "trained_at": _utc_now(),
    }
    ckpt_path = cfg.paths.latest_checkpoint
    torch.save(ckpt, ckpt_path)

    meta = {
        "model_id": MODEL_ID,
        "selected_model": best_name,
        "val_f1": best_f1,
        "candidate_scores": results,
        "n_train": int(len(y_train)),
        "S": int(X_train.shape[1]),
        "trained_at": _utc_now(),
    }
    (checkpoint_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    try:
        from src.thesis_archive import after_train

        after_train(ckpt_path, meta)
    except ImportError:
        pass

    print(f"Selected model: {best_name} (val F1={best_f1:.4f})")
    for name, scores in results.items():
        print(f"  {name}: val_f1={scores['val_f1']:.4f}")
    return ckpt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train MLDP classifiers and select best on val F1.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    ckpt = train_and_select(cfg)
    print(f"Checkpoint → {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
