"""P6 — evaluate fusion model on test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.data.dataloaders import build_dataloaders
from src.data.dataset import FusionDataset
from src.data.store import load_split_shards
from src.training.checkpoint import load_best_checkpoint, load_frozen_layout, restore_model_weights
from src.training.metrics import compute_metrics, format_metrics
from src.training.setup import build_fusion_model

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_test(cfg: PipelineConfig) -> dict:
    ensure_artifact_dirs(cfg)
    shards = load_split_shards(cfg)
    test_shard = shards["test"]
    test_ds = FusionDataset.from_shard(test_shard)

    _, val_loader, test_loader, dex_dim, receiver_dim, _ = build_dataloaders(cfg)

    ckpt_path = cfg.paths.checkpoints / "best.pt"
    payload = load_best_checkpoint(ckpt_path)
    model = build_fusion_model(cfg, dex_dim=dex_dim, receiver_dim=receiver_dim)
    restore_model_weights(model, payload)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    tuned_path = cfg.paths.metrics / "thresholds.json"
    default_t = float(cfg.evaluation.get("threshold", 0.5))
    threshold = default_t
    val_shard = shards["val"]
    val_score_batches: list[np.ndarray] = []
    val_label_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_h, batch_r, batch_y in val_loader:
            logits = model(batch_h.to(device), batch_r.to(device))
            scores = torch.sigmoid(logits).view(-1).cpu().numpy()
            val_score_batches.append(scores)
            val_label_batches.append(batch_y.numpy())
    y_val = np.concatenate(val_label_batches).astype(int)
    s_val = np.concatenate(val_score_batches)

    if bool(cfg.evaluation.get("tune_threshold_on_val", True)):
        from sklearn.metrics import f1_score

        best_t = default_t
        best_f1 = -1.0
        for t in np.linspace(0.05, 0.95, 19):
            preds = (s_val >= t).astype(int)
            f1 = float(f1_score(y_val, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        threshold = best_t
        tuned_path.write_text(
            json.dumps({"default": default_t, "tuned_val": threshold}, indent=2) + "\n",
            encoding="utf-8",
        )

    y_true_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch_h, batch_r, batch_y in test_loader:
            logits = model(batch_h.to(device), batch_r.to(device))
            scores = torch.sigmoid(logits).view(-1).cpu().numpy()
            y_true_list.append(batch_y.numpy())
            y_score_list.append(scores)

    y_true = np.concatenate(y_true_list).astype(int)
    y_score = np.concatenate(y_score_list)
    y_pred = (y_score >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_score)
    print(f"Test @ threshold={threshold:.3f}: {format_metrics(metrics)}")

    receiver_vocab, layout = load_frozen_layout(cfg.paths.processed)
    cm = [
        [int(metrics.get("tn", 0)), int(metrics.get("fp", 0))],
        [int(metrics.get("fn", 0)), int(metrics.get("tp", 0))],
    ]
    results = {
        "model_id": cfg.model_id,
        "split": "test",
        "train_years": cfg.splits.get("train_years"),
        "test_years": cfg.splits.get("holdout_years"),
        "n_samples": int(y_true.shape[0]),
        "feature_dims": {
            "dex_header": dex_dim,
            "receiver": receiver_dim,
            "d_R": int(cfg.model.get("receiver_embed_dim", 32)),
            "fused": 128 + int(cfg.model.get("receiver_embed_dim", 32)),
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics.get("roc_auc"),
        },
        "confusion_matrix": cm,
        "threshold": threshold,
        "ablations": {
            "header_only": {
                "f1": None,
                "note": "ref: deployed mlp_header measured metrics (BM1)",
            },
            "receivers_only": {"f1": None},
            "fusion": {"f1": metrics["f1"]},
        },
        "paper_baseline": {},
    }
    svm_path = cfg.paths.metrics / "svm_metrics.json"
    if svm_path.is_file():
        svm = json.loads(svm_path.read_text(encoding="utf-8"))
        if "test" in svm:
            results["paper_baseline"]["svm_rbf_concat"] = {"f1": svm["test"].get("f1")}

    out_path = cfg.paths.metrics / "test_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    from shared_calibration import find_repo_root, write_split_scores_bundle

    write_split_scores_bundle(
        model_id=cfg.model_id,
        split="val",
        metrics_dir=cfg.paths.metrics,
        apk_ids=list(val_shard.sha256),
        labels=y_val.astype(np.int64),
        scores=s_val.astype(np.float64),
        threshold=threshold,
        repo_root=find_repo_root(_PACKAGE_ROOT),
    )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 test evaluation.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    evaluate_test(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
