import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from pathlib import Path
from model.bytecnn import ByteCNN
from logger import get_logger


logger = get_logger("Model Evaluate")


def get_apk_files(year_dir):
    apk_files = []
    for label_folder in ["benign", "malware"]:
        folder = year_dir / label_folder
        if folder.exists():
            apk_files.extend(folder.glob("*.apk"))
    return apk_files


def load_bytes_segment(path, length, from_end=True):
    with open(path, "rb") as f:
        if from_end:
            f.seek(0, 2) 
            size = f.tell()
            if size >= length:
                f.seek(size - length)
                segment = f.read(length)
            else:
                f.seek(0)
                segment = f.read().rjust(length, b"\0")
        else:
            segment = f.read(length).ljust(length, b"\0")
    return segment


def tensorize_bytes(data, device):
    arr = torch.tensor(list(data), dtype=torch.long, device=device)
    return arr.unsqueeze(0)


def predict_file(model, path, length, from_end, device):
    segment = load_bytes_segment(path, length, from_end)
    x = tensorize_bytes(segment, device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    return probs[0], probs[1]


def evaluate_model(year_dir, model_path, report_path, byte_length=1024, threshold=0.5, from_end=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ByteCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    year_dir = Path(year_dir)
    apk_files = get_apk_files(year_dir)

    results = []

    for apk_path in apk_files:
        print("Analyzing ", apk_path.name)
        true_label = 0 if apk_path.parent.name.lower() == "benign" else 1
        
        with torch.no_grad():
            good_p, bad_p = predict_file(
                model, apk_path, length=byte_length, from_end=from_end, device=device
            )

        predicted_label = 1 if bad_p >= threshold else 0

        results.append({
            "filename": apk_path.stem, 
            "true_label": true_label,
            "predicted_label": predicted_label,
            "good_p": good_p,
            "bad_p": bad_p
        })
    
    df = pd.DataFrame(results)
    df.to_csv(report_path, index=False)

    y_true = df["true_label"]
    y_pred = df["predicted_label"]
    y_score = df["bad_p"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = None

    return metrics


    