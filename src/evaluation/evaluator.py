"""
Evaluation Module
Computes per-class and macro classification metrics, confusion matrix,
and exports reports/model_metrics.json, reports/classification_report.txt,
and reports/confusion_matrix.png.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def evaluate_model_performance(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
    model_name: str = "CandidateModel"
) -> Dict[str, Any]:
    """
    Evaluates multi-class predictions, generates textual and graphic reports,
    and returns metrics dictionary.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class_dict = {}
    for i, cls in enumerate(classes):
        per_class_dict[cls] = {
            "precision": round(float(p_per_class[i]), 4),
            "recall": round(float(r_per_class[i]), 4),
            "f1": round(float(f1_per_class[i]), 4),
            "support": int(support[i])
        }

    report_str = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    metrics_payload = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class_dict,
        "class_imbalance_analysis": {
            "min_support_class": classes[int(np.argmin(support))],
            "max_support_class": classes[int(np.argmax(support))],
            "support_ratio": round(float(np.max(support) / max(1, np.min(support))), 2)
        }
    }

    # 1. Save reports/model_metrics.json
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 2. Save reports/classification_report.txt
    report_txt_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Model Evaluation: {model_name} ===\n\n")
        f.write(report_str)

    # 3. Save reports/confusion_matrix.png
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: {model_name}",
        ylabel="True Category",
        xlabel="Predicted Category"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"Evaluation artifacts generated successfully in {REPORTS_DIR}:")
    print(f"  Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f}")
    return metrics_payload
