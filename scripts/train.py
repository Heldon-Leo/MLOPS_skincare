"""
Model Training and Comparison Pipeline
Trains Logistic Regression baseline and Random Forest classifier on real cosmetics features.
Logs parameters and metrics via tracking module, compares held-out validation metrics,
and serializes the winning pipeline to disk.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.features import build_feature_pipeline
from src.models.tracking import ExperimentTracker
from src.evaluation.evaluator import evaluate_model_performance

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def train_and_compare_models():
    """Executes multi-model training, tracking, and evaluation workflow."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load train, val, and test splits
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    target_col = "Label"
    classes = sorted(list(train_df[target_col].unique()))
    print(f"Dataset classes ({len(classes)}): {classes}")

    # Build candidates
    candidates = [
        {
            "name": "LogisticRegression_Baseline",
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "params": {"model_type": "LogisticRegression", "C": 1.0, "max_iter": 1000}
        },
        {
            "name": "RandomForest_Classifier",
            "model": RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42),
            "params": {"model_type": "RandomForest", "n_estimators": 150, "max_depth": 20}
        }
    ]

    results = []
    best_pipeline = None
    best_val_f1 = -1.0
    best_candidate_name = ""

    for candidate in candidates:
        name = candidate["name"]
        print(f"\n--- Training Candidate: {name} ---")
        tracker = ExperimentTracker(experiment_name="skincare_routine_category")
        tracker.start_run(run_name=name)
        tracker.log_params(candidate["params"])
        tracker.log_param("train_samples", len(train_df))
        tracker.log_param("val_samples", len(val_df))

        # Pipeline: Features -> Classifier
        pipe = Pipeline([
            ("features", build_feature_pipeline()),
            ("classifier", candidate["model"])
        ])

        # Fit
        pipe.fit(train_df, train_df[target_col])

        # Evaluate on Validation
        val_preds = pipe.predict(val_df)
        val_metrics = evaluate_model_performance(
            val_df[target_col], val_preds, classes, model_name=f"{name}_Validation"
        )

        tracker.log_metrics({
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"]
        })
        tracker.end_run(status="COMPLETED")

        results.append({
            "name": name,
            "pipeline": pipe,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"]
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_pipeline = pipe
            best_candidate_name = name

    print(f"\nModel Selection: Winning candidate is {best_candidate_name} with Val Macro-F1 = {best_val_f1:.4f}")

    # Evaluate best model on Held-Out Test Set
    test_preds = best_pipeline.predict(test_df)
    test_metrics = evaluate_model_performance(
        test_df[target_col], test_preds, classes, model_name=best_candidate_name
    )

    # Save fitted preprocessing + model together as serialized artifact
    candidate_model_path = os.path.join(MODELS_DIR, "candidate_pipeline.joblib")
    joblib.dump(best_pipeline, candidate_model_path)
    print(f"Saved candidate model pipeline to: {candidate_model_path}")

    # Generate reports/training_summary.md
    summary_path = os.path.join(REPORTS_DIR, "training_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Model Training & Experiment Summary\n\n")
        f.write(f"- **Execution Timestamp**: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"- **Selected Architecture**: `{best_candidate_name}`\n")
        f.write(f"- **Held-Out Test Accuracy**: `{test_metrics['accuracy']:.4f}`\n")
        f.write(f"- **Held-Out Test Macro-F1**: `{test_metrics['macro_f1']:.4f}`\n")
        f.write(f"- **Held-Out Test Weighted-F1**: `{test_metrics['weighted_f1']:.4f}`\n\n")
        f.write("## Candidate Model Comparison (Validation Set)\n\n")
        f.write("| Model Candidate | Val Accuracy | Val Macro-F1 |\n")
        f.write("|---|---|---|\n")
        for res in results:
            f.write(f"| {res['name']} | {res['val_accuracy']:.4f} | {res['val_macro_f1']:.4f} |\n")
        f.write("\n## Per-Class Test Performance\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for cls, vals in test_metrics["per_class"].items():
            f.write(f"| {cls} | {vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1']:.4f} | {vals['support']} |\n")
        f.write("\n## Class Imbalance & Uncertainty Evaluation\n")
        f.write(f"- Maximum to minimum support class ratio: `{test_metrics['class_imbalance_analysis']['support_ratio']}`\n")
        f.write(f"- Highest representation: `{test_metrics['class_imbalance_analysis']['max_support_class']}`\n")
        f.write(f"- Lowest representation: `{test_metrics['class_imbalance_analysis']['min_support_class']}`\n")

    return best_candidate_name, test_metrics


if __name__ == "__main__":
    train_and_compare_models()
