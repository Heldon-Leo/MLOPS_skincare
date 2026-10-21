"""
Train Ingredient Function Classifier — Standalone Script
=========================================================
Trains the character-level TF-IDF + Logistic Regression model for individual
ingredient function classification. Tracks experiment with MLflow, saves
model artifact, and generates full per-class evaluation report.

Usage:
    python scripts/train_ingredient_classifier.py

Output:
    models/ingredient_classifier.joblib
    reports/ingredient_classifier_metrics.json
    reports/ingredient_classifier_classification_report.txt
    MLflow run under experiment 'skincare_ingredient_classifier'
"""

import os
import sys
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ingredient_taxonomy.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_DEST = os.path.join(MODELS_DIR, "ingredient_classifier.joblib")
MLFLOW_DB = os.path.join(PROJECT_ROOT, "mlflow.db")
EXPERIMENT_NAME = "skincare_ingredient_classifier"

# ---- Hyperparameters ----
TFIDF_ANALYZER = "char_wb"
NGRAM_MIN = 2
NGRAM_MAX = 4
LR_C = 2.0
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 1000
CONFIDENCE_THRESHOLD = 0.25
RANDOM_STATE = 42
TEST_SIZE = 0.25


def main():
    if not os.path.exists(TAXONOMY_PATH):
        raise FileNotFoundError(
            f"Ingredient taxonomy CSV not found at:\n  {TAXONOMY_PATH}\n"
            "Run `python src/data/ingredient_taxonomy.py` first."
        )

    df = pd.read_csv(TAXONOMY_PATH)
    print(f"\nLoaded taxonomy: {len(df)} records, {df['functional_category'].nunique()} classes")
    print("Class distribution:")
    print(df["functional_category"].value_counts().to_string())

    # Guard: ensure no exact duplicate ingredient names bleed across splits
    df = df.drop_duplicates(subset=["ingredient_name"])
    X = df["ingredient_name"]
    y = df["functional_category"]
    classes = sorted(y.unique().tolist())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=TFIDF_ANALYZER,
            ngram_range=(NGRAM_MIN, NGRAM_MAX),
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            C=LR_C,
            solver=LR_SOLVER,
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE,
            multi_class="multinomial"
        ))
    ])

    # Cross-validation on train fold (to prevent leakage with small dataset)
    skf = StratifiedKFold(n_splits=min(5, y_train.value_counts().min()), shuffle=True, random_state=RANDOM_STATE)
    cv_f1_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro")
    print(f"\nCV Macro-F1 (train fold): {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")

    # Fit on train, evaluate on held-out val
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    acc = float(accuracy_score(y_val, y_pred))
    f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    cls_report = classification_report(y_val, y_pred, zero_division=0)

    print(f"\nValidation Accuracy : {acc:.4f}")
    print(f"Validation Macro-F1 : {f1_macro:.4f}")
    print(f"Validation Weighted-F1: {f1_weighted:.4f}")
    print(f"\nClassification Report:\n{cls_report}")

    # Retrain on full dataset for maximum production vocabulary coverage
    pipeline.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_DEST)
    print(f"\nSaved ingredient classifier → {MODEL_DEST}")

    # Save metrics JSON
    metrics_data = {
        "model_type": "LogisticRegression_CharNGram_TFIDF",
        "feature_config": {
            "analyzer": TFIDF_ANALYZER,
            "ngram_range": [NGRAM_MIN, NGRAM_MAX],
            "sublinear_tf": True
        },
        "hyperparameters": {
            "C": LR_C,
            "solver": LR_SOLVER,
            "max_iter": LR_MAX_ITER
        },
        "supported_classes": classes,
        "sample_count": len(df),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "validation_accuracy": round(acc, 4),
        "validation_macro_f1": round(f1_macro, 4),
        "validation_weighted_f1": round(f1_weighted, 4),
        "cv_macro_f1_mean": round(float(cv_f1_scores.mean()), 4),
        "cv_macro_f1_std": round(float(cv_f1_scores.std()), 4),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "limitations": (
            "Trained on a curated 77-record INCI taxonomy. Performance is bounded by small dataset size. "
            "Unknown or novel ingredients with confidence below threshold return 'Unknown / Insufficient evidence'. "
            "Does not generalize claims to all cosmetic ingredients."
        )
    }

    metrics_path = os.path.join(REPORTS_DIR, "ingredient_classifier_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Save per-class classification report
    report_path = os.path.join(REPORTS_DIR, "ingredient_classifier_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Ingredient Function Classifier — Per-Class Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(df)} records | Classes: {len(classes)}\n")
        f.write(f"Split: {len(X_train)} train / {len(X_val)} validation (stratified)\n\n")
        f.write(cls_report)
        f.write(f"\nCV Macro-F1: {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f} (k={skf.n_splits} folds)\n")

    # MLflow tracking
    try:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="ingredient_classifier_LR_charngram"):
            # Log parameters
            mlflow.log_param("tfidf_analyzer", TFIDF_ANALYZER)
            mlflow.log_param("ngram_range", f"{NGRAM_MIN}-{NGRAM_MAX}")
            mlflow.log_param("lr_C", LR_C)
            mlflow.log_param("lr_solver", LR_SOLVER)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("num_classes", len(classes))

            # Log metrics
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_macro_f1", f1_macro)
            mlflow.log_metric("val_weighted_f1", f1_weighted)
            mlflow.log_metric("cv_macro_f1_mean", float(cv_f1_scores.mean()))
            mlflow.log_metric("cv_macro_f1_std", float(cv_f1_scores.std()))

            # Log artifacts
            mlflow.log_artifact(MODEL_DEST)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(report_path)

            # Log model
            mlflow.sklearn.log_model(pipeline, artifact_path="ingredient_classifier_model")

            print(f"\nMLflow run logged to experiment: '{EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"[MLflow] Tracking failed (non-critical): {e}")

    print("\n✅ Ingredient classifier training complete.")
    print(f"   Model  : {MODEL_DEST}")
    print(f"   Metrics: {metrics_path}")
    print(f"   Report : {report_path}")
    return metrics_data


if __name__ == "__main__":
    main()
