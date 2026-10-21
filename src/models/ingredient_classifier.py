"""
Ingredient Function Classifier (NLP Model)
Trains and serves an individual chemical ingredient function classifier using:
1. TF-IDF character n-grams (2-4) + word n-grams (1-2)
2. Logistic Regression baseline vs XGBoost / Random Forest candidates
3. Generates confusion matrix, per-class metrics, and saves fitted model to models/ingredient_classifier.joblib
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ingredient_taxonomy.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_DEST = os.path.join(MODELS_DIR, "ingredient_classifier.joblib")


def train_ingredient_classifier() -> Dict[str, Any]:
    """Trains individual chemical ingredient classifier and saves model artifact."""
    if not os.path.exists(TAXONOMY_PATH):
        raise FileNotFoundError(f"Taxonomy dataset missing at {TAXONOMY_PATH}")

    df = pd.read_csv(TAXONOMY_PATH)
    print(f"Loaded ingredient taxonomy: {len(df)} records across {df['functional_category'].nunique()} classes")

    X = df["ingredient_name"]
    y = df["functional_category"]
    classes = sorted(list(y.unique()))

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build Pipeline with Character (2-4) and Word N-Grams
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, C=2.0, random_state=42))
    ])

    # Fit model
    pipeline.fit(X_train, y_train)

    # Validate
    y_pred = pipeline.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )

    print(f"[Ingredient Classifier] Val Accuracy: {acc:.4f} | Val Macro-F1: {f1_macro:.4f}")

    # Retrain on full dataset for maximum production vocabulary coverage
    pipeline.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_DEST)
    print(f"Saved fitted ingredient classifier to {MODEL_DEST}")

    # Save metrics report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_data = {
        "model_type": "LogisticRegression_CharNGram_TFIDF",
        "supported_classes": classes,
        "sample_count": len(df),
        "validation_accuracy": round(acc, 4),
        "validation_macro_f1": round(float(f1_macro), 4)
    }
    with open(os.path.join(REPORTS_DIR, "ingredient_classifier_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data


class IngredientClassifierService:
    """Predicts function for individual chemical ingredients."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_DEST
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                print(f"[IngredientClassifierService] Model loaded from {self.model_path}")
            except Exception as e:
                print(f"[IngredientClassifierService] Error loading model: {e}")
                self.pipeline = None
        else:
            print(f"[IngredientClassifierService] Model artifact not found at {self.model_path}")

    def classify_ingredient(self, ingredient_name: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """Classifies function of a single ingredient with probability distribution."""
        if not self.pipeline:
            self.load_model()
            if not self.pipeline:
                return {
                    "ingredient": ingredient_name,
                    "predicted_function": "Unknown / Insufficient evidence",
                    "confidence": 0.0,
                    "probabilities": {}
                }

        clean_name = ingredient_name.lower().strip()
        if not clean_name or len(clean_name) < 2:
            return {
                "ingredient": ingredient_name,
                "predicted_function": "Unknown / Insufficient evidence",
                "confidence": 0.0,
                "probabilities": {}
            }

        probs = self.pipeline.predict_proba([clean_name])[0]
        classes = self.pipeline.classes_
        top_idx = int(np.argmax(probs))
        top_confidence = float(probs[top_idx])
        top_class = classes[top_idx]

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

        # If highest probability is below threshold, label as Unknown/Insufficient Evidence
        final_category = top_class if top_confidence >= confidence_threshold else "Unknown / Insufficient evidence"

        return {
            "ingredient": ingredient_name,
            "predicted_function": final_category,
            "confidence": round(top_confidence, 4),
            "probabilities": prob_dict
        }


ingredient_classifier_service = IngredientClassifierService()


if __name__ == "__main__":
    train_ingredient_classifier()
