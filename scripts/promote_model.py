"""
Model Promotion Gate and Versioning Script
Enforces formal gate criteria:
1. Candidate accuracy >= min_accuracy (threshold 0.40 for 6-class balanced random baseline of 0.166)
2. Candidate Macro-F1 >= min_macro_f1 (threshold 0.40)
3. Zero fatal pipeline check errors on sample inputs

If criteria pass:
- Promotes candidate_pipeline.joblib -> approved_model_v{version}.joblib
- Updates models/production_model.joblib (symbolic/canonical active version)
- Writes models/model_registry.json with metadata, metrics, and rollback history
"""

import os
import sys
import json
import shutil
import joblib
import pandas as pd
import sys
from datetime import datetime
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

# Documented Promotion Criteria
MIN_ACCURACY = 0.40
MIN_MACRO_F1 = 0.40


def load_registry() -> Dict[str, Any]:
    """Loads existing model registry or initializes a new one."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "current_production_version": None,
        "models": [],
        "rollback_history": []
    }


def promote_candidate_model() -> bool:
    """Evaluates candidate metrics against criteria and updates model registry."""
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    candidate_path = os.path.join(MODELS_DIR, "candidate_pipeline.joblib")

    if not os.path.exists(metrics_path) or not os.path.exists(candidate_path):
        print(f"Error: Candidate model or metrics file missing.")
        return False

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    acc = metrics.get("accuracy", 0.0)
    macro_f1 = metrics.get("macro_f1", 0.0)
    model_name = metrics.get("model_name", "UnknownModel")

    print(f"Evaluating candidate '{model_name}':")
    print(f"  Accuracy: {acc:.4f} (Threshold: >={MIN_ACCURACY})")
    print(f"  Macro-F1: {macro_f1:.4f} (Threshold: >={MIN_MACRO_F1})")

    # Gate verification
    if acc < MIN_ACCURACY or macro_f1 < MIN_MACRO_F1:
        print("Model promotion REJECTED: Candidate failed minimum quality criteria.")
        return False

    # Inference sanity check
    try:
        pipeline = joblib.load(candidate_path)
        sample_df = pd.DataFrame([{
            "Clean_Ingredients": "water glycerin hyaluronic acid dimethicone niacinamide",
            "Price": 35.0,
            "Rank": 4.5,
            "Combination": 1,
            "Dry": 1,
            "Normal": 1,
            "Oily": 0,
            "Sensitive": 1
        }])
        pred = pipeline.predict(sample_df)
        probs = pipeline.predict_proba(sample_df)
        print(f"  Sanity test passed: Sample predicted as '{pred[0]}' with confidence {probs.max():.2f}")
    except Exception as e:
        print(f"Model promotion REJECTED: Inference sanity check failed ({e})")
        return False

    # Versioning
    registry = load_registry()
    current_ver = registry["current_production_version"] or 0
    new_version = current_ver + 1
    version_filename = f"model_v{new_version}.joblib"
    version_dest = os.path.join(MODELS_DIR, version_filename)
    prod_dest = os.path.join(MODELS_DIR, "production_model.joblib")

    # Copy files
    shutil.copyfile(candidate_path, version_dest)
    shutil.copyfile(candidate_path, prod_dest)

    entry = {
        "version": new_version,
        "artifact_file": version_filename,
        "model_architecture": model_name,
        "promoted_at": datetime.utcnow().isoformat() + "Z",
        "status": "PRODUCTION",
        "metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": metrics.get("weighted_f1", 0.0)
        },
        "promotion_gate": {
            "min_accuracy_threshold": MIN_ACCURACY,
            "min_macro_f1_threshold": MIN_MACRO_F1,
            "passed": True
        }
    }

    # Mark previous production version as ARCHIVED
    for m in registry["models"]:
        if m["status"] == "PRODUCTION":
            m["status"] = "ARCHIVED"

    registry["models"].append(entry)
    registry["current_production_version"] = new_version

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"\nSUCCESS: Candidate successfully promoted to Model Version v{new_version}!")
    print(f"Canonical active model written to: {prod_dest}")
    return True


if __name__ == "__main__":
    success = promote_candidate_model()
    if not success:
        sys.exit(1)
