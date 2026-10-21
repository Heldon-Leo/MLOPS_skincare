"""
Automated Retraining and Continuous Evaluation Script
Enforces:
1. Schema & provenance verification of new dataset
2. Retraining of pipeline on augmented or updated data
3. Evaluation against held-out validation set
4. Candidate vs currently approved production model comparison
5. Conditional promotion only if candidate exceeds or matches current quality thresholds
6. Rollback preservation
"""

import os
import sys
import json
import shutil
import joblib
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.validate_data import validate_raw_dataset
from scripts.train import train_and_compare_models
from scripts.promote_model import promote_candidate_model, load_registry, REGISTRY_PATH

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def rollback_to_previous_model() -> bool:
    """Rolls back production model to previous approved version if available."""
    registry = load_registry()
    prod_ver = registry.get("current_production_version", 1)

    archived = [m for m in registry.get("models", []) if m["version"] < prod_ver]
    if not archived:
        print("Rollback aborted: No prior archived model version available in registry.")
        return False

    prev_model = max(archived, key=lambda x: x["version"])
    target_ver = prev_model["version"]
    prev_file = os.path.join(MODELS_DIR, f"model_v{target_ver}.joblib")
    prod_file = os.path.join(MODELS_DIR, "production_model.joblib")

    if not os.path.exists(prev_file):
        print(f"Rollback error: Artifact {prev_file} missing.")
        return False

    shutil.copyfile(prev_file, prod_file)

    # Update statuses
    for m in registry["models"]:
        if m["version"] == prod_ver:
            m["status"] = "ROLLED_BACK"
        elif m["version"] == target_ver:
            m["status"] = "PRODUCTION"

    registry["current_production_version"] = target_ver
    registry["rollback_history"].append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "from_version": prod_ver,
        "to_version": target_ver,
        "reason": "Operator initiated rollback"
    })

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"SUCCESS: Rolled back production model from v{prod_ver} to v{target_ver}.")
    return True


def run_retraining_workflow(new_dataset_path: str = None) -> bool:
    """Runs full automated retraining, validation, and conditional promotion."""
    if new_dataset_path is None:
        new_dataset_path = os.path.join(PROJECT_ROOT, "data", "raw", "cosmetics.csv")

    print(f"\n=== Step 1: Validating Dataset Schema ({new_dataset_path}) ===")
    valid, rep = validate_raw_dataset(new_dataset_path)
    if not valid:
        print(f"Retraining aborted: Dataset validation failed: {rep}")
        return False

    print("=== Step 2: Training and Model Comparison ===")
    best_name, metrics = train_and_compare_models()

    print("=== Step 3: Gating and Model Promotion ===")
    promoted = promote_candidate_model()
    if promoted:
        print("Retraining workflow completed: Candidate promoted to active production.")
    else:
        print("Retraining workflow completed: Candidate model did not satisfy promotion criteria.")
    return promoted


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        rollback_to_previous_model()
    else:
        run_retraining_workflow()
