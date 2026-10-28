"""
Dataset Validation Module
Validates schemas, data types, null boundaries, and expected distributions.
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, Any, Tuple

REQUIRED_COLUMNS = [
    "Label", "Brand", "Name", "Price", "Rank", "Ingredients",
    "Combination", "Dry", "Normal", "Oily", "Sensitive"
]

VALID_LABELS = {
    "Moisturizer", "Cleanser", "Face Mask", "Treatment", "Eye cream", "Sun protect"
}


def validate_raw_dataset(filepath: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates a cosmetics dataset CSV.
    Returns (is_valid, validation_report)
    """
    if not os.path.exists(filepath):
        return False, {"error": f"File does not exist: {filepath}"}

    df = pd.read_csv(filepath)
    report = {
        "record_count": len(df),
        "columns_found": list(df.columns),
        "missing_columns": [c for c in REQUIRED_COLUMNS if c not in df.columns],
        "null_counts": df.isnull().sum().to_dict(),
        "invalid_label_count": 0,
        "duplicate_rows": int(df.duplicated(subset=["Brand", "Name"]).sum()),
        "status": "PASS"
    }

    if report["missing_columns"]:
        report["status"] = "FAIL"
        report["error"] = f"Missing required columns: {report['missing_columns']}"
        return False, report

    # Check labels
    unique_labels = set(df["Label"].unique())
    unknown_labels = unique_labels - VALID_LABELS
    if unknown_labels:
        report["invalid_label_count"] = len(unknown_labels)
        report["unknown_labels"] = list(unknown_labels)
        report["status"] = "WARN"

    # Check numeric ranges
    if (df["Price"] < 0).any():
        report["status"] = "FAIL"
        report["error"] = "Negative price detected"
        return False, report

    if (df["Rank"] < 0).any() or (df["Rank"] > 5.0).any():
        report["status"] = "FAIL"
        report["error"] = "Rating rank out of [0, 5] bounds"
        return False, report

    # Check skin suitability binary flags
    flag_cols = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
    for col in flag_cols:
        invalid_flags = df[~df[col].isin([0, 1])]
        if not invalid_flags.empty:
            report["status"] = "FAIL"
            report["error"] = f"Column {col} has non-binary values"
            return False, report

    return True, report


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Validate cosmetics dataset")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to CSV dataset to validate")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = args.file
    if not target_path:
        # Default to raw/cosmetics.csv if exists, else fallback to interim/cleaned_cosmetics.csv
        raw_path = os.path.join(project_root, "data", "raw", "cosmetics.csv")
        interim_path = os.path.join(project_root, "data", "interim", "cleaned_cosmetics.csv")
        target_path = raw_path if os.path.exists(raw_path) else interim_path

    valid, rep = validate_raw_dataset(target_path)
    print(f"Validating: {target_path}")
    print("Validation Result:", "SUCCESS" if valid else "FAILED")
    print(json.dumps(rep, indent=2))
    if not valid:
        sys.exit(1)
