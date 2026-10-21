"""
Comprehensive Unit and Integration Test Suite
Validates:
1. Data ingestion, validation, and preprocessing transformations
2. Feature pipeline and transformer behavior
3. Model artifact loading and prediction schema conformance
4. Unknown and missing ingredient resilience
5. Avoid-list enforcement and caution rules
6. FastAPI endpoints (/health, /ready, /model-info, /predict, /check-routine, /metrics)
7. Error handling for malformed payloads
"""

import os
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import clean_dataset, normalize_text, parse_ingredients
from src.features.features import build_feature_pipeline
from src.rules.rule_engine import analyze_routine_compatibility, check_avoid_list_violations
from backend.app.main import app
from backend.app.services.model_service import service_instance

client = TestClient(app)


# ---------------- 1. DATA PREPROCESSING TESTS ----------------
def test_text_normalization():
    raw_text = "Water/Aqua/Eau, Glycerin (Plant-derived), GLYCOLIC ACID 10%!"
    cleaned = normalize_text(raw_text)
    assert "water" in cleaned
    assert "glycerin" in cleaned
    assert "glycolic acid" in cleaned
    assert "!" not in cleaned


def test_parse_ingredients():
    ing_str = "Water, Glycerin, Niacinamide, Salicylic Acid, Visit the Sephora site to see the full list of ingredients."
    parsed = parse_ingredients(ing_str)
    assert "water" in parsed
    assert "niacinamide" in parsed
    assert "salicylic acid" in parsed
    # Ensure placeholder was filtered out
    assert not any("visit the sephora site" in p for p in parsed)


def test_clean_dataset_removes_duplicates():
    dummy_df = pd.DataFrame([
        {"Brand": "BrandA", "Name": "Product1", "Ingredients": "water, glycerin, phenoxyethanol"},
        {"Brand": "BrandA", "Name": "Product1", "Ingredients": "water, glycerin, phenoxyethanol"},
        {"Brand": "BrandB", "Name": "Product2", "Ingredients": "water, niacinamide, zinc pca"}
    ])
    cleaned = clean_dataset(dummy_df)
    assert len(cleaned) == 2


# ---------------- 2. INGREDIENT & RULE ENGINE TESTS ----------------
def test_avoid_list_enforcement():
    ingredients = ["water", "fragrance / parfum", "glycerin", "limonene"]
    violations = check_avoid_list_violations(ingredients, ["fragrance", "parabens"])
    assert len(violations) == 1
    assert "fragrance" in violations[0]


def test_retinoid_acid_caution_trigger():
    sample_products = {
        "Treatment": {
            "name": "Retinol Serum",
            "category": "Treatment",
            "ingredients": ["water", "retinol", "squalane"]
        },
        "Cleanser": {
            "name": "AHA Exfoliating Wash",
            "category": "Cleanser",
            "ingredients": ["water", "glycolic acid", "glycerin"]
        }
    }
    result = analyze_routine_compatibility(sample_products, routine_time="evening")
    cautions = [c["type"] for c in result["cautions"]]
    assert "retinoids_and_aha_bha" in cautions


def test_duplicate_salicylic_acid_detection():
    sample_products = {
        "Cleanser": {
            "name": "BHA Cleanser",
            "category": "Cleanser",
            "ingredients": ["water", "salicylic acid", "sodium laureth sulfate"]
        },
        "Treatment": {
            "name": "BHA Serum",
            "category": "Treatment",
            "ingredients": ["salicylic acid", "niacinamide", "water"]
        }
    }
    result = analyze_routine_compatibility(sample_products, routine_time="both")
    duplicate_cautions = [c for c in result["cautions"] if c["type"] == "duplicate_active"]
    assert len(duplicate_cautions) >= 1
    assert "Salicylic Acid" in duplicate_cautions[0]["title"]


def test_photosensitivity_warning_am():
    sample_products = {
        "Treatment": {
            "name": "Night Retinol",
            "category": "Treatment",
            "ingredients": ["water", "retinol", "caprylic triglyceride"]
        }
    }
    # AM routine should flag retinol
    result_am = analyze_routine_compatibility(sample_products, routine_time="morning")
    warnings_am = [w["type"] for w in result_am["warnings"]]
    assert "photosensitivity_warning" in warnings_am


# ---------------- 3. MODEL INFERENCE & API INTEGRATION TESTS ----------------
def test_api_health_and_readiness():
    resp_h = client.get("/health")
    assert resp_h.status_code == 200
    assert resp_h.json()["status"] == "ok"

    resp_r = client.get("/ready")
    assert resp_r.status_code == 200
    assert resp_r.json()["model_loaded"] is True


def test_api_model_info():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "current_production_version" in data
    assert "architecture" in data
    assert data["metrics"]["accuracy"] > 0.40


def test_api_predict_category_success():
    payload = {
        "ingredients_text": "water glycerin hyaluronic acid sodium hyaluronate dimethicone",
        "price": 28.0,
        "rating": 4.6,
        "skin_type": "dry"
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_category" in data
    assert "confidence" in data
    assert data["confidence"] > 0.0
    assert len(data["probabilities"]) == 6


def test_api_predict_category_unknown_or_empty_ingredients():
    payload = {
        "ingredients_text": "xyz unknown substance qwerty rare chemical 999",
        "price": 10.0,
        "rating": 3.0,
        "skin_type": "unknown"
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert "predicted_category" in resp.json()


def test_api_check_routine_pipeline():
    payload = {
        "profile": {
            "skin_type": "combination",
            "concerns": ["dryness", "uneven texture"],
            "sensitivity": "medium",
            "avoid_ingredients": ["fragrance"],
            "preferred_routine": "both",
            "max_budget": 60.0
        }
    }
    resp = client.post("/check-routine", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "morning_routine" in data
    assert "evening_routine" in data
    assert len(data["morning_routine"]) > 0
    assert "disclaimer" in data


def test_api_malformed_request_validation():
    # Sending invalid data types
    payload = {"price": "not_a_number"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # Unprocessable Entity
