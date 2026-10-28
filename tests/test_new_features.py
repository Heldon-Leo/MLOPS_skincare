"""
Additional Tests for Updated Features
=====================================
Covers:
1. Product lookup brand filter & fuzzy threshold enforcement.
2. Ingredient parsing includes `concentration_source` field.
3. `/drift` endpoint returns expected structure and fields.
4. Active synergies and Skin Barrier Load scoring.
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# -------------------------------------------------------------------
# 1. Product Lookup – Brand Filter & Fuzzy Threshold
# -------------------------------------------------------------------
def test_product_lookup_brand_filter():
    resp = client.get("/products/search", params={"q": "cream", "brand": "LA MER", "limit": 5})
    assert resp.status_code == 200, "Search endpoint should respond 200"
    data = resp.json()
    for r in data.get("results", []):
        assert r["brand"].lower() == "la mer".lower()


def test_product_lookup_fuzzy_threshold():
    resp = client.get("/products/search", params={"q": "Cleanser", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    for r in data.get("results", []):
        assert r["match_score"] >= 50 or r["match_type"] in ("EXACT", "PARTIAL")


# -------------------------------------------------------------------
# 2. Ingredient Parsing – concentration_source field
# -------------------------------------------------------------------
def test_parse_ingredients_concentration_source():
    payload = {
        "ingredients_text": "Aqua (Water), Niacinamide 10%, Glycerin, Squalane"
    }
    resp = client.post("/ingredients/parse", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "parsed_ingredients" in data
    niacinamide_item = next(
        (i for i in data["parsed_ingredients"] if "niacinamide" in i["canonical_name"].lower()),
        None
    )
    assert niacinamide_item is not None
    assert "concentration_source" in niacinamide_item
    assert niacinamide_item["concentration_source"] != "Not provided"


# -------------------------------------------------------------------
# 3. Drift Endpoint
# -------------------------------------------------------------------
def test_drift_endpoint_structure():
    resp = client.get("/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "vocabulary_drift" in data
    assert "prediction_drift" in data
    assert "drift_severity" in data


# -------------------------------------------------------------------
# 4. Active Synergy & Skin Barrier Load Novelty Tests
# -------------------------------------------------------------------
def test_synergy_and_barrier_load():
    payload = {
        "products": [
            {
                "name": "Vitamin C Booster",
                "brand": "Skincare Pro",
                "category": "Treatment",
                "ingredients": "Water, Ascorbic Acid 15%, Tocopherol, Ferulic Acid"
            },
            {
                "name": "Barrier Recovery Cream",
                "brand": "Ceramide Lab",
                "category": "Moisturizer",
                "ingredients": "Water, Glycerin, Ceramide 3, Hyaluronic Acid, Squalane"
            }
        ],
        "routine_time": "both",
        "avoid_ingredients": [],
        "user_concerns": ["dryness"]
    }
    resp = client.post("/check-routine", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Verify barrier load calculation
    assert "barrier_load" in data
    b_load = data["barrier_load"]
    assert 0 <= b_load["score"] <= 100
    assert b_load["level"] in ["LOW", "MODERATE", "HIGH"]
    assert b_load["buffering_ingredients_detected"] >= 2

    # Verify clinical synergy detection (Vitamin C + E + Ferulic Acid)
    assert "active_synergies" in data
    syn_titles = [s["title"] for s in data["active_synergies"]]
    assert any("Gold Standard Photoprotective Network" in t for t in syn_titles)
