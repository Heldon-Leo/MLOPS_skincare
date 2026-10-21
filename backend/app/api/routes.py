"""
Extended FastAPI Routes
Includes:
- Product Search & ID lookup
- Individual Ingredient Function Classifier
- Structured Ingredient Parser (with concentration_source)
- Label Image OCR Extractor
- Multi-Product Ingredient Conflict Check
- Product Category Classifier (retained)
- Health, Readiness, Model-Info, Drift, and Telemetry
"""

import time
import json
import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from backend.app.schemas.schemas import (
    ProductSearchResponse,
    ProductSearchResult,
    ParseIngredientsRequest,
    ParseIngredientsResponse,
    ClassifyIngredientRequest,
    ClassifyIngredientResponse,
    MultiProductRoutineCheckRequest,
    RoutineCheckResponse,
    PredictCategoryRequest,
    PredictCategoryResponse,
    ModelInfoResponse
)
from backend.app.services.model_service import service_instance
from backend.app.services.ocr_service import label_ocr_service
from backend.app.core.metrics import metrics_collector
from src.data.product_lookup import product_lookup_service
from src.models.ingredient_classifier import ingredient_classifier_service
from src.data.preprocess import parse_ingredients_with_metadata
from src.rules.rule_engine import evaluate_multi_product_routine
from src.evaluation.drift_monitor import drift_monitor

router = APIRouter()


# ---------------- 1. PRODUCT LOOKUP ENDPOINTS ----------------
@router.get("/products/search", response_model=ProductSearchResponse, tags=["Product Lookup"])
async def search_products(
    q: str = Query(..., description="Product name or keywords to search for"),
    brand: Optional[str] = Query(None, description="Optional brand filter"),
    limit: int = Query(10, ge=1, le=50)
):
    """Searches local cosmetics dataset using exact, partial, and fuzzy Levenshtein matching."""
    t0 = time.time()
    try:
        results = product_lookup_service.search_products(query=q, brand_filter=brand, limit=limit)
        latency = time.time() - t0
        metrics_collector.record_request("/products/search", latency, 200)
        drift_monitor.log_endpoint_latency("/products/search", latency)
        return {
            "query": q,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        latency = time.time() - t0
        metrics_collector.record_request("/products/search", latency, 500)
        drift_monitor.log_endpoint_latency("/products/search", latency)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/products/{product_id}", response_model=ProductSearchResult, tags=["Product Lookup"])
async def get_product_by_id(product_id: str):
    """Retrieves full product record and ingredient list by product ID."""
    prod = product_lookup_service.get_product_by_id(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail=f"Product with ID '{product_id}' not found.")
    return prod


# ---------------- 2. INGREDIENT NLP & PARSER ENDPOINTS ----------------
@router.post("/ingredients/parse", response_model=ParseIngredientsResponse, tags=["Ingredients NLP"])
async def parse_ingredients(payload: ParseIngredientsRequest):
    """Parses raw text into structured canonical INCI tokens with explicit percentages and concentration_source."""
    t0 = time.time()
    items = parse_ingredients_with_metadata(payload.ingredients_text)
    enhanced_items = []
    canonical_tokens = []

    for itm in items:
        cls_res = ingredient_classifier_service.classify_ingredient(itm["canonical_name"])
        drift_monitor.log_classification_result(cls_res["predicted_function"])
        canonical_tokens.append(itm["canonical_name"])
        enhanced_items.append({
            "raw_token": itm["raw_token"],
            "canonical_name": itm["canonical_name"],
            "concentration": itm["concentration"],
            "has_explicit_concentration": itm["has_explicit_concentration"],
            "concentration_source": itm.get("concentration_source", "Not provided"),
            "functional_category": cls_res["predicted_function"]
        })

    # Log ingredient tokens for drift monitoring
    drift_monitor.log_observed_ingredients(canonical_tokens)
    latency = time.time() - t0
    metrics_collector.record_request("/ingredients/parse", latency, 200)
    drift_monitor.log_endpoint_latency("/ingredients/parse", latency)

    return {
        "raw_text": payload.ingredients_text,
        "ingredient_count": len(enhanced_items),
        "parsed_ingredients": enhanced_items
    }


@router.post("/ingredients/classify", response_model=ClassifyIngredientResponse, tags=["Ingredients NLP"])
async def classify_ingredient(payload: ClassifyIngredientRequest):
    """Predicts functional category for individual chemical ingredient with confidence-based abstention."""
    t0 = time.time()
    try:
        res = ingredient_classifier_service.classify_ingredient(payload.ingredient_name)
        drift_monitor.log_classification_result(res["predicted_function"])
        drift_monitor.log_observed_ingredients([payload.ingredient_name])
        latency = time.time() - t0
        metrics_collector.record_request("/ingredients/classify", latency, 200)
        drift_monitor.log_endpoint_latency("/ingredients/classify", latency)
        return res
    except Exception as e:
        latency = time.time() - t0
        metrics_collector.record_request("/ingredients/classify", latency, 500)
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


# ---------------- 3. OCR IMAGE EXTRACTION ENDPOINT ----------------
@router.post("/ocr/extract", tags=["Multi-Modal OCR"])
async def extract_label_ocr(file: UploadFile = File(...)):
    """Accepts uploaded packaging photograph and runs OCR to extract ingredient text."""
    t0 = time.time()
    try:
        content = await file.read()
        res = label_ocr_service.extract_text_from_bytes(content, filename=file.filename or "upload.jpg")
        latency = time.time() - t0
        metrics_collector.record_request("/ocr/extract", latency, 200)
        drift_monitor.log_endpoint_latency("/ocr/extract", latency)
        return res
    except Exception as e:
        latency = time.time() - t0
        metrics_collector.record_request("/ocr/extract", latency, 500)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


# ---------------- 4. MULTI-PRODUCT ROUTINE CHECK ENDPOINT ----------------
@router.post("/check-routine", response_model=RoutineCheckResponse, tags=["Routine Analysis"])
async def check_multi_product_routine(payload: MultiProductRoutineCheckRequest):
    """
    Evaluates multi-product routine for duplicate actives, formulation clashes,
    avoid violations, and sequencing order.
    """
    t0 = time.time()
    try:
        # Convert Pydantic items to dicts
        prod_dicts = []
        for p in payload.products:
            # If product has product_id and missing ingredients, load from catalog
            if p.product_id and (not p.ingredients or p.ingredients == ""):
                cat_prod = product_lookup_service.get_product_by_id(p.product_id)
                if cat_prod and cat_prod.get("ingredients_raw"):
                    p.ingredients = cat_prod["ingredients_raw"]
                    p.brand = cat_prod["brand"]
                    p.name = cat_prod["name"]
                    p.category = cat_prod["category"]
                    p.price = cat_prod["price"]
                    p.rating = cat_prod["rating"]
            prod_dicts.append(p.model_dump())

        eval_res = evaluate_multi_product_routine(
            products=prod_dicts,
            routine_time=payload.routine_time,
            avoid_list=payload.avoid_ingredients,
            user_concerns=payload.user_concerns
        )

        # Log all ingredient tokens across all products for drift monitoring
        all_tokens = []
        for prod_result in eval_res.get("products_analyzed", []):
            for ing in prod_result.get("ingredients", []):
                token = ing.get("canonical_name", "")
                if token:
                    all_tokens.append(token)
                    drift_monitor.log_classification_result(ing.get("functional_category", "Unknown"))
        if all_tokens:
            drift_monitor.log_observed_ingredients(all_tokens)

        latency = time.time() - t0
        metrics_collector.record_request("/check-routine", latency, 200)
        drift_monitor.log_endpoint_latency("/check-routine", latency)
        return eval_res
    except Exception as e:
        latency = time.time() - t0
        metrics_collector.record_request("/check-routine", latency, 500)
        raise HTTPException(status_code=500, detail=f"Routine evaluation failed: {str(e)}")


# ---------------- 5. LEGACY PREDICT ENDPOINT ----------------
@router.post("/predict", response_model=PredictCategoryResponse, tags=["Product Classifier"])
async def predict_product_category(payload: PredictCategoryRequest):
    """Predicts routine category for complete product (retained secondary model)."""
    t0 = time.time()
    try:
        res = service_instance.predict_category(payload)
        metrics_collector.record_prediction(res["predicted_category"])
        latency = time.time() - t0
        metrics_collector.record_request("/predict", latency, 200)
        drift_monitor.log_endpoint_latency("/predict", latency)
        return res
    except Exception as e:
        latency = time.time() - t0
        metrics_collector.record_request("/predict", latency, 500)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ---------------- 6. SYSTEM & TELEMETRY ENDPOINTS ----------------
@router.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "skincare-routine-ingredient-checker-api"}


@router.get("/ready", tags=["System"])
async def readiness_check():
    if not service_instance.is_ready:
        raise HTTPException(status_code=503, detail="Model artifact not ready")
    return {"status": "ready", "models_loaded": True}


@router.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def get_model_info():
    info = service_instance.get_model_info()
    # Attach ingredient classifier metadata
    rep_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "reports", "ingredient_classifier_metrics.json"
    )
    if os.path.exists(rep_path):
        with open(rep_path, "r", encoding="utf-8") as f:
            info["ingredient_classifier"] = json.load(f)
    return info


@router.get("/metrics", tags=["Operational Telemetry"])
async def get_operational_metrics():
    """Returns operational telemetry combined with current drift summary."""
    ops = metrics_collector.get_summary()
    drift = drift_monitor.get_drift_report()
    return {
        "operational": ops,
        "drift": drift
    }


@router.get("/drift", tags=["Operational Telemetry"])
async def get_drift_metrics():
    """Returns comprehensive data and prediction drift report."""
    return drift_monitor.get_drift_report()
