"""
FastAPI Router Endpoints
Implements /predict, /check-routine, /model-info, and operational endpoints.
"""

import time
from fastapi import APIRouter, HTTPException, Request, Response
from typing import Dict, Any

from backend.app.schemas.schemas import (
    SkinProfileRequest,
    PredictCategoryRequest,
    PredictCategoryResponse,
    RoutineCheckRequest,
    RoutineCheckResponse,
    ModelInfoResponse
)
from backend.app.services.model_service import service_instance
from backend.app.core.metrics import metrics_collector

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    """Confirms web service is alive."""
    return {"status": "ok", "service": "skincare-routine-checker-api"}


@router.get("/ready", tags=["System"])
async def readiness_check():
    """Confirms model pipeline is loaded and ready to serve inferences."""
    if not service_instance.is_ready:
        raise HTTPException(status_code=503, detail="Model artifact not ready or failed to load")
    return {"status": "ready", "model_loaded": True}


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Returns approved model version, architecture, and gate metrics."""
    return service_instance.get_model_info()


@router.post("/predict", response_model=PredictCategoryResponse, tags=["Inference"])
async def predict_category(payload: PredictCategoryRequest):
    """Predicts routine category for given product ingredients using real trained model."""
    t0 = time.time()
    try:
        res = service_instance.predict_category(payload)
        metrics_collector.record_prediction(res["predicted_category"])
        metrics_collector.record_request("/predict", time.time() - t0, 200)
        return res
    except Exception as e:
        metrics_collector.record_request("/predict", time.time() - t0, 500)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/check-routine", response_model=RoutineCheckResponse, tags=["Recommendation"])
async def check_routine(payload: RoutineCheckRequest):
    """Generates complete educational skincare routine with evidence-based compatibility checks."""
    t0 = time.time()
    try:
        res = service_instance.assemble_routine_for_profile(
            profile=payload.profile,
            custom_products=payload.custom_products
        )
        metrics_collector.record_request("/check-routine", time.time() - t0, 200)
        return res
    except Exception as e:
        metrics_collector.record_request("/check-routine", time.time() - t0, 500)
        raise HTTPException(status_code=500, detail=f"Routine check error: {str(e)}")


@router.get("/metrics", tags=["Operational Telemetry"])
async def get_operational_metrics():
    """Exposes aggregate operational telemetry and request distribution without PII."""
    return metrics_collector.get_summary()
