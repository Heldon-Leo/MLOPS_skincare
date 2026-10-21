"""
FastAPI Request and Response Pydantic Schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SkinProfileRequest(BaseModel):
    """User skin profile input schema."""
    skin_type: str = Field(..., description="User skin type: oily, dry, combination, normal, unknown")
    concerns: List[str] = Field(default_factory=list, description="Skin concerns: dryness, oiliness, uneven texture, appearance of blemishes")
    sensitivity: str = Field(default="low", description="Sensitivity level: low, medium, high")
    avoid_ingredients: List[str] = Field(default_factory=list, description="List of ingredients to avoid")
    preferred_routine: str = Field(default="both", description="morning, evening, or both")
    max_budget: Optional[float] = Field(default=None, description="Max budget per item in USD")


class PredictCategoryRequest(BaseModel):
    """Single product category inference request schema."""
    ingredients_text: str = Field(..., description="Full or partial ingredient list")
    price: float = Field(default=35.0, description="Estimated price in USD")
    rating: float = Field(default=4.2, description="Average customer rating")
    skin_type: str = Field(default="normal", description="Target skin type")


class PredictCategoryResponse(BaseModel):
    """Single product category inference response schema."""
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str


class RoutineCheckRequest(BaseModel):
    """Full routine check and recommendation request schema."""
    profile: SkinProfileRequest
    custom_products: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Optional custom products provided by user")


class RoutineCheckResponse(BaseModel):
    """Complete routine assessment response schema."""
    profile_summary: Dict[str, Any]
    morning_routine: List[Dict[str, Any]]
    evening_routine: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    cautions: List[Dict[str, Any]]
    avoid_violations: List[str]
    beneficial_insights: List[Dict[str, Any]]
    model_version_used: str
    disclaimer: str


class ModelInfoResponse(BaseModel):
    """Model version and registry metadata response."""
    current_production_version: int
    architecture: str
    promoted_at: str
    metrics: Dict[str, float]
    status: str
