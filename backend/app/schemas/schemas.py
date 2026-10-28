"""
FastAPI Pydantic Schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------- 1. PRODUCT SEARCH SCHEMAS ----------------
class ProductSearchResult(BaseModel):
    product_id: str
    brand: str
    name: str
    category: str
    price: float
    rating: float
    has_ingredients: bool
    source: str
    match_type: str
    match_score: float
    ingredients_raw: Optional[str] = None
    skin_suitability: Dict[str, int]


class ProductSearchResponse(BaseModel):
    query: str
    count: int
    results: List[ProductSearchResult]


# ---------------- 2. INGREDIENT PARSING & CLASSIFICATION SCHEMAS ----------------
class ParseIngredientsRequest(BaseModel):
    ingredients_text: str = Field(..., description="Raw text of ingredients list")


class ParsedIngredientItem(BaseModel):
    raw_token: str
    canonical_name: str
    concentration: Optional[float] = None
    has_explicit_concentration: bool
    concentration_source: str = "Not provided"
    functional_category: Optional[str] = None


class ParseIngredientsResponse(BaseModel):
    raw_text: str
    ingredient_count: int
    parsed_ingredients: List[ParsedIngredientItem]


class ClassifyIngredientRequest(BaseModel):
    ingredient_name: str = Field(..., description="Individual ingredient name to classify")


class ClassifyIngredientResponse(BaseModel):
    ingredient: str
    predicted_function: str
    confidence: float
    probabilities: Dict[str, float]


# ---------------- 3. MULTI-PRODUCT ROUTINE CHECK SCHEMAS ----------------
class ProductInputItem(BaseModel):
    product_id: Optional[str] = None
    name: str
    brand: Optional[str] = "Custom"
    category: Optional[str] = "Treatment"
    price: Optional[float] = 0.0
    rating: Optional[float] = 0.0
    ingredients: Optional[Any] = None  # Can be raw string or list of tokens
    source: Optional[str] = "User Input"


class MultiProductRoutineCheckRequest(BaseModel):
    products: List[ProductInputItem] = Field(..., min_length=1, description="List of products in routine")
    routine_time: str = Field(default="both", description="morning, evening, or both")
    avoid_ingredients: List[str] = Field(default_factory=list, description="Ingredients to avoid")
    user_concerns: List[str] = Field(default_factory=list, description="Primary skin concerns")


class DuplicateActiveItem(BaseModel):
    active_name: str
    products_involved: List[str]
    explanation: str
    severity: str


class RuleWarningItem(BaseModel):
    rule_id: str
    title: str
    description: str
    severity: str
    source: str


class ActiveSynergyItem(BaseModel):
    synergy_id: str
    title: str
    paired_ingredients: List[str]
    benefit: str
    source: str


class BarrierLoadDetail(BaseModel):
    score: float
    level: str
    level_label: str
    color: str
    summary: str
    buffering_ingredients_detected: int
    buffering_relief_points: float
    clash_penalty_applied: float
    duplicate_penalty_applied: float
    contributing_actives: List[Dict[str, Any]]


class RoutineCheckResponse(BaseModel):
    primary_status: str
    all_statuses: List[str]
    total_products: int
    products_analyzed: List[Dict[str, Any]]
    duplicate_actives: List[DuplicateActiveItem]
    warnings: List[RuleWarningItem]
    cautions: List[RuleWarningItem]
    avoid_violations: List[Dict[str, Any]]
    insufficient_info: List[Dict[str, Any]]
    concern_insights: List[Dict[str, Any]]
    active_synergies: Optional[List[ActiveSynergyItem]] = []
    barrier_load: Optional[BarrierLoadDetail] = None
    sequencing_recommendation: Dict[str, List[Dict[str, Any]]]
    disclaimer: str


# ---------------- 4. LEGACY / EXISTING COMPATIBILITY SCHEMAS ----------------
class SkinProfileRequest(BaseModel):
    skin_type: str = Field(..., description="User skin type: oily, dry, combination, normal, unknown")
    concerns: List[str] = Field(default_factory=list)
    sensitivity: str = Field(default="low")
    avoid_ingredients: List[str] = Field(default_factory=list)
    preferred_routine: str = Field(default="both")
    max_budget: Optional[float] = None


class PredictCategoryRequest(BaseModel):
    ingredients_text: str
    price: float = 35.0
    rating: float = 4.2
    skin_type: str = "normal"


class PredictCategoryResponse(BaseModel):
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str


class ModelInfoResponse(BaseModel):
    current_production_version: int
    architecture: str
    promoted_at: str
    metrics: Dict[str, float]
    status: str
    ingredient_classifier: Optional[Dict[str, Any]] = None
