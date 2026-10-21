"""
Model Inference and Routine Recommendation Service
Loads serialized production model pipeline at startup, executes inferences,
and coordinates recommendation logic.
"""

import os
import sys
import json
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional

# Ensure project root is available
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rules.rule_engine import analyze_routine_compatibility
from backend.app.schemas.schemas import SkinProfileRequest, PredictCategoryRequest

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROD_MODEL_PATH = os.path.join(MODELS_DIR, "production_model.joblib")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "cleaned_cosmetics.csv")


class SkincareModelService:
    """Manages loaded model pipeline, catalog lookup, and prediction inference."""

    def __init__(self):
        self.model_pipeline = None
        self.registry_data = {}
        self.product_catalog = None
        self.is_ready = False
        self.load_artifacts()

    def load_artifacts(self):
        """Loads model pipeline, registry, and product catalog into memory."""
        try:
            if os.path.exists(PROD_MODEL_PATH):
                self.model_pipeline = joblib.load(PROD_MODEL_PATH)
                self.is_ready = True
                print(f"[Service] Loaded production model from {PROD_MODEL_PATH}")
            else:
                print(f"[Service Warning] Production model not found at {PROD_MODEL_PATH}")

            if os.path.exists(REGISTRY_PATH):
                with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                    self.registry_data = json.load(f)

            if os.path.exists(CLEANED_DATA_PATH):
                self.product_catalog = pd.read_csv(CLEANED_DATA_PATH)
                print(f"[Service] Loaded product catalog: {len(self.product_catalog)} records")
        except Exception as e:
            print(f"[Service Error] Failed loading artifacts: {e}")
            self.is_ready = False

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active production model."""
        if not self.registry_data or "models" not in self.registry_data:
            return {
                "current_production_version": 1,
                "architecture": "RandomForest_Classifier",
                "promoted_at": "Unknown",
                "metrics": {},
                "status": "LOADED" if self.is_ready else "NOT_LOADED"
            }
        curr_ver = self.registry_data.get("current_production_version", 1)
        for m in self.registry_data.get("models", []):
            if m.get("version") == curr_ver:
                return {
                    "current_production_version": curr_ver,
                    "architecture": m.get("model_architecture", "Unknown"),
                    "promoted_at": m.get("promoted_at", ""),
                    "metrics": m.get("metrics", {}),
                    "status": "PRODUCTION"
                }
        return {"current_production_version": curr_ver, "architecture": "Standard", "status": "ACTIVE"}

    def predict_category(self, req: PredictCategoryRequest) -> Dict[str, Any]:
        """Predicts routine category for given ingredient list and metadata."""
        if not self.is_ready or self.model_pipeline is None:
            raise RuntimeError("Model pipeline not loaded or service not ready")

        # Prepare single row dataframe matching feature transformer schema
        skin = req.skin_type.lower()
        row = {
            "Clean_Ingredients": req.ingredients_text.lower(),
            "Price": float(req.price),
            "Rank": float(req.rating),
            "Combination": 1 if skin in ["combination", "unknown"] else 0,
            "Dry": 1 if skin in ["dry", "unknown"] else 0,
            "Normal": 1 if skin in ["normal", "unknown"] else 0,
            "Oily": 1 if skin in ["oily", "unknown"] else 0,
            "Sensitive": 1 if skin in ["sensitive"] else 0
        }
        df = pd.DataFrame([row])
        pred_label = self.model_pipeline.predict(df)[0]
        probs = self.model_pipeline.predict_proba(df)[0]
        classes = self.model_pipeline.classes_

        prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
        confidence = float(max(probs))

        return {
            "predicted_category": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "model_version": f"v{self.registry_data.get('current_production_version', 1)}"
        }

    def assemble_routine_for_profile(self, profile: SkinProfileRequest, custom_products: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Selects optimal items from real product catalog matching skin type, budget,
        and user preferences, then runs comprehensive rule analysis.
        """
        skin_col = profile.skin_type.capitalize()
        budget = profile.max_budget

        selected_products: Dict[str, Dict[str, Any]] = {}
        target_categories = ["Cleanser", "Moisturizer", "Sun protect", "Treatment", "Face Mask", "Eye cream"]

        if self.product_catalog is not None:
            catalog = self.product_catalog.copy()
            # Filter by budget if provided
            if budget and budget > 0:
                budget_filtered = catalog[catalog["Price"] <= budget]
                if not budget_filtered.empty:
                    catalog = budget_filtered

            # Filter or sort by skin-type match if column exists
            for cat in target_categories:
                cat_df = catalog[catalog["Label"] == cat]
                if cat_df.empty:
                    continue

                if skin_col in cat_df.columns:
                    matched = cat_df[cat_df[skin_col] == 1]
                    if not matched.empty:
                        best_item = matched.sort_values(by="Rank", ascending=False).iloc[0]
                    else:
                        best_item = cat_df.sort_values(by="Rank", ascending=False).iloc[0]
                else:
                    best_item = cat_df.sort_values(by="Rank", ascending=False).iloc[0]

                selected_products[cat] = {
                    "category": cat,
                    "brand": best_item["Brand"],
                    "name": best_item["Name"],
                    "price": float(best_item["Price"]),
                    "rank": float(best_item["Rank"]),
                    "ingredients": [i.strip() for i in str(best_item["Ingredients"]).split(",")[:15]]
                }

        # Override or augment with any user custom products
        if custom_products:
            for cp in custom_products:
                cat = cp.get("category", "Treatment")
                selected_products[cat] = {
                    "category": cat,
                    "brand": cp.get("brand", "Custom"),
                    "name": cp.get("name", "User Provided Item"),
                    "price": float(cp.get("price", 25.0)),
                    "rank": 4.5,
                    "ingredients": cp.get("ingredients", [])
                }

        # Run Rule Engine Analysis
        rule_eval = analyze_routine_compatibility(
            products_by_category=selected_products,
            routine_time=profile.preferred_routine,
            avoid_list=profile.avoid_ingredients,
            user_concerns=profile.concerns
        )

        return {
            "profile_summary": {
                "skin_type": profile.skin_type,
                "concerns": profile.concerns,
                "sensitivity": profile.sensitivity,
                "preferred_routine": profile.preferred_routine,
                "budget_cap": profile.max_budget
            },
            "morning_routine": rule_eval["am_steps_ordered"] if profile.preferred_routine in ["morning", "both"] else [],
            "evening_routine": rule_eval["pm_steps_ordered"] if profile.preferred_routine in ["evening", "both"] else [],
            "warnings": rule_eval["warnings"],
            "cautions": rule_eval["cautions"],
            "avoid_violations": rule_eval["avoid_violations"],
            "beneficial_insights": rule_eval["beneficial_insights"],
            "model_version_used": f"v{self.registry_data.get('current_production_version', 1)}",
            "disclaimer": rule_eval["disclaimer"]
        }


# Singleton instance
service_instance = SkincareModelService()
