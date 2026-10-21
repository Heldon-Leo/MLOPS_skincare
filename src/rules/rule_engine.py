"""
Multi-Product Ingredient Conflict and Compatibility Rule Engine
Evaluates multiple selected products for:
1. Cross-product active ingredient clashes (e.g., Retinoids + AHAs/BHAs, Benzoyl Peroxide + Retinoids).
2. Duplicate active ingredients (e.g., salicylic acid in cleanser AND serum).
3. User-defined ingredient avoid list enforcement.
4. AM photosensitizing active warnings (e.g. retinoids in morning routines).
5. Missing concentration and incomplete data identification.
6. Returns standardized PPT result categories:
   - NO_CONFIGURED_CONFLICT_FOUND
   - POTENTIAL_CONFLICT
   - DUPLICATE_ACTIVE
   - USER_AVOID_LIST_MATCH
   - INSUFFICIENT_INFORMATION
"""

import re
from typing import List, Dict, Any, Optional
from src.rules.rules_data import (
    INGREDIENT_RULES,
    ROUTINE_STEP_ORDER_AM,
    ROUTINE_STEP_ORDER_PM,
    CONCERN_KEY_INGREDIENTS
)
from src.data.preprocess import parse_ingredients_with_metadata, normalize_ingredient_name
from src.models.ingredient_classifier import ingredient_classifier_service

# Recognized potent actives for duplicate checks
MONITORED_ACTIVES = [
    "salicylic acid", "glycolic acid", "lactic acid", "retinol", "tretinoin",
    "adapalene", "benzoyl peroxide", "ascorbic acid", "azelaic acid"
]


def evaluate_multi_product_routine(
    products: List[Dict[str, Any]],
    routine_time: str = "both",
    avoid_list: Optional[List[str]] = None,
    user_concerns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive multi-product analysis across all supplied products.
    """
    if avoid_list is None:
        avoid_list = []
    if user_concerns is None:
        user_concerns = []

    result_statuses = set()
    warnings = []
    cautions = []
    avoid_violations = []
    insufficient_info_notes = []

    # Map to track which product contains which active
    active_tracker: Dict[str, List[Dict[str, Any]]] = {act: [] for act in MONITORED_ACTIVES}

    processed_products = []
    all_flattened_canonical = []

    # 1. Process Each Product & Parse Ingredients
    for prod in products:
        p_name = prod.get("name", "Unnamed Product")
        p_brand = prod.get("brand", "Unknown Brand")
        p_cat = prod.get("category", "General")
        ing_input = prod.get("ingredients")

        parsed_items = []
        if isinstance(ing_input, list):
            # Already list of strings or dicts
            for item in ing_input:
                if isinstance(item, dict):
                    parsed_items.append(item)
                else:
                    norm = normalize_ingredient_name(str(item))
                    parsed_items.append({
                        "raw_token": str(item),
                        "canonical_name": norm,
                        "concentration": None,
                        "has_explicit_concentration": False
                    })
        elif isinstance(ing_input, str):
            parsed_items = parse_ingredients_with_metadata(ing_input)

        if not parsed_items:
            insufficient_info_notes.append({
                "product_name": p_name,
                "issue": "Missing or unparseable ingredient list",
                "recommendation": "Paste full INCI ingredient list manually for verified conflict checking"
            })
            result_statuses.add("INSUFFICIENT_INFORMATION")

        # Classify ingredient functions where possible
        classified_ingredients = []
        canonical_names_for_prod = []

        for item in parsed_items:
            c_name = item["canonical_name"]
            canonical_names_for_prod.append(c_name)
            all_flattened_canonical.append(c_name)

            # Classify function
            cls_res = ingredient_classifier_service.classify_ingredient(c_name)
            classified_ingredients.append({
                "raw_token": item["raw_token"],
                "canonical_name": c_name,
                "concentration": item["concentration"],
                "concentration_display": f"{item['concentration']}%" if item["concentration"] is not None else "Unknown / Not provided",
                "functional_category": cls_res["predicted_function"],
                "classifier_confidence": cls_res["confidence"]
            })

            # Check if this ingredient is a monitored active
            for act in MONITORED_ACTIVES:
                if act in c_name:
                    active_tracker[act].append({
                        "product_name": p_name,
                        "brand": p_brand,
                        "category": p_cat,
                        "concentration": item["concentration"]
                    })

            # Check Avoid List violations
            for avoid_term in avoid_list:
                clean_avoid = normalize_ingredient_name(avoid_term)
                if clean_avoid and clean_avoid in c_name:
                    violation_entry = {
                        "product_name": p_name,
                        "brand": p_brand,
                        "matched_ingredient": item["raw_token"],
                        "avoid_query": avoid_term
                    }
                    avoid_violations.append(violation_entry)
                    result_statuses.add("USER_AVOID_LIST_MATCH")

        processed_products.append({
            "product_id": prod.get("product_id", ""),
            "brand": p_brand,
            "name": p_name,
            "category": p_cat,
            "price": float(prod.get("price", 0.0)),
            "rating": float(prod.get("rating", 0.0)),
            "source": prod.get("source", "User Input / Local Catalog"),
            "ingredients": classified_ingredients,
            "ingredient_count": len(classified_ingredients)
        })

    # 2. Check Duplicate Monitored Actives across multiple products
    duplicate_actives_list = []
    for active_key, appearances in active_tracker.items():
        if len(appearances) > 1:
            result_statuses.add("DUPLICATE_ACTIVE")
            dup_details = {
                "active_name": active_key.title(),
                "products_involved": [f"{app['brand']} {app['product_name']} ({app['category']})" for app in appearances],
                "explanation": (
                    f"Multiple products in your selected routine contain {active_key.title()}. "
                    "Layering duplicate high-potency actives can disrupt the acid mantle and compromise barrier integrity."
                ),
                "severity": "caution"
            }
            duplicate_actives_list.append(dup_details)
            cautions.append({
                "rule_id": "duplicate_active_check",
                "title": f"Duplicate Active: {active_key.title()}",
                "description": dup_details["explanation"],
                "severity": "caution",
                "source": "Dermatological Safety Guidelines (CosIng/CIR)"
            })

    # 3. Check Known Adverse Pairings using Configured Ingredient Rules
    for rule_id, rule in INGREDIENT_RULES.items():
        if rule_id == "duplicate_strong_actives":
            continue

        # Photosensitizing actives in AM routine
        if rule_id == "photosensitizing_actives_in_am":
            if routine_time in ["morning", "both"]:
                matched_am = [ing for ing in all_flattened_canonical if any(t in ing for t in rule["ingredients_a"])]
                if matched_am:
                    result_statuses.add("POTENTIAL_CONFLICT")
                    warnings.append({
                        "rule_id": rule_id,
                        "title": rule["title"],
                        "description": f"{rule['description']} (Detected active: {', '.join(set(matched_am))})",
                        "severity": rule["severity"],
                        "source": "Reviewed Dermatological Literature: UV Photosensitization"
                    })
            continue

        # Cross-ingredient clash detection
        matches_a = [ing for ing in all_flattened_canonical if any(t in ing for t in rule["ingredients_a"])]
        matches_b = [ing for ing in all_flattened_canonical if any(t in ing for t in rule["ingredients_b"])]

        if matches_a and matches_b:
            result_statuses.add("POTENTIAL_CONFLICT")
            entry = {
                "rule_id": rule_id,
                "title": rule["title"],
                "description": f"{rule['description']} (Interacting actives found: {', '.join(set(matches_a))} + {', '.join(set(matches_b))})",
                "severity": rule["severity"],
                "source": "Reviewed Chemical Compatibility Database"
            }
            if rule["severity"] == "warning":
                warnings.append(entry)
            else:
                cautions.append(entry)

    # 4. Check Skin Concern Ingredient Matching
    concern_insights = []
    for concern in user_concerns:
        norm_c = concern.lower()
        target_keys = CONCERN_KEY_INGREDIENTS.get(norm_c, [])
        found_matches = [ing for ing in all_flattened_canonical if any(t in ing for t in target_keys)]
        if found_matches:
            concern_insights.append({
                "concern": concern,
                "status": "MATCHED",
                "matched_ingredients": list(set(found_matches)),
                "message": f"Routine contains recognized beneficial ingredients for '{concern}': {', '.join(set(found_matches))}."
            })
        else:
            concern_insights.append({
                "concern": concern,
                "status": "NO_MATCH",
                "matched_ingredients": [],
                "message": f"No recognized high-potency actives explicitly detected for '{concern}'. Consider discussing target treatments with a dermatologist."
            })

    # 5. Routine Sequencing Recommendations
    sequenced_am = []
    sequenced_pm = []
    category_map = {p["category"]: p for p in processed_products}

    for cat in ROUTINE_STEP_ORDER_AM:
        if cat in category_map:
            sequenced_am.append(category_map[cat])
    for cat in ROUTINE_STEP_ORDER_PM:
        if cat in category_map:
            sequenced_pm.append(category_map[cat])

    # If no conflict found, set status
    if not warnings and not cautions and not avoid_violations:
        result_statuses.add("NO_CONFIGURED_CONFLICT_FOUND")

    # Primary status summary
    priority_order = [
        "POTENTIAL_CONFLICT",
        "DUPLICATE_ACTIVE",
        "USER_AVOID_LIST_MATCH",
        "INSUFFICIENT_INFORMATION",
        "NO_CONFIGURED_CONFLICT_FOUND"
    ]
    primary_status = next((s for s in priority_order if s in result_statuses), "NO_CONFIGURED_CONFLICT_FOUND")

    return {
        "primary_status": primary_status,
        "all_statuses": list(result_statuses),
        "products_analyzed": processed_products,
        "total_products": len(processed_products),
        "duplicate_actives": duplicate_actives_list,
        "warnings": warnings,
        "cautions": cautions,
        "avoid_violations": avoid_violations,
        "insufficient_info": insufficient_info_notes,
        "concern_insights": concern_insights,
        "sequencing_recommendation": {
            "morning": sequenced_am if routine_time in ["morning", "both"] else [],
            "evening": sequenced_pm if routine_time in ["evening", "both"] else []
        },
        "disclaimer": (
            "Educational and formulation analysis only. This software does not provide medical diagnosis, "
            "allergy testing, or clinical prescriptions. Absence of a configured conflict does not guarantee "
            "individual tolerability. Always patch test cosmetic items and consult a board-certified dermatologist."
        )
    }


def analyze_routine_compatibility(
    products_by_category: Dict[str, Dict[str, Any]],
    routine_time: str = "both",
    avoid_list: Optional[List[str]] = None,
    user_concerns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compatibility wrapper converting dict of category->product into list of products."""
    products_list = list(products_by_category.values())
    res = evaluate_multi_product_routine(
        products=products_list,
        routine_time=routine_time,
        avoid_list=avoid_list,
        user_concerns=user_concerns
    )
    return {
        "status": res["primary_status"],
        "warnings": res["warnings"],
        "cautions": res["cautions"],
        "avoid_violations": [f"{v['brand']} {v['product_name']}: contains avoid item '{v['avoid_query']}'" for v in res["avoid_violations"]],
        "beneficial_insights": res["concern_insights"],
        "am_steps_ordered": res["sequencing_recommendation"]["morning"],
        "pm_steps_ordered": res["sequencing_recommendation"]["evening"],
        "disclaimer": res["disclaimer"]
    }

