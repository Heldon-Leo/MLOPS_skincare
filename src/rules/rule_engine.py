"""
Multi-Product Ingredient Conflict, Synergy, and Skin Barrier Load Rule Engine
Evaluates multiple selected products for:
1. Cross-product active ingredient clashes (Retinoids + AHAs/BHAs, Benzoyl Peroxide + Retinoids).
2. Duplicate active ingredients (e.g. multiple leave-on salicylic acids).
3. Clinically validated Active Synergies (e.g., Vitamin C + E + Ferulic, Retinol + Ceramides).
4. Skin Barrier Load Score (0-100) taking into account leave-on vs rinse-off vehicle and active concentration.
5. User-defined ingredient avoid list enforcement.
6. AM photosensitizing active warnings.
7. Standardized result statuses and structured explanations.
"""

import re
from typing import List, Dict, Any, Optional
from src.rules.rules_data import (
    INGREDIENT_RULES,
    ACTIVE_SYNERGIES,
    ROUTINE_STEP_ORDER_AM,
    ROUTINE_STEP_ORDER_PM,
    CONCERN_KEY_INGREDIENTS
)
from src.data.preprocess import parse_ingredients_with_metadata, normalize_ingredient_name
from src.models.ingredient_classifier import ingredient_classifier_service

# Recognized potent actives for duplicate checks and irritation scoring
MONITORED_ACTIVES = [
    "salicylic acid", "glycolic acid", "lactic acid", "retinol", "tretinoin",
    "adapalene", "benzoyl peroxide", "ascorbic acid", "azelaic acid"
]

# Base irritation weight per monitored active (on leave-on formulations)
ACTIVE_IRRITATION_WEIGHTS = {
    "tretinoin": 35.0,
    "adapalene": 30.0,
    "benzoyl peroxide": 25.0,
    "retinol": 22.0,
    "glycolic acid": 20.0,
    "salicylic acid": 18.0,
    "lactic acid": 14.0,
    "ascorbic acid": 15.0,
    "azelaic acid": 12.0
}

# Product category vehicle multipliers
# Wash-off products (Cleansers) expose skin to actives for ~60s, yielding far lower irritation load
CATEGORY_VEHICLE_MULTIPLIERS = {
    "Cleanser": 0.25,
    "Face Mask": 0.50,
    "Treatment": 1.0,
    "Moisturizer": 0.85,
    "Eye cream": 0.90,
    "Sun protect": 0.70
}


def compute_barrier_load(
    products: List[Dict[str, Any]],
    clash_count: int,
    duplicate_count: int
) -> Dict[str, Any]:
    """
    Computes a scientific, formulation-aware Skin Barrier Load Score (0-100).
    Factors in:
    - Active chemical potency
    - Vehicle type (leave-on serum vs rinse-off cleanser)
    - Position/concentration of active in formula
    - Buffering ingredients present (ceramides, squalane, panthenol, hyaluronic acid)
    - Penalty for formulation clashes and duplications
    """
    raw_score = 0.0
    active_load_items = []
    buffering_count = 0

    buffering_ingredients = ["ceramide", "panthenol", "centella", "squalane", "allantoin", "colloidal oatmeal", "madecassoside"]

    for prod in products:
        p_name = prod.get("name", "Unnamed")
        p_cat = prod.get("category", "Treatment")
        multiplier = CATEGORY_VEHICLE_MULTIPLIERS.get(p_cat, 1.0)

        ingredients = prod.get("ingredients", [])
        total_ings = len(ingredients)

        for idx, ing in enumerate(ingredients):
            c_name = ing.get("canonical_name", "").lower()
            conc = ing.get("concentration")

            # Check for barrier soothing / buffering ingredients
            if any(b in c_name for b in buffering_ingredients):
                buffering_count += 1

            # Check for monitored potent active
            for active_key, base_weight in ACTIVE_IRRITATION_WEIGHTS.items():
                if active_key in c_name:
                    # Position weight: earlier ingredients have higher concentration
                    position_factor = 1.2 if idx < 5 else (1.0 if idx < 12 else 0.7)

                    # Explicit concentration adjustment if provided
                    conc_factor = 1.0
                    if conc is not None:
                        if conc > 5.0:
                            conc_factor = 1.4
                        elif conc > 2.0:
                            conc_factor = 1.2
                        elif conc < 0.5:
                            conc_factor = 0.8

                    item_load = base_weight * multiplier * position_factor * conc_factor
                    raw_score += item_load
                    active_load_items.append({
                        "product": p_name,
                        "category": p_cat,
                        "active": active_key.title(),
                        "vehicle_type": "Wash-off (Low contact)" if multiplier < 0.4 else "Leave-on (Full absorption)",
                        "calculated_points": round(item_load, 1)
                    })
                    break

    # Penalty for clashes and duplicates
    clash_penalty = clash_count * 15.0
    duplicate_penalty = duplicate_count * 10.0
    raw_score += (clash_penalty + duplicate_penalty)

    # Mitigation discount from barrier buffering ingredients (max 20 point relief)
    buffering_discount = min(20.0, buffering_count * 2.5)
    final_score = max(5.0, raw_score - buffering_discount)
    final_score = min(100.0, final_score)
    final_score = round(final_score, 1)

    # Classify barrier load level
    if final_score <= 35.0:
        level = "LOW"
        level_label = "Optimal / Barrier-Friendly"
        color = "green"
        summary = "Your routine maintains low chemical irritation potential and respects the skin acid mantle."
    elif final_score <= 65.0:
        level = "MODERATE"
        level_label = "Moderate Exfoliation / Active Load"
        color = "amber"
        summary = "Moderate barrier load. Well-tolerated by conditioned skin, but ensure adequate moisturizing."
    else:
        level = "HIGH"
        level_label = "High Stress / Irritation Risk"
        color = "red"
        summary = "Heavy active load detected. High risk of erythema, flaking, and lipid barrier depletion."

    return {
        "score": final_score,
        "level": level,
        "level_label": level_label,
        "color": color,
        "summary": summary,
        "buffering_ingredients_detected": buffering_count,
        "buffering_relief_points": round(buffering_discount, 1),
        "clash_penalty_applied": clash_penalty,
        "duplicate_penalty_applied": duplicate_penalty,
        "contributing_actives": active_load_items
    }


def find_active_synergies(all_canonical_ingredients: List[str]) -> List[Dict[str, Any]]:
    """Identifies clinically supported positive synergistic combinations present in the routine."""
    detected_synergies = []
    flattened_str = " ".join(all_canonical_ingredients).lower()

    for syn in ACTIVE_SYNERGIES:
        reqs = syn["ingredients_required"]
        matched_reqs = []
        for req in reqs:
            if any(req in ing for ing in all_canonical_ingredients):
                matched_reqs.append(req)

        if len(matched_reqs) == len(reqs):
            detected_synergies.append({
                "synergy_id": syn["synergy_id"],
                "title": syn["title"],
                "paired_ingredients": [r.title() for r in reqs],
                "benefit": syn["benefit"],
                "source": syn["source"]
            })

    return detected_synergies


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

        classified_ingredients = []
        for item in parsed_items:
            c_name = item["canonical_name"]
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

    # 5. Detect Active Synergies (Novelty Feature)
    synergies_detected = find_active_synergies(all_flattened_canonical)

    # 6. Compute Formulation-Aware Barrier Load Score (Novelty Feature)
    barrier_load = compute_barrier_load(
        products=processed_products,
        clash_count=len(warnings) + len(cautions),
        duplicate_count=len(duplicate_actives_list)
    )

    # 7. Routine Sequencing Recommendations
    sequenced_am = []
    sequenced_pm = []
    category_map = {p["category"]: p for p in processed_products}

    for cat in ROUTINE_STEP_ORDER_AM:
        if cat in category_map:
            sequenced_am.append(category_map[cat])
    for cat in ROUTINE_STEP_ORDER_PM:
        if cat in category_map:
            sequenced_pm.append(category_map[cat])

    if not warnings and not cautions and not avoid_violations:
        result_statuses.add("NO_CONFIGURED_CONFLICT_FOUND")

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
        "active_synergies": synergies_detected,
        "barrier_load": barrier_load,
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
        "synergies": res["active_synergies"],
        "barrier_load": res["barrier_load"],
        "am_steps_ordered": res["sequencing_recommendation"]["morning"],
        "pm_steps_ordered": res["sequencing_recommendation"]["evening"],
        "disclaimer": res["disclaimer"]
    }
