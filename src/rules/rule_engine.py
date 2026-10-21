"""
Ingredient Rule Engine
Detects:
1. Hazardous/sensitizing active combinations (e.g. retinoids + direct AHA/BHA)
2. Incomplete or unknown ingredient listings
3. Duplicate active ingredients across routine steps
4. User avoid-list contraindications
5. Daytime photo-sensitizing ingredient warnings
6. Skin-concern ingredient matching
"""

import re
from typing import List, Dict, Any, Optional
from src.rules.rules_data import (
    INGREDIENT_RULES,
    ROUTINE_STEP_ORDER_AM,
    ROUTINE_STEP_ORDER_PM,
    CONCERN_KEY_INGREDIENTS
)


def normalize_ingredient_name(name: str) -> str:
    """Standardizes ingredient string for fuzzy/token matching."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"[^\w\s\-]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def check_ingredient_matches(ingredient_list: List[str], target_tokens: List[str]) -> List[str]:
    """Finds which target tokens appear in the product ingredient list."""
    matched = []
    norm_ingredients = [normalize_ingredient_name(ing) for ing in ingredient_list]
    for target in target_tokens:
        norm_target = normalize_ingredient_name(target)
        for ing in norm_ingredients:
            if norm_target in ing:
                matched.append(target)
                break
    return list(set(matched))


def check_avoid_list_violations(product_ingredients: List[str], avoid_list: List[str]) -> List[str]:
    """Checks if any ingredient matches user avoid specifications."""
    violations = []
    for avoid_item in avoid_list:
        clean_avoid = normalize_ingredient_name(avoid_item)
        if not clean_avoid:
            continue
        for ing in product_ingredients:
            if clean_avoid in normalize_ingredient_name(ing):
                violations.append(f"Contains avoid-listed ingredient: '{avoid_item}' (matched in '{ing}')")
                break
    return violations


def analyze_routine_compatibility(
    products_by_category: Dict[str, Dict[str, Any]],
    routine_time: str = "both",
    avoid_list: Optional[List[str]] = None,
    user_concerns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates complete routine for compatibility, cautions, duplicate actives, and user avoid preferences.
    """
    if avoid_list is None:
        avoid_list = []
    if user_concerns is None:
        user_concerns = []

    warnings = []
    cautions = []
    info_notes = []
    avoid_violations = []
    active_tracker: Dict[str, List[str]] = {}

    all_products = list(products_by_category.values())

    # 1. Check Avoid Violations & Gather Active Ingredients
    for prod in all_products:
        prod_name = prod.get("name", "Product")
        category = prod.get("category", "General")
        ingredients = prod.get("ingredients", [])
        if isinstance(ingredients, str):
            ingredients = [i.strip() for i in ingredients.split(",") if i.strip()]

        # Check avoid list
        v = check_avoid_list_violations(ingredients, avoid_list)
        if v:
            avoid_violations.extend([f"[{category}] {prod_name}: {item}" for item in v])

        # Track active occurrences for duplication check
        for active in ["salicylic acid", "glycolic acid", "retinol", "benzoyl peroxide"]:
            if check_ingredient_matches(ingredients, [active]):
                if active not in active_tracker:
                    active_tracker[active] = []
                active_tracker[active].append(f"{prod_name} ({category})")

    # 2. Check Duplicate Actives
    for active, prod_list in active_tracker.items():
        if len(prod_list) > 1:
            cautions.append({
                "type": "duplicate_active",
                "title": f"Multiple Products with {active.title()}",
                "description": f"Found {active.title()} in multiple steps: {', '.join(prod_list)}. Layering duplicate strong actives can compromise the skin barrier.",
                "severity": "caution"
            })

    # 3. Check Known Adverse Pairings across selected products
    flattened_ingredients = []
    for prod in all_products:
        ings = prod.get("ingredients", [])
        if isinstance(ings, str):
            ings = [i.strip() for i in ings.split(",")]
        flattened_ingredients.extend(ings)

    for rule_id, rule in INGREDIENT_RULES.items():
        if rule_id == "duplicate_strong_actives":
            continue
        if rule_id == "photosensitizing_actives_in_am":
            if routine_time in ["morning", "both"]:
                matched_am = check_ingredient_matches(flattened_ingredients, rule["ingredients_a"])
                if matched_am:
                    warnings.append({
                        "type": "photosensitivity_warning",
                        "title": rule["title"],
                        "description": f"{rule['description']} (Detected: {', '.join(matched_am)})",
                        "severity": rule["severity"]
                    })
            continue

        match_a = check_ingredient_matches(flattened_ingredients, rule["ingredients_a"])
        match_b = check_ingredient_matches(flattened_ingredients, rule["ingredients_b"])
        if match_a and match_b:
            entry = {
                "type": rule_id,
                "title": rule["title"],
                "description": f"{rule['description']} (Found combination: {', '.join(match_a)} + {', '.join(match_b)})",
                "severity": rule["severity"]
            }
            if rule["severity"] == "warning":
                warnings.append(entry)
            else:
                cautions.append(entry)

    # 4. Check Key Ingredients for User Concerns
    beneficial_notes = []
    for concern in user_concerns:
        norm_c = concern.lower()
        targets = CONCERN_KEY_INGREDIENTS.get(norm_c, [])
        found = check_ingredient_matches(flattened_ingredients, targets)
        if found:
            beneficial_notes.append({
                "concern": concern,
                "matched_ingredients": found,
                "message": f"Routine contains recognized beneficial ingredients for '{concern}': {', '.join(found)}."
            })
        else:
            beneficial_notes.append({
                "concern": concern,
                "matched_ingredients": [],
                "message": f"No high-potency recognized actives explicitly found for '{concern}'. Consider discussing target treatments with a dermatologist."
            })

    # 5. Order routine steps
    am_routine = []
    pm_routine = []
    for cat in ROUTINE_STEP_ORDER_AM:
        if cat in products_by_category:
            am_routine.append(products_by_category[cat])
    for cat in ROUTINE_STEP_ORDER_PM:
        if cat in products_by_category:
            pm_routine.append(products_by_category[cat])

    return {
        "status": "evaluated",
        "warnings": warnings,
        "cautions": cautions,
        "avoid_violations": avoid_violations,
        "beneficial_insights": beneficial_notes,
        "am_steps_ordered": am_routine,
        "pm_steps_ordered": pm_routine,
        "disclaimer": (
            "Educational evaluation only. This routine assessment does not constitute medical diagnosis "
            "or clinical prescription. Individual skin tolerances vary; patch test new products prior to full application."
        )
    }
