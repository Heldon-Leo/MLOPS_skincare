"""
Compatibility alias in rule_engine.py to support existing analyze_routine_compatibility signature
while using the new evaluate_multi_product_routine engine.
"""

from typing import Dict, Any, Optional, List
from src.rules.rule_engine import evaluate_multi_product_routine


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
    # Map back to legacy field names expected by old callers if any
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
