"""
Ingredient & Product Rules Reference
Evidence-backed compatibility, cautions, contraindications, and sequencing rules.
Source: Dermatology literature, CosIng database, and clinical ingredient guides.
Reviewed: 2026-10-23
Disclaimer: Educational reference only. Does not replace individualized medical advice.
"""

# Evidence-backed interactions and cautions
INGREDIENT_RULES = {
    "retinoids_and_aha_bha": {
        "ingredients_a": ["retinol", "retinoid", "retinal", "tretinoin", "adapalene"],
        "ingredients_b": ["glycolic acid", "salicylic acid", "lactic acid", "aha", "bha"],
        "severity": "caution",
        "title": "Simultaneous Exfoliant & Retinoid Caution",
        "description": "Combining strong direct acids (AHAs/BHAs) and retinoids in the same application window may increase barrier irritation, flaking, and sensitivity. Consider alternating evenings."
    },
    "vitamin_c_and_niacinamide": {
        "ingredients_a": ["ascorbic acid", "l-ascorbic acid"],
        "ingredients_b": ["niacinamide", "nicotinamide"],
        "severity": "info",
        "title": "Low pH Vitamin C & Niacinamide Notice",
        "description": "Historical concerns about niacinamide flushing with ascorbic acid are minimal in modern stabilized formulations; however, for reactive skin, staggered timing is often preferred."
    },
    "benzoyl_peroxide_and_retinoids": {
        "ingredients_a": ["benzoyl peroxide"],
        "ingredients_b": ["retinol", "tretinoin"],
        "severity": "caution",
        "title": "Benzoyl Peroxide & Unencapsulated Retinoid Oxidation",
        "description": "Benzoyl peroxide can oxidize certain unencapsulated retinoids (e.g., tretinoin, retinol) reducing their efficacy. Use benzoyl peroxide in the AM and retinoids in the PM."
    },
    "duplicate_strong_actives": {
        "ingredients_a": ["salicylic acid"],
        "ingredients_b": ["salicylic acid"],
        "severity": "warning",
        "title": "Duplicate Active Exfoliant (Salicylic Acid)",
        "description": "Multiple products in the routine contain Salicylic Acid (BHA). Layering multiple BHA products increases risk of barrier compromise."
    },
    "photosensitizing_actives_in_am": {
        "ingredients_a": ["retinol", "tretinoin", "glycolic acid"],
        "ingredients_b": [],  # Checked specifically for AM routine
        "severity": "warning",
        "title": "Photosensitizing Ingredient in Daytime Routine",
        "description": "Retinoids and strong alpha hydroxy acids increase skin photosensitivity to UV radiation. Best applied in evening routines; daily SPF 30+ is mandatory."
    }
}

# Clinically recognized ingredient synergy combinations (positive pairings)
ACTIVE_SYNERGIES = [
    {
        "synergy_id": "antioxidant_trio",
        "title": "Gold Standard Photoprotective Network",
        "ingredients_required": ["ascorbic acid", "tocopherol", "ferulic acid"],
        "benefit": "Combining L-Ascorbic Acid (Vitamin C), Alpha-Tocopherol (Vitamin E), and Ferulic Acid boosts chemical photoprotection 8-fold and dynamically regenerates oxidized Vitamin C.",
        "source": "Duke University Dermatological Study (Pinnell et al., J Invest Dermatol)"
    },
    {
        "synergy_id": "barrier_buffering",
        "title": "Retinoid Barrier-Buffering Synergy",
        "ingredients_required": ["retinol", "ceramide"],
        "benefit": "Ceramides strengthen intercellular lipids, significantly mitigating retinoid-induced trans-epidermal water loss (TEWL), flaking, and erythema.",
        "source": "Clinical & Aesthetic Dermatology Journal"
    },
    {
        "synergy_id": "sebum_pore_decongest",
        "title": "Sebum Regulating & Pore Clarifying Synergy",
        "ingredients_required": ["niacinamide", "salicylic acid"],
        "benefit": "Salicylic Acid (lipophilic BHA) penetrates and decongests sebaceous follicles, while Niacinamide downregulates sebum excretion and soothes micro-inflammation.",
        "source": "British Journal of Dermatology"
    },
    {
        "synergy_id": "multi_humectant_boost",
        "title": "Multi-Depth Hydration Synergy",
        "ingredients_required": ["hyaluronic acid", "glycerin"],
        "benefit": "Glycerin (low MW penetrating humectant) and Hyaluronic Acid (surface moisture binder) work synergistically across distinct epidermal strata for sustained hydration.",
        "source": "Cosmetic Dermatology Research"
    },
    {
        "synergy_id": "cica_soothing_synergy",
        "title": "Anti-Inflammatory Barrier Recovery Synergy",
        "ingredients_required": ["centella asiatica", "panthenol"],
        "benefit": "Madecassoside/Centella stimulates collagen type I synthesis while Panthenol accelerates epithelial repair and calms irritation.",
        "source": "Phytomedicine & Skin Pharmacology"
    }
]

# Standard routine sequence order (from lightest consistency / first step to heaviest / final step)
ROUTINE_STEP_ORDER_AM = [
    "Cleanser",
    "Treatment",
    "Eye cream",
    "Moisturizer",
    "Sun protect"
]

ROUTINE_STEP_ORDER_PM = [
    "Cleanser",
    "Face Mask",
    "Treatment",
    "Eye cream",
    "Moisturizer"
]

# Standard key beneficial ingredients mapped to skin concerns (evidence-reviewed)
CONCERN_KEY_INGREDIENTS = {
    "dryness": ["glycerin", "hyaluronic acid", "sodium hyaluronate", "ceramide", "squalane", "shea butter"],
    "oiliness": ["niacinamide", "zinc", "salicylic acid", "green tea", "tea tree"],
    "uneven texture": ["lactic acid", "glycolic acid", "mandelic acid", "niacinamide", "azelaic acid"],
    "appearance of blemishes": ["salicylic acid", "niacinamide", "zinc", "sulfur", "azelaic acid", "tea tree oil"]
}
