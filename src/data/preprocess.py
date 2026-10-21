"""
Data Preprocessing & Normalization Module
Handles:
1. Punctuation and whitespace normalization.
2. Cosmetic and INCI synonym mapping.
3. Ingredient parsing and delimiter splitting.
4. Active concentration percentage extraction (e.g. 'Niacinamide 10%' -> name: 'niacinamide', percentage: 10.0%).
"""

import os
import re
import json
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNONYMS_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "ingredient_synonyms.json")


def load_synonyms() -> Dict[str, str]:
    """Loads synonym mappings from JSON reference file."""
    if os.path.exists(SYNONYMS_PATH):
        try:
            with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("mappings", {})
        except Exception:
            return {}
    return {}


SYNONYMS_MAP = load_synonyms()


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing, unicode normalization, and standardizing whitespace."""
    if not isinstance(text, str):
        return ""
    # Unicode NFKD normalization (converts characters like é to e)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    # Replace slashes and semicolons with commas as delimiter markers
    text = re.sub(r"[/;\|]", ",", text)
    # Remove non-alphanumeric characters except standard punctuation needed for percentages, hyphens, commas
    text = re.sub(r"[^\w\s,%.\-]", " ", text)
    # Condense multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_concentration(raw_token: str) -> Tuple[str, Optional[float]]:
    """
    Extracts explicit active percentage if explicitly stated in ingredient token.
    Examples:
        'Salicylic Acid 2%' -> ('salicylic acid', 2.0)
        'Niacinamide 10.5 %' -> ('niacinamide', 10.5)
        'Glycerin' -> ('glycerin', None)
    """
    clean = normalize_text(raw_token)
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", clean)
    if match:
        pct = float(match.group(1))
        # Remove percentage portion from ingredient name
        name = re.sub(r"(\d+(?:\.\d+)?)\s*%", "", clean).strip()
        # Clean stray hyphens or punctuation
        name = re.sub(r"^[\s\-]+|[\s\-]+$", "", name).strip()
        return name, pct
    return clean, None


def normalize_ingredient_name(raw_name: str) -> str:
    """Applies canonical synonym mapping and cleans name."""
    clean_name, _ = extract_concentration(raw_name)
    clean_name = re.sub(r"\s+", " ", clean_name).strip()
    # Check if direct match exists in synonym map
    if clean_name in SYNONYMS_MAP:
        return SYNONYMS_MAP[clean_name]
    # Check if a mapped key is a distinct whole word inside the token
    for syn_key, canonical in SYNONYMS_MAP.items():
        if re.search(rf"\b{re.escape(syn_key)}\b", clean_name):
            return canonical
    return clean_name


def parse_ingredients_with_metadata(ingredients_str: str) -> List[Dict[str, Any]]:
    """
    Parses comma-separated ingredients into structured list containing:
    - raw_token: original text fragment
    - canonical_name: normalized INCI/chemical name
    - concentration: explicit numeric percentage if stated, else None
    """
    if not isinstance(ingredients_str, str):
        return []

    # Filter out disclaimer placeholders
    if "visit the sephora site" in ingredients_str.lower():
        return []

    raw_tokens = re.split(r",(?![^\(]*\))", ingredients_str)  # Split on commas outside parentheses
    parsed_items = []
    seen_canonical = set()

    for token in raw_tokens:
        token = token.strip()
        if not token or len(token) < 2:
            continue

        name_part, concentration = extract_concentration(token)
        canonical = normalize_ingredient_name(name_part)

        if not canonical or len(canonical) < 2:
            continue

        parsed_items.append({
            "raw_token": token,
            "canonical_name": canonical,
            "concentration": concentration,
            "has_explicit_concentration": concentration is not None,
            "concentration_source": "Explicitly provided by input" if concentration is not None else "Not provided"
        })
        seen_canonical.add(canonical)

    return parsed_items


def parse_ingredients(ingredients_str: str) -> List[str]:
    """Legacy compatibility helper returning list of canonical string tokens."""
    items = parse_ingredients_with_metadata(ingredients_str)
    return [item["canonical_name"] for item in items]


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataframe, handles duplicates and normalizes ingredients."""
    df = raw_df.copy()
    df = df.drop_duplicates(subset=["Brand", "Name"]).reset_index(drop=True)
    df["Clean_Ingredients"] = df["Ingredients"].apply(normalize_text)
    df = df[df["Clean_Ingredients"].str.len() > 10].reset_index(drop=True)
    return df
