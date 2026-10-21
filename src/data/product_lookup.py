"""
Product Lookup and Search Engine Service
Supports:
1. Exact product name & brand matching.
2. Partial string matching across Brand + Name.
3. Case-insensitive token search.
4. Fuzzy search with Levenshtein similarity ranking.
5. Missing ingredient detection and ambiguity handling.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz, process

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "cleaned_cosmetics.csv")


class ProductLookupService:
    """Provides product search, ranking, and ingredient retrieval over the cosmetics database."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or DATA_PATH
        self.df = None
        self.load_catalog()

    def load_catalog(self):
        """Loads and indexes the cosmetics dataset."""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            # Create a unique product ID column
            self.df["product_id"] = [f"prod_{i+1:04d}" for i in range(len(self.df))]
            # Search helper column combining Brand and Name
            self.df["search_key"] = (
                self.df["Brand"].astype(str).str.lower() + " " + self.df["Name"].astype(str).str.lower()
            )
            print(f"[ProductLookupService] Loaded {len(self.df)} products from {self.data_path}")
        else:
            print(f"[ProductLookupService] Warning: File not found at {self.data_path}")
            self.df = pd.DataFrame()

    def search_products(
        self,
        query: str,
        brand_filter: Optional[str] = None,
        limit: int = 10,
        fuzzy_threshold: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Searches catalog for products matching query.
        Returns ranked list of matching records with match type (EXACT, PARTIAL, FUZZY).
        """
        if self.df is None or self.df.empty or not query or not query.strip():
            return []

        clean_q = query.lower().strip()
        results = []

        sub_df = self.df
        if brand_filter and brand_filter.strip():
            b_clean = brand_filter.lower().strip()
            sub_df = sub_df[sub_df["Brand"].str.lower().str.contains(b_clean, na=False)]

        # 1. Exact Name / Search Key Matches
        exact_matches = sub_df[sub_df["search_key"] == clean_q]
        if not exact_matches.empty:
            for _, row in exact_matches.iterrows():
                results.append(self._format_product_row(row, match_type="EXACT", match_score=100.0))

        # 2. Partial Substring Matches
        partial_matches = sub_df[sub_df["search_key"].str.contains(re.escape(clean_q), na=False)]
        for _, row in partial_matches.iterrows():
            if row["product_id"] not in [r["product_id"] for r in results]:
                results.append(self._format_product_row(row, match_type="PARTIAL", match_score=85.0))

        # 3. Fuzzy Matching if results under limit
        if len(results) < limit:
            keys = sub_df["Name"].tolist()
            fuzzy_hits = process.extract(
                query,
                keys,
                scorer=fuzz.WRatio,
                limit=limit * 2
            )
            for hit in fuzzy_hits:
                match_str, score, idx = hit
                if score >= fuzzy_threshold:
                    row = sub_df.iloc[idx]
                    if row["product_id"] not in [r["product_id"] for r in results]:
                        results.append(self._format_product_row(row, match_type="FUZZY", match_score=float(score)))

        # Sort results: EXACT > PARTIAL > FUZZY (by score desc)
        def sort_key(item):
            order = {"EXACT": 0, "PARTIAL": 1, "FUZZY": 2}
            return (order.get(item["match_type"], 3), -item["match_score"])

        results = sorted(results, key=sort_key)
        return results[:limit]

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves single product record by its product_id."""
        if self.df is None or self.df.empty:
            return None
        match = self.df[self.df["product_id"] == product_id]
        if not match.empty:
            return self._format_product_row(match.iloc[0], match_type="ID_LOOKUP", match_score=100.0)
        return None

    def _format_product_row(self, row: pd.Series, match_type: str, match_score: float) -> Dict[str, Any]:
        """Formats DataFrame row into API-ready dictionary."""
        ing_raw = str(row.get("Ingredients", ""))
        has_ingredients = len(ing_raw.strip()) > 10 and "visit the sephora site" not in ing_raw.lower()

        return {
            "product_id": str(row.get("product_id", "")),
            "brand": str(row.get("Brand", "")),
            "name": str(row.get("Name", "")),
            "category": str(row.get("Label", "")),
            "price": float(row.get("Price", 0.0)),
            "rating": float(row.get("Rank", 0.0)),
            "ingredients_raw": ing_raw if has_ingredients else None,
            "has_ingredients": has_ingredients,
            "source": "Sephora & INCI Local Dataset",
            "match_type": match_type,
            "match_score": round(match_score, 1),
            "skin_suitability": {
                "combination": int(row.get("Combination", 0)),
                "dry": int(row.get("Dry", 0)),
                "normal": int(row.get("Normal", 0)),
                "oily": int(row.get("Oily", 0)),
                "sensitive": int(row.get("Sensitive", 0))
            }
        }


product_lookup_service = ProductLookupService()
