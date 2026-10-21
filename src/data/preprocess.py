"""
Preprocess and feature engineering module.
Performs text normalization, ingredient parsing, token extraction, and train/val/test splits.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cosmetics.csv")
INTERIM_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "cleaned_cosmetics.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")


def normalize_text(text: str) -> str:
    """Normalizes ingredient text string."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special non-alphanumeric punctuation except commas and dashes
    text = re.sub(r"[^\w\s,\-]", " ", text)
    # Condense multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_ingredients(ingredients_str: str) -> List[str]:
    """Parses a comma-separated ingredient string into a list of cleaned tokens."""
    if not isinstance(ingredients_str, str):
        return []
    parts = ingredients_str.split(",")
    cleaned = []
    for p in parts:
        token = normalize_text(p)
        if token and len(token) > 1 and "visit the sephora site" not in token:
            cleaned.append(token)
    return cleaned


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataframe, handles duplicates and normalizes ingredients."""
    df = raw_df.copy()
    # Drop exact brand + name duplicates if any
    df = df.drop_duplicates(subset=["Brand", "Name"]).reset_index(drop=True)
    
    # Text normalization
    df["Clean_Ingredients"] = df["Ingredients"].apply(normalize_text)
    
    # Filter out empty or placeholder ingredients
    df = df[df["Clean_Ingredients"].str.len() > 10].reset_index(drop=True)
    
    return df


def split_and_save_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    Splits dataset into stratified train, validation, and test sets.
    Saves sets to data/processed/.
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # First split train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Label"],
        random_state=random_state
    )
    
    # Split train vs val
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["Label"],
        random_state=random_state
    )
    
    train.to_csv(os.path.join(PROCESSED_PATH, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_PATH, "val.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_PATH, "test.csv"), index=False)
    
    print(f"Data split completed:")
    print(f"  Train: {len(train)} rows")
    print(f"  Val:   {len(val)} rows")
    print(f"  Test:  {len(test)} rows")
    return train, val, test


if __name__ == "__main__":
    print(f"Loading raw dataset from {RAW_PATH}")
    raw_data = pd.read_csv(RAW_PATH)
    cleaned = clean_dataset(raw_data)
    os.makedirs(os.path.dirname(INTERIM_PATH), exist_ok=True)
    cleaned.to_csv(INTERIM_PATH, index=False)
    print(f"Saved interim cleaned data: {len(cleaned)} rows")
    split_and_save_data(cleaned)
