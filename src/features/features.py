"""
Feature Engineering and Pipeline Transformers
Defines custom scikit-learn compatible transformers for skincare ingredient and tabular features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class TextSelector(BaseEstimator, TransformerMixin):
    """Selects a specific text column for TF-IDF processing."""
    def __init__(self, key: str = "Clean_Ingredients"):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.key].fillna("").astype(str)
        return pd.Series(X).fillna("").astype(str)


class TabularFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects and scales tabular numerical and binary suitability indicators."""
    def __init__(self, feature_cols=None):
        if feature_cols is None:
            feature_cols = ["Price", "Rank", "Combination", "Dry", "Normal", "Oily", "Sensitive"]
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Fill missing if any
            vals = X[self.feature_cols].copy()
            vals["Price"] = vals["Price"].fillna(vals["Price"].median())
            vals["Rank"] = vals["Rank"].fillna(vals["Rank"].mean())
            for col in ["Combination", "Dry", "Normal", "Oily", "Sensitive"]:
                if col in vals.columns:
                    vals[col] = vals[col].fillna(0).astype(float)
            return vals.values
        return np.zeros((len(X), len(self.feature_cols)))


def build_feature_pipeline() -> FeatureUnion:
    """
    Constructs a FeatureUnion combining:
    1. TF-IDF vectorizer for ingredients (unigrams + bigrams, top 1000 features)
    2. Scaled numerical and binary skin-type flags
    """
    text_pipe = Pipeline([
        ("selector", TextSelector(key="Clean_Ingredients")),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            sublinear_tf=True
        ))
    ])

    tabular_pipe = Pipeline([
        ("selector", TabularFeatureSelector()),
        ("scaler", StandardScaler())
    ])

    union = FeatureUnion([
        ("text", text_pipe),
        ("tabular", tabular_pipe)
    ])

    return union
