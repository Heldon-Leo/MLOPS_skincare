"""
Data & Feature Drift Monitoring Service
========================================
Monitors:
1. Unseen chemical ingredient vocabulary rates (vs training taxonomy).
2. Unknown prediction rate (% of classifications returning 'Unknown / Insufficient evidence').
3. Incomplete / missing ingredient report frequencies.
4. Category prediction distribution shifts against training baseline.
5. Request latency trends per endpoint.

Drift indicates distribution changes, not necessarily a decrease in accuracy.
Elevated unseen rates suggest the taxonomy should be expanded before retraining.

Limitations:
- Reference vocabulary is limited to ingredient_taxonomy.csv (77 records).
- Unseen token rate will naturally be HIGH for any real cosmetics product
  because the full INCI dictionary has 20,000+ entries.
- Use as a relative indicator: if rate increases sharply over time,
  investigate whether new actives require taxonomy expansion.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import Counter, deque
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ingredient_taxonomy.csv")
DRIFT_LOG_PATH = os.path.join(PROJECT_ROOT, "reports", "drift_history.json")

# Sliding window size for recent drift computation
WINDOW_SIZE = 100


class DriftMonitorService:
    """Computes statistical drift and unseen term frequency metrics."""

    def __init__(self):
        self.reference_vocab: set = set()
        self.reference_class_distribution: Dict[str, float] = {}

        # Cumulative counters
        self.observed_ingredients_total: int = 0
        self.unseen_ingredients_total: int = 0
        self.unseen_terms_seen: set = set()         # unique unseen terms (for de-duplication)
        self.unseen_terms_history: List[str] = []   # ordered list of new unseen terms

        # Prediction quality tracking
        self.total_classifications: int = 0
        self.unknown_classifications: int = 0       # predictions returning "Unknown / Insufficient evidence"
        self.class_prediction_counts: Counter = Counter()

        # Recent window for sliding drift estimate
        self._recent_seen_flags: deque = deque(maxlen=WINDOW_SIZE)  # True=known, False=unseen

        # Operational metrics
        self.endpoint_latencies: Dict[str, List[float]] = {}

        # Initialization
        self._load_reference_vocabulary()
        self._load_reference_class_distribution()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _load_reference_vocabulary(self):
        """Loads training ingredient vocabulary from taxonomy CSV."""
        if os.path.exists(TAXONOMY_PATH):
            try:
                df = pd.read_csv(TAXONOMY_PATH)
                self.reference_vocab = set(df["ingredient_name"].str.lower().str.strip().tolist())
            except Exception as e:
                print(f"[DriftMonitor] Could not load taxonomy: {e}")
        print(f"[DriftMonitor] Initialized with {len(self.reference_vocab)} reference vocabulary terms.")

    def _load_reference_class_distribution(self):
        """Loads reference class proportions from taxonomy CSV."""
        if os.path.exists(TAXONOMY_PATH):
            try:
                df = pd.read_csv(TAXONOMY_PATH)
                counts = df["functional_category"].value_counts()
                total = len(df)
                self.reference_class_distribution = {
                    cls: round(count / total, 4)
                    for cls, count in counts.items()
                }
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Ingestion methods (called during inference)
    # ------------------------------------------------------------------ #

    def log_observed_ingredients(self, ingredient_tokens: List[str]):
        """
        Logs observed ingredient tokens and calculates unseen token frequency.
        Call this after each ingredient parse operation.
        """
        for token in ingredient_tokens:
            clean = token.lower().strip()
            if not clean or len(clean) < 2:
                continue
            self.observed_ingredients_total += 1
            is_known = clean in self.reference_vocab
            self._recent_seen_flags.append(is_known)
            if not is_known:
                self.unseen_ingredients_total += 1
                if clean not in self.unseen_terms_seen:
                    self.unseen_terms_seen.add(clean)
                    self.unseen_terms_history.append(clean)

    def log_classification_result(self, predicted_function: str):
        """
        Records a classification prediction result.
        Call after each /ingredients/classify inference.
        """
        self.total_classifications += 1
        if "unknown" in predicted_function.lower() or "insufficient" in predicted_function.lower():
            self.unknown_classifications += 1
        else:
            self.class_prediction_counts[predicted_function] += 1

    def log_endpoint_latency(self, endpoint: str, latency_seconds: float):
        """Records endpoint response time for latency drift tracking."""
        if endpoint not in self.endpoint_latencies:
            self.endpoint_latencies[endpoint] = []
        self.endpoint_latencies[endpoint].append(round(latency_seconds, 4))
        # Keep only last 200 observations per endpoint
        if len(self.endpoint_latencies[endpoint]) > 200:
            self.endpoint_latencies[endpoint] = self.endpoint_latencies[endpoint][-200:]

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def _get_unseen_rate(self) -> float:
        if self.observed_ingredients_total == 0:
            return 0.0
        return round(self.unseen_ingredients_total / self.observed_ingredients_total, 4)

    def _get_recent_unseen_rate(self) -> float:
        """Sliding window unseen rate over last WINDOW_SIZE observations."""
        if not self._recent_seen_flags:
            return 0.0
        unseen = sum(1 for flag in self._recent_seen_flags if not flag)
        return round(unseen / len(self._recent_seen_flags), 4)

    def _get_unknown_prediction_rate(self) -> float:
        if self.total_classifications == 0:
            return 0.0
        return round(self.unknown_classifications / self.total_classifications, 4)

    def _get_class_distribution_shift(self) -> Dict[str, Any]:
        """Compares live class distribution to reference distribution."""
        if self.total_classifications == 0 or not self.class_prediction_counts:
            return {"status": "INSUFFICIENT_DATA", "details": {}}
        total = sum(self.class_prediction_counts.values())
        live_dist = {
            cls: round(count / total, 4)
            for cls, count in self.class_prediction_counts.items()
        }
        shifts = {}
        all_classes = set(list(self.reference_class_distribution.keys()) + list(live_dist.keys()))
        for cls in all_classes:
            ref_prop = self.reference_class_distribution.get(cls, 0.0)
            live_prop = live_dist.get(cls, 0.0)
            delta = round(live_prop - ref_prop, 4)
            shifts[cls] = {"reference": ref_prop, "live": live_prop, "delta": delta}
        return {"status": "COMPUTED", "details": shifts}

    def _get_latency_summary(self) -> Dict[str, Any]:
        summary = {}
        for ep, latencies in self.endpoint_latencies.items():
            if latencies:
                summary[ep] = {
                    "p50_ms": round(float(np.percentile(latencies, 50)) * 1000, 1),
                    "p95_ms": round(float(np.percentile(latencies, 95)) * 1000, 1),
                    "count": len(latencies)
                }
        return summary

    def get_drift_report(self) -> Dict[str, Any]:
        """Returns comprehensive drift metrics summary."""
        unseen_rate = self._get_unseen_rate()
        recent_unseen_rate = self._get_recent_unseen_rate()
        unknown_pred_rate = self._get_unknown_prediction_rate()

        # Composite severity — based on both unseen rate and unknown prediction rate
        if unseen_rate > 0.60 or unknown_pred_rate > 0.50:
            drift_severity = "HIGH"
        elif unseen_rate > 0.30 or unknown_pred_rate > 0.25:
            drift_severity = "MODERATE"
        else:
            drift_severity = "LOW"

        return {
            "status": "DRIFT_EVALUATED",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "vocabulary_drift": {
                "total_tokens_observed": self.observed_ingredients_total,
                "unseen_tokens_count": self.unseen_ingredients_total,
                "unique_unseen_terms": len(self.unseen_terms_seen),
                "cumulative_unseen_rate": unseen_rate,
                "recent_unseen_rate": recent_unseen_rate,
                f"recent_window_size": len(self._recent_seen_flags),
                "sample_unseen_terms": self.unseen_terms_history[-10:]
            },
            "prediction_drift": {
                "total_classifications": self.total_classifications,
                "unknown_classifications": self.unknown_classifications,
                "unknown_prediction_rate": unknown_pred_rate,
                "live_class_counts": dict(self.class_prediction_counts),
                "class_distribution_shift": self._get_class_distribution_shift()
            },
            "drift_severity": drift_severity,
            "latency_summary": self._get_latency_summary(),
            "statistical_limitation_note": (
                "Unseen ingredient rate naturally exceeds 90% for any product-scale inference because the "
                "reference vocabulary contains only the 77 taxonomy training records. Use the TREND of this "
                "metric over time rather than its absolute value. A sharp increase in new unseen terms relative "
                "to a stable baseline indicates taxonomy drift. Unknown prediction rate > 50% is a stronger "
                "signal that retraining or taxonomy expansion is warranted."
            ),
            "reference_vocab_size": len(self.reference_vocab),
            "reference_class_distribution": self.reference_class_distribution
        }

    def save_drift_snapshot(self):
        """Persists current drift state to JSON for historical analysis."""
        os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
        snapshot = self.get_drift_report()
        history = []
        if os.path.exists(DRIFT_LOG_PATH):
            try:
                with open(DRIFT_LOG_PATH, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(snapshot)
        # Keep last 100 snapshots
        history = history[-100:]
        with open(DRIFT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


drift_monitor = DriftMonitorService()
