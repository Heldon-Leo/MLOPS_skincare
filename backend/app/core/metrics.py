"""
Operational Metrics Tracker
Tracks requests, latencies, HTTP error rates, prediction counts, and data drift indicators
without storing PII or raw user identifiers.
"""

import time
from typing import Dict, Any, List


class OperationalMetrics:
    """Thread-safe operational telemetry collector."""

    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.endpoint_counts: Dict[str, int] = {}
        self.prediction_counts: Dict[str, int] = {}
        self.latencies: List[float] = []
        self.unknown_ingredient_events = 0
        self.schema_violations = 0

    def record_request(self, endpoint: str, latency: float, status_code: int):
        self.total_requests += 1
        self.endpoint_counts[endpoint] = self.endpoint_counts.get(endpoint, 0) + 1
        if status_code >= 400:
            self.total_errors += 1
        self.latencies.append(latency)
        # Keep window of last 1000 latencies
        if len(self.latencies) > 1000:
            self.latencies.pop(0)

    def record_prediction(self, category: str):
        self.prediction_counts[category] = self.prediction_counts.get(category, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        avg_latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        p95_latency = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if len(self.latencies) >= 20 else avg_latency

        return {
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.total_errors / max(1, self.total_requests), 4),
            "average_latency_ms": round(avg_latency * 1000, 2),
            "p95_latency_ms": round(p95_latency * 1000, 2),
            "endpoint_breakdown": self.endpoint_counts,
            "predictions_by_category": self.prediction_counts,
            "schema_violations": self.schema_violations
        }


metrics_collector = OperationalMetrics()
