"""
Lightweight Tracking Logger and MLflow Integration Bridge
Provides standard experiment logging (params, metrics, tags, artifacts, runs)
with an automatic fallback to local JSON experiment files if MLflow is initializing or running in slim mode.
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Standardized experiment tracker compatible with MLflow and local JSON logs."""

    def __init__(self, experiment_name: str = "skincare_classification"):
        self.experiment_name = experiment_name
        self.run_data = {
            "experiment_name": experiment_name,
            "run_id": "",
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
            "start_time": "",
            "end_time": "",
            "status": "RUNNING"
        }
        self.use_mlflow = MLFLOW_AVAILABLE
        self.active_run = None

    def start_run(self, run_name: Optional[str] = None):
        """Starts a new experiment tracking run."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
        self.run_data["run_id"] = self.run_id
        self.run_data["run_name"] = run_name or self.run_id
        self.run_data["start_time"] = datetime.utcnow().isoformat()

        if self.use_mlflow:
            try:
                mlflow.set_experiment(self.experiment_name)
                self.active_run = mlflow.start_run(run_name=self.run_data["run_name"])
            except Exception as e:
                print(f"MLflow start_run notice: {e}. Falling back to structured local tracker.")
                self.use_mlflow = False

        return self

    def log_param(self, key: str, value: Any):
        """Logs a single hyperparameter or configuration."""
        self.run_data["params"][key] = value
        if self.use_mlflow and self.active_run:
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass

    def log_params(self, params: Dict[str, Any]):
        """Logs multiple parameters."""
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float):
        """Logs an evaluation metric."""
        self.run_data["metrics"][key] = round(float(value), 4)
        if self.use_mlflow and self.active_run:
            try:
                mlflow.log_metric(key, float(value))
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float]):
        """Logs a dictionary of metrics."""
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_tag(self, key: str, value: str):
        """Logs metadata tag."""
        self.run_data["tags"][key] = str(value)
        if self.use_mlflow and self.active_run:
            try:
                mlflow.set_tag(key, str(value))
            except Exception:
                pass

    def end_run(self, status: str = "FINISHED"):
        """Finalizes run and writes local manifest."""
        self.run_data["status"] = status
        self.run_data["end_time"] = datetime.utcnow().isoformat()

        if self.use_mlflow and self.active_run:
            try:
                mlflow.end_run()
            except Exception:
                pass

        # Save run metadata locally in mlruns/runs
        runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mlruns", "tracking")
        os.makedirs(runs_dir, exist_ok=True)
        run_file = os.path.join(runs_dir, f"{self.run_id}.json")
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=2)

        return self.run_data
