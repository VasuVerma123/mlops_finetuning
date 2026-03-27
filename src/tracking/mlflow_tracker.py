"""
MLflow experiment tracker — log params, metrics, artifacts, and models.

Wraps mlflow to provide a clean interface for the pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..config import PipelineConfig


class MLflowTracker:
    """Thin wrapper around MLflow for experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "echo-finetuning",
        tracking_uri: str = "mlruns",
        enabled: bool = True,
    ):
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.run = None

        if self.enabled:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        """Start an MLflow run."""
        if not self.enabled:
            return
        self.run = mlflow.start_run(run_name=run_name, tags=tags or {})
        return self.run

    def end_run(self):
        """End the active run."""
        if not self.enabled or not self.run:
            return
        mlflow.end_run()
        self.run = None

    def log_pipeline_config(self, cfg: PipelineConfig):
        """Log all pipeline config parameters."""
        if not self.enabled:
            return

        mlflow.log_params({
            "model_name": cfg.model.model_name,
            "model_family": cfg.model.family,
            "model_size": cfg.model.size,
            "quantization": cfg.model.quantization,
            "lora_r": cfg.model.lora.r,
            "lora_alpha": cfg.model.lora.alpha,
            "lora_dropout": cfg.model.lora.dropout,
            "max_seq_length": cfg.model.max_seq_length,
            "enable_thinking": cfg.model.enable_thinking,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "gradient_accumulation": cfg.training.gradient_accumulation,
            "learning_rate": cfg.training.learning_rate,
            "scheduler": cfg.training.scheduler,
            "optimizer": cfg.training.optimizer,
            "adapter_stacking": cfg.adapter_path is not None,
        })

    def log_training_step(self, step: int, metrics: dict[str, Any]):
        """Log metrics from a training step."""
        if not self.enabled:
            return
        log_metrics = {}
        if "loss" in metrics:
            log_metrics["train_loss"] = metrics["loss"]
        if "lr" in metrics:
            log_metrics["learning_rate"] = metrics["lr"]
        if "eval_loss" in metrics:
            log_metrics["eval_loss"] = metrics["eval_loss"]
        if "gpu_used_gb" in metrics:
            log_metrics["gpu_memory_gb"] = metrics["gpu_used_gb"]
        if log_metrics:
            mlflow.log_metrics(log_metrics, step=step)

    def log_training_summary(self, summary: dict):
        """Log final training summary metrics."""
        if not self.enabled:
            return
        metrics = {}
        if summary.get("final_train_loss") is not None:
            metrics["final_train_loss"] = summary["final_train_loss"]
        if summary.get("best_eval_loss") is not None:
            metrics["best_eval_loss"] = summary["best_eval_loss"]
        if summary.get("total_time_s"):
            metrics["training_time_s"] = summary["total_time_s"]
        if summary.get("trainable_params"):
            metrics["trainable_params"] = summary["trainable_params"]
        if metrics:
            mlflow.log_metrics(metrics)

    def log_evaluation_results(self, results: dict):
        """Log evaluation metrics."""
        if not self.enabled:
            return
        mlflow.log_metrics({
            "overall_accuracy_pct": results.get("overall_accuracy_pct", 0),
            "avg_latency_s": results.get("avg_latency_s", 0),
            "avg_tokens_per_sec": results.get("avg_tokens_per_sec", 0),
            "avg_ttft_s": results.get("avg_ttft_s", 0),
            "p95_latency_s": results.get("p95_latency_s", 0),
        })

        for sc in results.get("scenarios", []):
            safe_name = sc["name"].replace(" ", "_").replace(".", "")
            mlflow.log_metric(f"acc_{safe_name}", sc["accuracy_pct"])

    def log_artifact(self, local_path: str, artifact_path: str = ""):
        """Log a file or directory as an artifact."""
        if not self.enabled:
            return
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)

    def log_dict_artifact(self, data: dict, filename: str):
        """Log a dict as a JSON artifact."""
        if not self.enabled:
            return
        tmp = Path(filename)
        tmp.write_text(json.dumps(data, indent=2, default=str))
        mlflow.log_artifact(str(tmp))
        tmp.unlink(missing_ok=True)

    def log_adapter(self, adapter_dir: str):
        """Log the final adapter directory as an artifact."""
        if not self.enabled or not os.path.isdir(adapter_dir):
            return
        mlflow.log_artifacts(adapter_dir, "adapter")

    def list_experiments(self) -> list[dict]:
        """List all runs in the current experiment."""
        if not self.enabled:
            return []
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        return runs.to_dict("records") if len(runs) > 0 else []

    def get_run_details(self, run_id: str) -> dict:
        """Get details for a specific run."""
        if not self.enabled:
            return {}
        run = mlflow.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }
