"""
Training callbacks — loss logging, progress reporting, MLflow integration.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from transformers import TrainerCallback


class LossLogger(TrainerCallback):
    """Collects train/eval losses and writes them to a JSON file."""

    def __init__(
        self,
        total_steps: int,
        log_file: str,
        on_log_fn: Optional[Callable] = None,
    ):
        self.total = total_steps
        self.log_file = log_file
        self.on_log_fn = on_log_fn    # optional callback for Streamlit / MLflow
        self.t0: Optional[float] = None
        self.train_losses: list[dict] = []
        self.eval_losses: list[dict] = []

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or self.t0 is None:
            return
        step = state.global_step
        elapsed = time.time() - self.t0

        entry: dict[str, Any] = {"step": step, "elapsed_s": round(elapsed, 1)}

        if "loss" in logs:
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0)
            self.train_losses.append({"step": step, "loss": loss, "lr": lr})
            entry.update(loss=loss, lr=lr)

            sps = elapsed / max(step, 1)
            entry["eta_s"] = round(sps * (self.total - step))

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                entry.update(gpu_used_gb=round(mem, 2), gpu_total_gb=round(total, 2))

        if "eval_loss" in logs:
            ev = logs["eval_loss"]
            self.eval_losses.append({"step": step, "eval_loss": ev})
            entry["eval_loss"] = ev

        if self.on_log_fn:
            try:
                self.on_log_fn(entry)
            except Exception:
                pass

    def on_save(self, args, state, control, **kw):
        self._dump()

    def on_train_end(self, args, state, control, **kw):
        self._dump()
        total_time = time.time() - (self.t0 or time.time())
        summary = {
            "total_time_s": round(total_time),
            "total_time_human": str(timedelta(seconds=int(total_time))),
            "final_loss": self.train_losses[-1]["loss"] if self.train_losses else None,
            "best_eval_loss": (
                min(e["eval_loss"] for e in self.eval_losses)
                if self.eval_losses else None
            ),
        }
        if self.on_log_fn:
            try:
                self.on_log_fn({"event": "train_end", **summary})
            except Exception:
                pass

    def _dump(self):
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump({
                "train_losses": self.train_losses,
                "eval_losses": self.eval_losses,
            }, f, indent=2)

    def get_summary(self) -> dict:
        total_time = time.time() - (self.t0 or time.time())
        return {
            "total_time_s": round(total_time),
            "steps_completed": len(self.train_losses),
            "final_train_loss": self.train_losses[-1]["loss"] if self.train_losses else None,
            "best_eval_loss": (
                min(e["eval_loss"] for e in self.eval_losses)
                if self.eval_losses else None
            ),
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
        }
