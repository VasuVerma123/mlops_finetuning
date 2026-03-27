"""
Adapter stacking — merge a previous LoRA adapter into the base model,
then apply a fresh LoRA config on top.

Strategy: load base → load old adapter → merge_and_unload → apply new LoRA.
This allows iterative improvement without losing prior fine-tuning.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch
from peft import PeftModel, LoraConfig as PeftLoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from ..config import ModelConfig


def merge_adapter(
    model: "PreTrainedModel",
    adapter_path: str,
) -> "PreTrainedModel":
    """Load a LoRA adapter and merge it into the base weights."""
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    gc.collect()
    torch.cuda.empty_cache()
    return model


def apply_lora(
    model: "PreTrainedModel",
    model_cfg: "ModelConfig",
) -> "PreTrainedModel":
    """Prepare model for k-bit training and apply a fresh LoRA config."""
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_cfg.lora.r,
        lora_alpha=model_cfg.lora.alpha,
        lora_dropout=model_cfg.lora.dropout,
        target_modules=model_cfg.lora.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


def stack_and_apply(
    model: "PreTrainedModel",
    model_cfg: "ModelConfig",
    previous_adapter_path: str | None = None,
) -> "PreTrainedModel":
    """
    Full adapter stacking pipeline:
    1. If previous_adapter_path is given, merge it into base weights
    2. Apply a fresh LoRA config for the new training run
    """
    if previous_adapter_path:
        model = merge_adapter(model, previous_adapter_path)

    model = apply_lora(model, model_cfg)
    return model
