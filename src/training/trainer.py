"""
Core QLoRA fine-tuning engine.

Loads a model with quantization, applies adapter stacking, tokenizes
the dataset, and runs SFTTrainer.  Config-driven — everything comes
from PipelineConfig.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from ..config import PipelineConfig
from ..data.loader import load_and_prepare
from .adapter_stack import stack_and_apply
from .callbacks import LossLogger


def _get_bnb_config(cfg: PipelineConfig) -> Optional[BitsAndBytesConfig]:
    quant = cfg.model.quantization
    if quant == "none":
        return None

    bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if bf16_ok else torch.float16

    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_base_model(cfg: PipelineConfig):
    """Load the base model + tokenizer with quantization."""
    bnb = _get_bnb_config(cfg)
    kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
    )
    if bnb:
        kwargs["quantization_config"] = bnb

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_training(
    cfg: PipelineConfig,
    on_log: Optional[Callable] = None,
    on_complete: Optional[Callable] = None,
) -> dict:
    """
    Execute a full training run.

    Returns a summary dict with final loss, eval loss, timing, paths.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Load model ──
    model, tokenizer = load_base_model(cfg)
    torch.cuda.empty_cache()
    gc.collect()

    # ── Adapter stacking + fresh LoRA ──
    model = stack_and_apply(model, cfg.model, cfg.adapter_path)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} "
          f"({trainable / total_params * 100:.2f}%)")

    # ── Load + tokenize data ──
    train_ds, val_ds, token_stats = load_and_prepare(
        cfg.train_file,
        cfg.val_file if cfg.val_file else None,
        tokenizer,
        seed=cfg.training.seed,
        enable_thinking=cfg.model.enable_thinking,
        input_col=cfg.input_col,
        output_col=cfg.output_col,
        system_prompt=cfg.system_prompt,
    )

    # ── Compute schedule ──
    tc = cfg.training
    eff_batch = tc.batch_size * tc.gradient_accumulation
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    total_steps = steps_per_epoch * tc.epochs
    warmup_steps = max(10, int(total_steps * tc.warmup_ratio))
    eval_steps = max(total_steps // 10, 30)

    bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    log_file = os.path.join(cfg.output_dir, "loss_log.json")
    logger = LossLogger(total_steps=total_steps, log_file=log_file, on_log_fn=on_log)

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=tc.epochs,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=tc.scheduler,
        max_length=cfg.model.max_seq_length,
        dataset_text_field="text",
        packing=False,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        logging_steps=tc.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=tc.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim=tc.optimizer,
        max_grad_norm=tc.max_grad_norm,
        seed=tc.seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[logger],
        args=training_args,
    )

    torch.cuda.empty_cache()
    gc.collect()

    # ── Train ──
    trainer.train()

    # ── Save final adapter ──
    final_dir = os.path.join(cfg.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    summary = logger.get_summary()
    summary.update({
        "adapter_path": final_dir,
        "model_name": cfg.model.model_name,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "token_stats": token_stats,
        "trainable_params": trainable,
        "total_params": total_params,
    })

    if on_complete:
        on_complete(summary)

    return summary
