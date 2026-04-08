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


def _compute_generation_accuracy(model, tokenizer, cfg, val_ds, max_samples=10):
    """Generate on val examples and score how well the model reproduces expected outputs."""
    import re, json as _json
    try:
        raw_file = cfg.val_file if cfg.val_file else cfg.train_file
        if not raw_file:
            return None
        with open(raw_file, "r", encoding="utf-8") as f:
            raw = [_json.loads(line) for line in f if line.strip()]
        if not raw:
            return None

        raw = raw[:max_samples]
        model.eval()
        correct = 0
        total = 0

        for ex in raw:
            msgs = ex.get("messages", [])
            if len(msgs) < 2:
                continue

            expected_msg = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
            if not expected_msg:
                continue
            expected = expected_msg["content"]

            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            try:
                prompt = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False,
                    add_generation_prompt=True, enable_thinking=cfg.model.enable_thinking)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=400,
                                     temperature=0.3, do_sample=True, top_p=0.9)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            del inputs, out

            has_tool_expected = "<tool_call>" in expected
            has_tool_generated = "<tool_call>" in resp

            if has_tool_expected:
                tool_match = re.search(r'"name"\s*:\s*"(\w+)"', expected)
                gen_match = re.search(r'"name"\s*:\s*"(\w+)"', resp)
                if tool_match and gen_match and tool_match.group(1) == gen_match.group(1):
                    correct += 1
                elif not has_tool_generated and not has_tool_expected:
                    correct += 1
            else:
                if not has_tool_generated:
                    correct += 1

            total += 1

        torch.cuda.empty_cache()
        if total == 0:
            return None
        return round(correct / total * 100, 1)
    except Exception:
        return None


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

    warmup_steps = min(max(1, int(total_steps * tc.warmup_ratio)), total_steps // 3)
    eval_steps = max(1, min(total_steps // 4, 30))
    log_every = max(1, min(tc.logging_steps, total_steps // 3))

    print(f"[SCHEDULE] steps/epoch={steps_per_epoch} total={total_steps} "
          f"log_every={log_every} eval_every={eval_steps} warmup={warmup_steps}")

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
        logging_steps=log_every,
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

    # ── Auto-eval on validation set ──
    import math
    eval_metrics = {}
    try:
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss")
        if eval_loss is not None:
            eval_metrics["eval_loss"] = round(eval_loss, 4)
            eval_metrics["perplexity"] = round(math.exp(min(eval_loss, 20)), 2)
    except Exception:
        pass

    # Quick generation accuracy: generate on val examples and check quality
    val_accuracy = _compute_generation_accuracy(model, tokenizer, cfg, val_ds)
    if val_accuracy is not None:
        eval_metrics["val_accuracy_pct"] = val_accuracy

    summary = logger.get_summary()
    summary.update({
        "adapter_path": final_dir,
        "model_name": cfg.model.model_name,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "token_stats": token_stats,
        "trainable_params": trainable,
        "total_params": total_params,
        **eval_metrics,
    })

    if on_complete:
        on_complete(summary)

    return summary
