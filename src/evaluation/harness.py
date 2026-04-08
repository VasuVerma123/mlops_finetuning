"""
Evaluation harness — load model + adapter, run test scenarios, collect
scored results with latency metrics.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import PipelineConfig
from ..data.loader import load_jsonl
from .scorer import score_response
from .metrics import compute_scenario_metrics, format_comparison_table


def _load_model(cfg: PipelineConfig, adapter_path: str):
    """Load base model + adapter for evaluation."""
    quant = cfg.model.quantization
    bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if bf16_ok else torch.float16

    bnb_config = None
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if bnb_config:
        kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer


def _generate(model, tokenizer, system_prompt: str, user_msg: str, cfg: PipelineConfig):
    """Generate a response and measure latency."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if cfg.model.enable_thinking:
        template_kwargs["enable_thinking"] = True

    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    ev = cfg.evaluation
    gen_kwargs = dict(
        max_new_tokens=ev.max_new_tokens,
        temperature=ev.temperature,
        top_p=ev.top_p,
        do_sample=True,
        repetition_penalty=ev.repetition_penalty,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    total_time = time.perf_counter() - t0

    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tokens_generated = len(new_tokens)

    del inputs, output_ids
    torch.cuda.empty_cache()

    return response, total_time, tokens_generated


def run_evaluation(
    cfg: PipelineConfig,
    adapter_path: str,
    test_file: Optional[str] = None,
    system_prompt: str = "",
    on_progress: Optional[callable] = None,
) -> dict:
    """
    Run the full evaluation harness.

    Parameters
    ----------
    test_file : path to JSONL with test cases. Each line must have:
                name, user_msg, expected (dict).
    system_prompt : system prompt to use for all tests.

    Returns
    -------
    dict with overall metrics + per-scenario breakdown.
    """
    test_path = test_file or cfg.test_file
    if not test_path or not Path(test_path).exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    tests = load_jsonl(test_path)
    model, tokenizer = _load_model(cfg, adapter_path)

    scenario_results = []
    for i, test in enumerate(tests):
        name = test.get("name", f"Test {i+1}")
        user_msg = test["user_msg"]
        expected = test["expected"]

        if on_progress:
            on_progress({"step": i + 1, "total": len(tests), "name": name})

        msg = user_msg
        if "schedule_state" in test:
            msg = f"{user_msg}\n\n{test['schedule_state']}"

        prompt_system = test.get("system_prompt", system_prompt)
        response, latency, tokens = _generate(
            model, tokenizer, prompt_system, msg, cfg,
        )
        scores = score_response(response, expected)

        scenario_results.append({
            "name": name,
            "scores": scores,
            "latency_s": latency,
            "tokens_generated": tokens,
            "time_to_first_token_s": 0,  # approximated as total/tokens for now
            "response": response,
        })

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    metrics = compute_scenario_metrics(scenario_results)
    metrics["raw_results"] = scenario_results

    # Save results
    results_path = os.path.join(cfg.output_dir, "eval_results.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics
