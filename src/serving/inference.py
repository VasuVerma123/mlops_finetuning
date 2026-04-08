"""
Inference engine — load model + adapter and generate responses.

Used by both the Streamlit chat interface and the CLI serve command.
"""

from __future__ import annotations

import gc
import time
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import PipelineConfig


class InferenceEngine:
    """Loads a model + LoRA adapter and generates responses."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.cfg: Optional[PipelineConfig] = None
        self._loaded_adapter: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self, cfg: PipelineConfig, adapter_path: str):
        """Load base model with quantization and apply LoRA adapter."""
        if self.is_loaded and self._loaded_adapter == adapter_path:
            return  # already loaded

        self.unload()
        self.cfg = cfg

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

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name, trust_remote_code=True, padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._loaded_adapter = adapter_path

        gc.collect()
        torch.cuda.empty_cache()

    def unload(self):
        """Free model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded_adapter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_new_tokens: int = 800,
        repetition_penalty: float = 1.1,
    ) -> dict:
        """
        Generate a response from a list of chat messages.

        Returns dict with: response, latency_s, tokens_generated, tokens_per_sec
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.cfg and self.cfg.model.enable_thinking:
            template_kwargs["enable_thinking"] = True

        try:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
            )
        latency = time.perf_counter() - t0

        new_tokens = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens_generated = len(new_tokens)

        del inputs, output_ids
        torch.cuda.empty_cache()

        return {
            "response": response,
            "latency_s": round(latency, 3),
            "tokens_generated": tokens_generated,
            "tokens_per_sec": round(tokens_generated / latency, 1) if latency > 0 else 0,
        }
