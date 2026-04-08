"""
Config loader — YAML to typed dataclasses.

Reads model YAMLs from configs/models/, merges with defaults.yaml,
and produces ModelConfig / TrainingConfig / EvalConfig objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
MODELS_DIR = CONFIGS_DIR / "models"


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class ModelConfig:
    model_name: str = ""
    display_name: str = ""
    family: str = ""
    size: str = ""
    quantization: str = "4bit"
    compute_dtype: str = "bfloat16"
    max_seq_length: int = 1536
    enable_thinking: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    save_total_limit: int = 2
    eval_steps_fraction: float = 0.1
    logging_steps: int = 5
    seed: int = 42


@dataclass
class EvalConfig:
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 800
    repetition_penalty: float = 1.1


@dataclass
class PipelineConfig:
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvalConfig
    output_dir: str = ""
    experiment_name: str = "default"
    adapter_path: Optional[str] = None      # previous adapter to stack on
    train_file: str = ""
    val_file: str = ""
    test_file: str = ""
    # CSV column mapping (only used when dataset is CSV)
    input_col: str = ""
    output_col: str = ""
    system_prompt: str = ""


# ── Loaders ──────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_available_models() -> List[dict]:
    """Return list of {file, display_name, model_name} for every YAML in configs/models/."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for p in sorted(MODELS_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            data = load_yaml(p)
            models.append({
                "file": p.name,
                "display_name": data.get("display_name", p.stem),
                "model_name": data.get("model_name", ""),
                "family": data.get("family", ""),
                "size": data.get("size", ""),
            })
        except Exception:
            continue
    return models


def load_model_config(model_file: str) -> ModelConfig:
    """Load a model YAML and return ModelConfig."""
    path = MODELS_DIR / model_file
    data = load_yaml(path)
    lora_data = data.pop("lora", {})
    lora = LoraConfig(**lora_data)
    return ModelConfig(lora=lora, **data)


def build_custom_model_config(
    hf_model_id: str,
    quantization: str = "4bit",
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_seq_length: int = 2048,
    enable_thinking: bool = False,
) -> ModelConfig:
    """
    Build a ModelConfig from a raw HuggingFace model ID.

    Use this when the user types a custom model instead of picking a preset.
    Applies sensible defaults for LoRA target modules (works for most
    Llama / Qwen / Mistral / Gemma / Phi architectures).
    """
    parts = hf_model_id.split("/")
    short_name = parts[-1] if parts else hf_model_id
    return ModelConfig(
        model_name=hf_model_id,
        display_name=short_name,
        family="custom",
        size="unknown",
        quantization=quantization,
        compute_dtype="bfloat16",
        max_seq_length=max_seq_length,
        enable_thinking=enable_thinking,
        lora=LoraConfig(r=lora_r, alpha=lora_alpha),
    )


def load_defaults() -> dict:
    """Load defaults.yaml and return raw dict."""
    return load_yaml(CONFIGS_DIR / "defaults.yaml")


def build_pipeline_config(
    model_file: str,
    train_file: str,
    val_file: str = "",
    test_file: str = "",
    output_dir: str = "",
    experiment_name: str = "default",
    adapter_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> PipelineConfig:
    """
    Assemble a full PipelineConfig from a model YAML + defaults + overrides.

    Parameters
    ----------
    model_file : name of YAML in configs/models/  (e.g. "qwen3_4b.yaml")
    overrides  : flat dict of overrides like {"training.epochs": 5, "training.learning_rate": 1e-4}
    """
    model_cfg = load_model_config(model_file)
    defaults = load_defaults()

    train_data = defaults.get("training", {})
    eval_data = defaults.get("evaluation", {})

    if overrides:
        for key, val in overrides.items():
            parts = key.split(".")
            if len(parts) == 2:
                section, param = parts
                if section == "training":
                    train_data[param] = val
                elif section == "evaluation":
                    eval_data[param] = val
                elif section == "model":
                    setattr(model_cfg, param, val)
                elif section == "lora":
                    setattr(model_cfg.lora, param, val)

    training_cfg = TrainingConfig(**train_data)
    eval_cfg = EvalConfig(**eval_data)

    if not output_dir:
        output_dir = str(ROOT / "outputs" / experiment_name)

    return PipelineConfig(
        model=model_cfg,
        training=training_cfg,
        evaluation=eval_cfg,
        output_dir=output_dir,
        experiment_name=experiment_name,
        adapter_path=adapter_path,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )
