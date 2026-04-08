# MLOps Fine-Tuning Pipeline

End-to-end, config-driven pipeline for fine-tuning language models with QLoRA.  
Built for the Echo scheduling assistant but fully reusable for any SFT task.

## Architecture

```
mlops_finetuning/
├── configs/
│   ├── models/           # Model-specific YAML configs
│   │   ├── qwen3_4b.yaml
│   │   ├── qwen3_8b.yaml
│   │   ├── qwen2.5_7b.yaml
│   │   └── _template.yaml
│   └── defaults.yaml     # Default training hyperparameters
├── src/
│   ├── config.py          # YAML → typed dataclasses
│   ├── data/
│   │   ├── loader.py      # JSONL loading, train/val split, tokenization
│   │   └── validator.py   # Schema validation + dataset stats
│   ├── training/
│   │   ├── trainer.py     # Core QLoRA fine-tuning eRA
│   │   └── callbacks.py   # Loss logging + progress callbacks
│   ├── evaluation/
│   │   ├── harness.py     # Run test scenarios, collect results
│   │   ├── scorer.py      # Score responses (tool correctness, params, etc.)
│   │   └── metrics.py     # Compute accuracy, latency, per-scenario breakdown
│   ├── tracking/
│   │   └── mlflow_tracker.py  # MLflow experiment logging
│   └── serving/
│       └── inference.py   # Load model + adapter, generate responses
├── app.py                 # Streamlit UI (3 tabs: Train, Evaluate, Experiments)
├── run.py                 # CLI entry point
├── outputs/               # Training outputs (adapters, logs, results)
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (for training/inference)
- 16GB+ VRAM recommended (24GB for 8B models)

### Installation

```bash
cd mlops_finetuning
pip install -r requirements.txt
```

## Usage

### Option 1: Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens a web interface with three tabs:

| Tab | Description |
|-----|-------------|
| **Train** | Select model, upload dataset, configure hyperparameters, launch training with live loss curves |
| **Evaluate** | Load adapter, upload test cases, run evaluation with accuracy/latency dashboard |
| **Experiments** | Browse MLflow experiment history, compare runs side by side |

### Option 2: CLI

```bash
# List available models
python run.py models

# Validate a dataset
python run.py validate --file my_data.jsonl

# Train a model
python run.py train \
  --model qwen3_4b.yaml \
  --train train.jsonl \
  --val val.jsonl \
  --name my_experiment \
  --epochs 3

# Train with adapter stacking (build on a previous fine-tune)
python run.py train \
  --model qwen3_4b.yaml \
  --train targeted_data.jsonl \
  --name v2_experiment \
  --adapter outputs/my_experiment/final_adapter

# Evaluate
python run.py eval \
  --model qwen3_4b.yaml \
  --adapter outputs/my_experiment/final_adapter \
  --test test_cases.jsonl

# Interactive chat
python run.py serve \
  --model qwen3_4b.yaml \
  --adapter outputs/my_experiment/final_adapter
```

## Dataset Format

### Training Data (JSONL)

Each line is a JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Book a meeting at 3 PM"},
    {"role": "assistant", "content": "<think>reasoning...</think>\n<tool_call>{\"name\": \"book_appointment\", \"arguments\": {...}}</tool_call>"}
  ]
}
```

### Test Cases (JSONL)

Each line defines a test scenario:

```json
{
  "name": "Direct Booking",
  "user_msg": "Book a session at 3 PM on March 24th",
  "system_prompt": "You are Echo...",
  "expected": {
    "should_call_tool": true,
    "tool_name": "book_appointment",
    "key_params": ["date", "time_slot", "client_name"],
    "should_not_call": ["get_weather"],
    "extra_check": null
  }
}
```

### Supported `extra_check` values

| Check | What it verifies |
|-------|-----------------|
| `conflict_handling` | Model detects schedule conflicts and suggests alternatives |
| `privacy` | Model does not leak other clients' names |
| `distractor` | Model declines off-topic requests |

## Adding a New Model

1. Copy `configs/models/_template.yaml`
2. Fill in the HuggingFace model ID, LoRA settings, etc.
3. Save as `configs/models/my_model.yaml`
4. The model will automatically appear in both the UI and CLI

## Adapter Stacking

Train iteratively by building on previous adapters:

1. **V1**: Train from scratch → produces `outputs/v1/final_adapter`
2. **V2**: Train with `--adapter outputs/v1/final_adapter` → merges V1 into base, applies fresh LoRA
3. **V3**: Train with `--adapter outputs/v2/final_adapter` → builds on V2

This preserves previous learning while fine-tuning on new or targeted data.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of checks passed across all scenarios |
| **Latency** | End-to-end generation time per scenario |
| **Tokens/sec** | Throughput (tokens generated / time) |
| **TTFT** | Time to first token |
| **Per-scenario** | Breakdown with individual check pass/fail |

## MLflow Tracking

All training and evaluation runs are logged to MLflow:

```bash
# Launch MLflow UI
mlflow ui --port 5000
```

Logged data includes: hyperparameters, loss curves, evaluation metrics, and adapter artifacts.

## Configuration Reference

### defaults.yaml

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 3 | Number of training epochs |
| `batch_size` | 1 | Per-device batch size |
| `gradient_accumulation` | 8 | Effective batch = batch_size x grad_accum |
| `learning_rate` | 2e-4 | Peak learning rate |
| `scheduler` | cosine | LR scheduler type |
| `optimizer` | paged_adamw_8bit | Memory-efficient optimizer |
| `save_total_limit` | 2 | Max checkpoints to keep |

### Model YAML

| Field | Description |
|-------|-------------|
| `model_name` | HuggingFace model ID |
| `quantization` | `4bit`, `8bit`, or `none` |
| `lora.r` | LoRA rank (capacity vs VRAM) |
| `lora.alpha` | LoRA scaling (typically 2x rank) |
| `enable_thinking` | Enable `<think>` reasoning tags |
