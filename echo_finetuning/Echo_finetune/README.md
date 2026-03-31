# Echo Fine-Tune

Fine-tune **Qwen 2.5 Instruct 7B** for customer service using QLoRA. Includes synthetic data generation, training, and inference — all in three scripts.

## Overview

| Script | Purpose |
|---|---|
| `generate_data.py` | Generate synthetic multi-turn customer service conversations |
| `finetune.py` | Fine-tune Qwen 2.5 7B with QLoRA (or full fine-tune) |
| `inference.py` | Run interactive chat or single-query inference |

The synthetic data covers **5 customer service domains**: billing, technical support, order & shipping, account management, and product inquiries. Each conversation includes realistic multi-turn exchanges with greetings, clarification questions, user responses, resolutions, and closing messages.

## Setup

```bash
# Clone the repository
git clone https://github.com/aabid6396/Echo_finetune.git
cd Echo_finetune

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- CUDA-compatible GPU (minimum ~16 GB VRAM for QLoRA, ~60 GB for full fine-tune)
- Dependencies listed in `requirements.txt`

## Usage

### 1. Generate Training Data

```bash
# Default: 1000 conversations, 90/10 train/val split
python generate_data.py

# Custom settings
python generate_data.py --num-samples 5000 --seed 123 --output data/train.jsonl
```

This produces `data/train.jsonl` and `data/val.jsonl` in Qwen chat-template format. Each sample is a multi-turn conversation with system, user, and assistant messages.

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--num-samples` | `1000` | Total conversations to generate |
| `--output` | `data/train.jsonl` | Output file path |
| `--seed` | `42` | Random seed for reproducibility |

### 2. Fine-Tune the Model

```bash
# QLoRA fine-tuning (default, ~16 GB VRAM)
python finetune.py

# Custom hyperparameters
python finetune.py --epochs 5 --lr 1e-4 --lora-r 64 --lora-alpha 128

# Full fine-tuning (requires ~60 GB VRAM)
python finetune.py --full-finetune

# Merge LoRA weights into base model after training
python finetune.py --merge-and-save
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model-name` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID or local path |
| `--train-data` | `data/train.jsonl` | Training data path |
| `--val-data` | `data/val.jsonl` | Validation data path |
| `--output-dir` | `output/qwen2.5-7b-customer-service` | Output directory |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `1` | Per-device batch size |
| `--grad-accum` | `16` | Gradient accumulation steps |
| `--lr` | `2e-4` | Peak learning rate |
| `--max-seq-length` | `1024` | Max sequence length |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--full-finetune` | `false` | Full fine-tune instead of QLoRA |
| `--merge-and-save` | `false` | Merge LoRA into base model after training |

**Training details:**
- Optimizer: `paged_adamw_8bit` (QLoRA) / `adamw_torch` (full)
- Scheduler: cosine with 5% warmup
- Gradient checkpointing enabled
- Evaluation every 50 steps, checkpoint every 100 steps
- Best model selected by validation loss
- Logging via TensorBoard

### 3. Run Inference

```bash
# Interactive multi-turn chat (default)
python inference.py

# Point to a specific model
python inference.py --model-path output/qwen2.5-7b-customer-service/final

# Use a merged model
python inference.py --model-path output/qwen2.5-7b-customer-service/merged

# LoRA adapter on base model
python inference.py --model-path output/qwen2.5-7b-customer-service/final \
                    --base-model Qwen/Qwen2.5-7B-Instruct

# Single query (non-interactive)
python inference.py --query "I was double charged on my last bill"
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model-path` | `output/qwen2.5-7b-customer-service/final` | Path to fine-tuned model |
| `--base-model` | `None` | Base model ID (only for LoRA adapter mode) |
| `--query` | `None` | Single query instead of interactive chat |

During interactive chat, type `reset` to clear conversation history or `quit` to exit.

## Project Structure

```
.
├── data/
│   ├── train.jsonl          # Training data (generated)
│   └── val.jsonl            # Validation data (generated)
├── output/                  # Training outputs (created during fine-tuning)
│   └── qwen2.5-7b-customer-service/
│       ├── final/           # Fine-tuned model / adapter
│       ├── merged/          # Merged model (if --merge-and-save)
│       └── logs/            # TensorBoard logs
├── generate_data.py         # Synthetic data generator
├── finetune.py              # QLoRA / full fine-tuning script
├── inference.py             # Interactive chat & single-query inference
├── requirements.txt         # Python dependencies
└── README.md
```

## Data Format

Each JSONL entry follows the Qwen chat-template structure:

```json
{
  "id": "uuid",
  "domain": "billing",
  "messages": [
    {"role": "system", "content": "You are a helpful, professional customer service assistant..."},
    {"role": "user", "content": "Hi, I was charged $50 but I didn't make this purchase."},
    {"role": "assistant", "content": "I'm sorry to hear that. Could you provide your account number?"},
    {"role": "user", "content": "Sure, my account email is john@example.com."},
    {"role": "assistant", "content": "I've reviewed your account and issued a refund..."},
    {"role": "user", "content": "Thank you so much!"},
    {"role": "assistant", "content": "You're welcome! Have a great day!"}
  ]
}
```

## License

This project is provided as-is for educational and research purposes.
