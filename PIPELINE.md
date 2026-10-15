# Echo v4.7 — Ollama Deployment Pipeline

End-to-end guide for fine-tuning, converting, deploying, and sharing Echo as an Ollama model.

> Start at [README.md](README.md) for what Echo is and how it scores.
> Benchmark detail and training recipes: [RESULTS.md](RESULTS.md).

---

## What Is Echo?

Echo is a fine-tuned AI scheduling assistant for Dr. Maya Chen (Licensed Therapist).

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B` |
| Fine-tuning method | LoRA (PEFT) via SFT |
| Training platform | AWS SageMaker |
| Adapter stack | V4.6 → V4.7 (stacked) |
| GGUF quantization | Q8_0 |
| GGUF size | ~4.3 GB |
| Context length | 4096 tokens |
| Temperature | 0.3 |
| Repeat penalty | 1.1 |
| Accuracy | 98% (47/48 checks) — see [RESULTS.md](RESULTS.md) |
| Ollama Hub | [vasuverma/echo-v4.7](https://ollama.com/vasuverma/echo-v4.7) |

---

## For Friends — Run Echo in 3 Steps

> No Python fine-tuning needed. Just install Ollama, pull the model, and run.

### Step 1 — Install Ollama

Download and install from: **https://ollama.com/download**
- Windows: run the `.exe` installer
- Mac: run the `.dmg` installer
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`

Verify it works:
```bash
ollama --version
```

---

### Step 2 — Pull the Echo model (~4.3 GB download)

```bash
ollama pull vasuverma/echo-v4.7
```

Verify it downloaded:
```bash
ollama list
# Should show: vasuverma/echo-v4.7   ...   4.3 GB
```

---

### Step 3 — Chat with Echo

**Option A — Terminal chat (built-in Ollama CLI)**
```bash
ollama run vasuverma/echo-v4.7
```

**Option B — Python test script (recommended)**

Install the Python dependency:
```bash
pip install ollama colorama
```

Run the test script:
```bash
python Echo_finetune/ollama_test.py
```

The script:
- Auto-detects if the model is installed and offers to download it
- Shows 8 numbered quick-test prompts you can fire by typing a number
- Displays tool calls (e.g. `book_appointment`) clearly formatted
- Supports multi-turn conversation with history
- Type `clear` to reset, `exit` to quit

---

### Sample Prompts to Test

| # | Scenario | Prompt |
|---|---|---|
| 1 | Book appointment | `Book John at 10am on 2026-04-01 for therapy` |
| 2 | Working hours | `What are Dr. Chen's working hours?` |
| 3 | Cancel appointment | `Cancel Sarah's 2pm appointment on 2026-04-02` |
| 4 | Check availability | `Check availability for 2026-04-05` |
| 5 | Reschedule | `Reschedule Mike from 9am to 11am on 2026-04-03` |
| 6 | Send message | `Send a reminder to David about his session tomorrow` |
| 7 | Policy question | `What is the cancellation policy?` |
| 8 | Emergency booking | `Book an emergency session for Lisa today at 3pm` |

---

### Expected Model Behaviour

When you ask Echo to book/cancel/reschedule, it will respond with:
1. A reasoning block inside `<think>...</think>` (internal logic, can ignore)
2. A tool call in JSON inside `<tool_call>...</tool_call>`, e.g.:

```json
<tool_call>
{
  "function": "book_appointment",
  "arguments": {
    "date": "2026-04-01",
    "time_slot": "10:00 AM",
    "client_name": "John",
    "service": "therapy"
  }
}
```

For non-action questions (hours, policy, greeting), Echo replies in plain text.

---

## Files in This Repository

| File | Purpose |
|---|---|
| `aws_finetune_v4_6.py` | Training script for V4.6 (self-contained: generates its own dataset) |
| `aws_finetune_v4_7.py` | Training script for V4.7 — surgical fix for Reschedule + Distractor |
| `aws_eval_v4_7.py` | The 8-scenario benchmark that produces the 98% figure |
| `Echo_finetune/prepare_ollama_v47.py` | Merges V4.6 + V4.7 LoRA adapters into Qwen3-4B, converts to GGUF, registers in Ollama |
| `Echo_finetune/ollama_test.py` | Ready-to-share test script — talks to the published model, no GPU needed |
| `Echo_finetune/streamlit_echo.py` | Full Streamlit UI with calendar and tool execution (local adapters) |
| `Echo_finetune/interactive_agent_v47.py` | Command-line interactive agent (local adapters, no Ollama) |
| `Echo_finetune/echo_logger.py` | Logging module for `streamlit_echo.py` |
| `Echo_finetune/output/Modelfile_v4.7` | Ollama config — GGUF path, chat template, system prompt, parameters |
| `Echo_finetune/output/echo-v4.7-merged/` | Full merged model, HuggingFace safetensors (~8 GB) — **gitignored** |
| `Echo_finetune/output/echo_v4.7_q8.gguf` | GGUF model file (~4.3 GB, Q8_0), what Ollama loads — **gitignored** |

---

## Full Deployment Pipeline (For Developers)

### Prerequisites

```bash
# Install Python deps (in project venv)
uv pip install transformers peft torch accelerate --python .venv/Scripts/python.exe

# Install Ollama (see above)
```

---

### Step 1 — Merge V4.6 + V4.7 Adapters into Base Model

**File:** `Echo_finetune/prepare_ollama_v47.py`

The fine-tuned model is saved as two stacked LoRA adapters. This step merges both into the base `Qwen3-4B` to produce a single standalone model.

```
Qwen/Qwen3-4B base
    → merge V4.6 adapter → intermediate merged model
    → merge V4.7 adapter → final merged model (output/echo-v4.7-merged/)
```

```bash
cd Echo_finetune
python prepare_ollama_v47.py
```

**Output:** `output/echo-v4.7-merged/` (~8 GB)

> This takes ~1-2 hours. Skip if `output/echo-v4.7-merged/` already exists.

---

### Step 2 — Convert Merged Model to GGUF

**Tool:** `llama.cpp/convert_hf_to_gguf.py`

Ollama only reads GGUF format. This step converts the HuggingFace safetensors to GGUF.

```bash
# Clone llama.cpp (one-time setup)
git clone https://github.com/ggerganov/llama.cpp --depth 1

# Install conversion requirements
pip install numpy sentencepiece transformers gguf protobuf

# Convert (Q8_0 = high quality, ~4.3 GB)
python llama.cpp/convert_hf_to_gguf.py output/echo-v4.7-merged \
    --outfile output/echo_v4.7_q8.gguf \
    --outtype q8_0
```

**Output:** `output/echo_v4.7_q8.gguf` (~4.3 GB)

---

### Step 3 — Write Modelfile

**File:** `Echo_finetune/output/Modelfile_v4.7`

The Modelfile configures Ollama with:
- **`FROM`** — path to the GGUF file
- **`TEMPLATE`** — Qwen3 chat template (see critical note below)
- **`SYSTEM`** — Echo's full persona + tool definitions
- **`PARAMETER`** — temperature, top_p, stop tokens

```
FROM ./echo_v4.7_q8.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are Echo..."""

PARAMETER temperature 0.3
PARAMETER stop "<|im_end|>"
PARAMETER stop "</tool_call>"
```

> **CRITICAL — Qwen3 Chat Template**
> Qwen3 models require the `TEMPLATE` block above. Without it, Ollama defaults to
> `{{ .Prompt }}` (raw text), the chat tokens are never applied, and the model enters an
> **infinite thinking loop** — hanging every request forever. Always include the
> `TEMPLATE` block when creating Modelfiles for Qwen3-based models.

---

### Step 4 — Register with Local Ollama

```bash
ollama create echo-v4.7 -f Echo_finetune/output/Modelfile_v4.7
```

Verify registration:
```bash
ollama list
# echo-v4.7   ...   4.3 GB
```

Test locally:
```bash
ollama run echo-v4.7
```

---

### Step 5 — Push to Ollama Hub

```bash
# 1. Create a free account at https://ollama.com
# 2. Get your API key: https://ollama.com/settings/keys

# 3. Copy model under your username
ollama cp echo-v4.7 vasuverma/echo-v4.7

# 4. Push (Windows PowerShell)
$env:OLLAMA_API_KEY="your_key_here"
ollama push vasuverma/echo-v4.7

# 4. Push (Mac/Linux)
OLLAMA_API_KEY="your_key_here" ollama push vasuverma/echo-v4.7
```

**Live at:** https://ollama.com/vasuverma/echo-v4.7

---

### Step 6 — Share `ollama_test.py`

Send friends the `Echo_finetune/ollama_test.py` file. They only need:

```bash
pip install ollama colorama
python ollama_test.py
```

The script auto-offers to pull the model on first run.

---

## Full Flow Diagram

```
AWS SageMaker fine-tuning
    ├── output/scheduling-agent-v4.6-aws/final_v4.6/   (V4.6 LoRA adapter, ~126 MB)
    └── output/scheduling-agent-v4.7-aws/final_v4.7/   (V4.7 LoRA adapter, ~126 MB)
              │
              │ prepare_ollama_v47.py
              │  1. Load Qwen/Qwen3-4B base
              │  2. Apply + merge V4.6 adapter
              │  3. Apply + merge V4.7 adapter on top
              ▼
    output/echo-v4.7-merged/                            (Full model, ~8 GB safetensors)
              │
              │ llama.cpp/convert_hf_to_gguf.py
              ▼
    output/echo_v4.7_q8.gguf                            (GGUF Q8_0, ~4.3 GB)
              │
              │ ollama create echo-v4.7 -f Echo_finetune/output/Modelfile_v4.7
              ▼
    Local Ollama  →  echo-v4.7
              │
              │ ollama cp + ollama push
              ▼
    Ollama Hub  →  vasuverma/echo-v4.7  (public)
              │
              │ friends run:  ollama pull vasuverma/echo-v4.7
              ▼
    Friends  →  python Echo_finetune/ollama_test.py  →  chat with Echo
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Model hangs on every request | Missing `TEMPLATE` in Modelfile | Add the Qwen3 `TEMPLATE` block (see Step 3 above) |
| `ollama: command not found` | Ollama not installed or not in PATH | Re-install from https://ollama.com/download |
| `connection refused` on port 11434 | Ollama server not running | Run `ollama serve` in a separate terminal |
| Model not found locally | Not pulled yet | Run `ollama pull vasuverma/echo-v4.7` |
| `ResponseError` from ollama SDK | Wrong model name in script | Make sure `MODEL = "vasuverma/echo-v4.7"` |
| VRAM out of memory | Q8_0 needs ~5 GB VRAM | Use CPU: `OLLAMA_NUM_GPU=0 ollama run vasuverma/echo-v4.7` |
| Slow responses (>30s) | Running on CPU | Normal on CPU — GPU greatly speeds it up |
