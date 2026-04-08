# MLOps Fine-Tuning Hub — Complete Pipeline Documentation

## Directory Structure

```
mlops_finetuning/
├── app.py                          # Streamlit UI (main entry point)
├── run.py                          # CLI entry point (train/eval/serve/models/validate)
├── requirements.txt                # Python dependencies
├── mlflow.db                       # SQLite database for MLflow experiment tracking
├── PIPELINE.md                     # This file
├── README.md                       # Setup and usage guide
│
├── .streamlit/
│   └── config.toml                 # Streamlit dark theme config
│
├── configs/
│   ├── defaults.yaml               # Default hyperparameters (epochs, lr, batch_size, etc.)
│   └── models/
│       ├── _template.yaml          # Template for adding custom models
│       ├── qwen3_4b.yaml           # Qwen 3 - 4B preset
│       ├── qwen3_8b.yaml           # Qwen 3 - 8B preset
│       ├── qwen2.5_7b.yaml         # Qwen 2.5 - 7B Instruct preset
│       ├── llama3_8b.yaml          # Llama 3 - 8B Instruct preset
│       ├── llama3.1_8b.yaml        # Llama 3.1 - 8B Instruct preset
│       └── llama3.2_3b.yaml        # Llama 3.2 - 3B Instruct preset
│
├── sample_data/
│   ├── train.jsonl                 # 8-example dummy training data
│   └── val.jsonl                   # 4-example dummy validation data
│
├── src/
│   ├── config.py                   # YAML → typed dataclasses (ModelConfig, TrainingConfig, etc.)
│   ├── data/
│   │   ├── loader.py               # Dataset ingestion (JSONL/JSON/CSV), tokenization, splitting
│   │   └── validator.py            # Schema validation before training
│   ├── training/
│   │   ├── trainer.py              # Core QLoRA fine-tuning engine (run_training)
│   │   ├── adapter_stack.py        # Merge previous adapter + apply fresh LoRA
│   │   └── callbacks.py            # LossLogger TrainerCallback for step-level logging
│   ├── evaluation/
│   │   ├── harness.py              # Evaluation orchestrator (run_evaluation)
│   │   ├── scorer.py               # Response scoring (tool calls, privacy, distractors)
│   │   └── metrics.py              # Accuracy, latency, per-scenario breakdown
│   ├── serving/
│   │   └── inference.py            # InferenceEngine class for chat
│   └── tracking/
│       └── mlflow_tracker.py       # MLflow helper class (used by CLI)
│
└── outputs/                        # Training outputs (gitignored)
    └── <experiment_name>/
        ├── data/                   # Uploaded dataset copies
        ├── final_adapter/          # Saved LoRA adapter weights
        ├── checkpoint-*/           # Intermediate checkpoints
        └── loss_log.json           # Step-by-step loss history
```

---

## End-to-End Pipeline Flow

### Phase 1: Configuration

**What happens:** User picks a model, uploads a dataset, sets hyperparameters.

**Where it happens:**

| Step | File | Function/Section | What It Does |
|------|------|-----------------|--------------|
| Model selection | `app.py` L46-52 | `PRESET_MODELS` dict | Maps HuggingFace IDs to display names |
| Custom model | `src/config.py` L140-167 | `build_custom_model_config()` | Creates a `ModelConfig` dataclass from a raw HF model ID with sensible LoRA defaults |
| Load YAML preset | `src/config.py` L131-137 | `load_model_config()` | Reads `configs/models/*.yaml`, returns `ModelConfig` |
| Default hyperparams | `src/config.py` L170-172 | `load_defaults()` | Reads `configs/defaults.yaml` → raw dict |
| Assemble full config | `src/config.py` L175-229 | `build_pipeline_config()` | Merges model YAML + defaults + overrides → `PipelineConfig` |

**Dataclasses (src/config.py):**

```
PipelineConfig
├── model: ModelConfig
│   ├── model_name        "Qwen/Qwen3-4B"
│   ├── quantization      "4bit" | "8bit" | "none"
│   ├── max_seq_length    1536
│   ├── enable_thinking   True/False
│   └── lora: LoraConfig
│       ├── r             16
│       ├── alpha          32
│       ├── dropout        0.05
│       └── target_modules ["q_proj", "k_proj", "v_proj", ...]
├── training: TrainingConfig
│   ├── epochs            3
│   ├── batch_size        1
│   ├── gradient_accumulation  8
│   ├── learning_rate     2e-4
│   ├── scheduler         "cosine"
│   └── optimizer         "paged_adamw_8bit"
├── evaluation: EvalConfig
│   ├── temperature       0.3
│   └── max_new_tokens    800
├── output_dir            "outputs/exp_20260327_1753"
├── train_file            "outputs/.../data/train.jsonl"
├── val_file              "outputs/.../data/val.jsonl"
└── adapter_path          null (or path to previous adapter for stacking)
```

---

### Phase 2: Data Validation & Loading

**What happens:** Uploaded file is validated for correct format, then loaded and tokenized.

#### 2a. Validation (before training starts)

| Format | File | Function | Checks |
|--------|------|----------|--------|
| JSONL/JSON | `src/data/validator.py` L115-170 | `validate_file()` | Each record has `messages` list with ≥2 entries, roles are valid, assistant message exists |
| CSV | `src/data/validator.py` L49-112 | `validate_csv()` | Input/output columns exist, no empty rows |
| Test cases | `src/data/validator.py` L173-198 | `validate_test_cases()` | Each record has `name`, `user_msg`, `expected` with `should_call_tool` |

Statistics collected: role distribution, tool_call count, think-augmented count, avg turns.

#### 2b. Loading & Tokenization

| Step | File | Function | What It Does |
|------|------|----------|--------------|
| Auto-detect format | `src/data/loader.py` L74-83 | `detect_file_type()` | Checks `.jsonl`, `.json`, `.csv` extension |
| Load JSONL | `src/data/loader.py` L21-33 | `load_jsonl()` | Parses each line as JSON → list of `{"messages": [...]}` |
| Load CSV | `src/data/loader.py` L36-71 | `load_csv()` | Reads CSV, maps `input_col` → user message, `output_col` → assistant message, prepends optional system prompt |
| Train/val split | `src/data/loader.py` L116-126 | `train_val_split()` | Random 90/10 split if no separate val file |
| Chat template | `src/data/loader.py` L134-158 | `format_for_training()` | Calls `tokenizer.apply_chat_template()` on each example, appends EOS token, produces `"text"` column |
| Token stats | `src/data/loader.py` L161-177 | `compute_token_stats()` | Samples up to 200 examples, reports avg/max/min/p50/p95 token lengths |
| Full pipeline | `src/data/loader.py` L180-211 | `load_and_prepare()` | Chains all of the above → returns `(train_ds, val_ds, token_stats)` |

**Data transformation example (JSONL):**

```
Input:  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Book at 3pm"}, {"role": "assistant", "content": "<think>...</think>\n<tool_call>..."}]}
  ↓ apply_chat_template()
Output: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\nBook at 3pm<|im_end|>\n<|im_start|>assistant\n<think>...</think>\n<tool_call>...<|im_end|><|endoftext|>"
```

---

### Phase 3: Model Loading & Adapter Stacking

**What happens:** Base model is downloaded from HuggingFace, quantized to 4-bit, previous adapters are merged if stacking, and a fresh LoRA is applied.

| Step | File | Function | What It Does |
|------|------|----------|--------------|
| Quantization config | `src/training/trainer.py` L26-43 | `_get_bnb_config()` | Creates `BitsAndBytesConfig` for 4-bit NF4 or 8-bit quantization |
| Load base model | `src/training/trainer.py` L46-63 | `load_base_model()` | `AutoModelForCausalLM.from_pretrained()` with quantization + `AutoTokenizer` |
| Merge previous adapter | `src/training/adapter_stack.py` L22-31 | `merge_adapter()` | `PeftModel.from_pretrained()` → `merge_and_unload()` — bakes adapter weights into base |
| Apply fresh LoRA | `src/training/adapter_stack.py` L34-51 | `apply_lora()` | `prepare_model_for_kbit_training()` → `get_peft_model()` with LoRA config |
| Full stack pipeline | `src/training/adapter_stack.py` L54-68 | `stack_and_apply()` | If adapter_path given: merge first, then apply fresh LoRA |

**Adapter stacking visualized:**

```
Base Model (Qwen3-4B, 4-bit)
  ↓ merge_adapter(v4.6_adapter)
Base + V4.6 merged weights
  ↓ apply_lora(new LoRA config)
Base + V4.6 (frozen) + New LoRA (trainable ~1.5% of params)
```

**Memory footprint:** With 4-bit quantization, a 4B parameter model uses ~2.5 GB VRAM. LoRA adds ~33M trainable parameters (~1.48% of total).

---

### Phase 4: Training

**What happens:** SFTTrainer runs gradient descent on the tokenized dataset, logging losses at each step.

| Step | File | Function/Line | What It Does |
|------|------|---------------|--------------|
| Compute schedule | `src/training/trainer.py` L170-181 | Inline computation | Calculates `total_steps`, `warmup_steps`, `eval_steps`, `log_every` from dataset size and hyperparams |
| Create SFTConfig | `src/training/trainer.py` L188-218 | `SFTConfig(...)` | Sets all HuggingFace training arguments |
| Create SFTTrainer | `src/training/trainer.py` L220-227 | `SFTTrainer(...)` | Combines model, tokenizer, datasets, callbacks, args |
| Train | `src/training/trainer.py` L233 | `trainer.train()` | Runs the training loop |
| Save adapter | `src/training/trainer.py` L236-238 | `trainer.save_model()` | Saves LoRA weights to `outputs/<exp>/final_adapter/` |

**Training callback — LossLogger (src/training/callbacks.py):**

| Event | Method | What It Logs |
|-------|--------|-------------|
| Training starts | `on_train_begin()` L33 | Records start time |
| Each log step | `on_log()` L36-67 | Train loss, learning rate, eval loss, GPU memory, ETA. Calls `on_log_fn` callback which updates Streamlit charts + MLflow |
| Checkpoint saved | `on_save()` L69 | Dumps `loss_log.json` to disk |
| Training ends | `on_train_end()` L72-88 | Final summary, dumps JSON, notifies callback |

**Live Streamlit updates (app.py):**

The `on_log` callback defined in `app.py` L347-385 receives each log entry and:
1. Logs to MLflow: `mlflow.log_metrics({"train_loss": ..., "learning_rate": ...}, step=N)`
2. Updates a Plotly chart via `chart_slot.plotly_chart()` — live loss curve
3. Updates metric cards via `metrics_slot` — step count, current loss, GPU usage
4. Updates the `st.status()` label with current step and loss

**Training schedule example (8 examples, batch=1, grad_accum=8, epochs=3):**

```
eff_batch = 1 × 8 = 8
steps_per_epoch = max(1, 8 // 8) = 1
total_steps = 1 × 3 = 3
log_every = max(1, min(5, 3 // 3)) = 1      ← logs every step
eval_steps = max(1, min(3 // 4, 30)) = 1     ← evals every step
warmup_steps = min(max(1, 0.05 × 3), 1) = 1
```

---

### Phase 5: Post-Training Auto-Evaluation

**What happens:** After training, the pipeline automatically computes eval loss, perplexity, and generation accuracy.

| Step | File | Function | What It Does |
|------|------|----------|--------------|
| Eval loss | `src/training/trainer.py` L243-250 | `trainer.evaluate()` | Runs the model on val set, gets cross-entropy loss |
| Perplexity | `src/training/trainer.py` L248 | `math.exp(eval_loss)` | Exponential of eval loss — lower = better |
| Generation accuracy | `src/training/trainer.py` L66-130 | `_compute_generation_accuracy()` | Generates responses for up to 10 val examples, checks if tool calls match expected |

**Generation accuracy scoring logic:**

```python
For each validation example:
  1. Extract expected assistant message
  2. Build prompt from system + user messages only
  3. Generate response with model
  4. Compare:
     - If expected has <tool_call>: check if generated tool name matches
     - If expected has NO tool call: check that model didn't hallucinate one
  5. Count correct / total → accuracy %
```

---

### Phase 6: MLflow Experiment Tracking

**What happens:** All parameters, metrics, and artifacts are logged to a local SQLite database.

**Backend:** `sqlite:///C:/.../mlops_finetuning/mlflow.db` (set in `app.py` L43-44)

#### What gets logged (app.py L436-465):

| Category | Key | Value | When |
|----------|-----|-------|------|
| **Parameters** | `model_name` | `"Qwen/Qwen3-4B"` | At run start |
| | `quantization` | `"4bit"` | At run start |
| | `lora_r` | `16` | At run start |
| | `lora_alpha` | `32` | At run start |
| | `epochs` | `3` | At run start |
| | `learning_rate` | `0.0002` | At run start |
| | `gradient_accumulation` | `8` | At run start |
| | `max_seq_length` | `1536` | At run start |
| | `adapter_stacking` | `True/False` | At run start |
| | `train_file` | `"train.jsonl"` | At run start |
| **Step Metrics** | `train_loss` | `2.7170` | Each log step |
| | `learning_rate` | `0.0002` | Each log step |
| | `eval_loss` | `2.6295` | Each eval step |
| **Summary Metrics** | `final_train_loss` | `2.7170` | After training |
| | `best_eval_loss` | `2.6295` | After training |
| | `eval_loss` | `2.6295` | After training |
| | `perplexity` | `13.87` | After training |
| | `accuracy_pct` | `50.0` | After auto-eval |
| | `training_time_s` | `180` | After training |
| | `trainable_params` | `33030144` | After training |
| **Artifacts** | `adapter/` | LoRA weights | After training |
| | `loss_log.json` | Loss history | After training |

#### Optional test harness metrics (if test cases uploaded):

| Key | Value |
|-----|-------|
| `accuracy_pct` | Overall accuracy from harness (overrides auto-eval) |
| `avg_latency_s` | Average generation time per test |
| `avg_tokens_per_sec` | Throughput |
| `ttft_s` | Time to first token |

---

### Phase 7: Evaluation Harness (Optional — Evaluate Tab)

**What happens:** User uploads a test JSONL with scenarios. Each scenario generates a response and is scored against expected behavior.

| Step | File | Function | What It Does |
|------|------|----------|--------------|
| Load model + adapter | `src/evaluation/harness.py` L25-57 | `_load_model()` | Same as training: base + quantization + PeftModel |
| Generate response | `src/evaluation/harness.py` L60-102 | `_generate()` | Builds prompt, generates, measures latency and token count |
| Score response | `src/evaluation/scorer.py` L28-100 | `score_response()` | Checks tool correctness, params, privacy, conflicts, distractors |
| Compute metrics | `src/evaluation/metrics.py` L10-77 | `compute_scenario_metrics()` | Per-scenario and overall accuracy, latency, tokens/sec, TTFT |
| Run all | `src/evaluation/harness.py` L105-174 | `run_evaluation()` | Orchestrates the full loop, saves `eval_results.json` |

**Scoring checks (src/evaluation/scorer.py):**

| Check | What It Verifies |
|-------|-----------------|
| `called_tool` | Model produced a `<tool_call>` block when expected |
| `correct_tool` | The tool name inside `<tool_call>` matches expected |
| `has_params` | All required parameters are present in the arguments |
| `valid_json` | The tool call JSON is parseable |
| `correctly_no_tool` | Model did NOT produce a tool call when none expected |
| `avoided_<tool>` | Model did not call a blacklisted tool |
| `detected_conflict` | Response mentions unavailability/conflict keywords |
| `suggested_alternatives` | Response offers alternative times |
| `kept_privacy` | Response does not leak private client names |
| `declined_off_topic` | Response redirects off-topic requests to scheduling |
| `has_reasoning` | Response contains `<think>` reasoning block |

---

### Phase 8: Chat Interface (Chat Tab)

**What happens:** User selects a trained adapter, loads it, and has a multi-turn conversation.

| Step | File | Location | What It Does |
|------|------|----------|--------------|
| List adapters | `app.py` L88-125 | `find_adapters()` | Scans `outputs/*/final_adapter/` for adapter configs, reads loss from `loss_log.json` |
| Load model | `app.py` L616-652 | Inline in Chat tab | `AutoModelForCausalLM` + `BitsAndBytesConfig` + `PeftModel.from_pretrained()` |
| Build prompt | `app.py` L737-748 | Inline | Concatenates system prompt + full chat history → `apply_chat_template()` |
| Generate | `app.py` L750-757 | Inline | `model.generate()` with temperature, top_p, repetition_penalty |
| Parse response | `app.py` L759-764 | Inline | Extracts `<think>` blocks for expandable reasoning, strips for visible text |
| Display | `app.py` L700-765 | Chat tab | Shows user/assistant messages, expandable reasoning blocks, tool calls |

**Chat flow:**

```
User types message
  ↓
Appended to st.session_state["chat_messages"]
  ↓
System prompt + all messages → tokenizer.apply_chat_template()
  ↓
model.generate(max_new_tokens, temperature, top_p)
  ↓
Decode → extract <think> blocks → strip for display
  ↓
Append assistant message to chat history → st.rerun()
```

---

### Phase 9: Experiments Dashboard (Experiments Tab)

**What happens:** All MLflow runs are displayed in a rich dashboard with tables, charts, and drill-down details.

| Section | File | Location | Data Source |
|---------|------|----------|-------------|
| Overview cards | `app.py` L812-821 | 4-column metrics | Total runs, finished, best loss, best accuracy from `get_mlflow_runs()` |
| All Runs table | `app.py` L824-853 | `st.dataframe()` | Columns: Run, Status, Model, Train Loss, Eval Loss, Perplexity, Accuracy, Latency, Tok/s, TTFT, Duration, Params |
| Train Loss chart | `app.py` L859-869 | Plotly bar chart | Green = best, red = worst |
| Accuracy chart | `app.py` L871-881 | Plotly bar chart | Green = best, red = worst |
| Eval Loss chart | `app.py` L886-895 | Plotly bar chart | Orange bars |
| Latency chart | `app.py` L897-906 | Plotly bar chart | Purple bars |
| Run Details | `app.py` L911-977 | Expandable section | Status badge, metric cards, loss curve from `loss_log.json`, parameters grid, tags |

**Loss curve in Run Details:**

```
Reads: outputs/<experiment_name>/loss_log.json
Format: {"train_losses": [{"step": 1, "loss": 2.71, "lr": 0.0002}, ...],
         "eval_losses": [{"step": 1, "eval_loss": 2.63}, ...]}
Plotly: Train loss line (blue) + Eval loss line (orange) + Trend line (green dashed)
Shows: "Loss drop: 2.7170 → 2.5123 (7.5% reduction over 3 steps)"
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI | Streamlit | Interactive web app with 4 tabs |
| Charts | Plotly | Loss curves, comparison bar charts |
| Training | HuggingFace TRL (`SFTTrainer`) | Supervised fine-tuning |
| Quantization | bitsandbytes | 4-bit NF4 / 8-bit quantization (QLoRA) |
| Adapters | PEFT (LoRA) | Parameter-efficient fine-tuning (~1.5% trainable) |
| Experiment Tracking | MLflow (SQLite backend) | Params, metrics, artifacts, run history |
| Model Hub | HuggingFace | Base model downloads |
| Package Manager | uv | Fast venv creation and pip installs |

---

## Supported Dataset Formats

### JSONL (recommended)

```json
{"messages": [
  {"role": "system", "content": "You are a scheduling assistant."},
  {"role": "user", "content": "Book at 3pm tomorrow."},
  {"role": "assistant", "content": "<think>...</think>\n<tool_call>{\"name\": \"book_appointment\", ...}</tool_call>"}
]}
```

### CSV

```csv
input,output
"Book at 3pm tomorrow","<tool_call>{\"name\": \"book_appointment\", ...}</tool_call>"
```

User maps `input` → user column, `output` → assistant column in the UI.

### Test Cases JSONL (for evaluation)

```json
{
  "name": "Basic booking",
  "user_msg": "Book for John at 3pm on March 28",
  "expected": {
    "should_call_tool": true,
    "tool_name": "book_appointment",
    "key_params": ["date", "time_slot", "client_name"],
    "should_not_call": ["get_weather"]
  }
}
```

---

## Metrics Reference

| Metric | Range | Good Value | Source |
|--------|-------|-----------|--------|
| Train Loss | 0 → ∞ | < 1.0 | Trainer logging |
| Eval Loss | 0 → ∞ | < 1.5 | `trainer.evaluate()` |
| Perplexity | 1 → ∞ | < 5 | `exp(eval_loss)` |
| Accuracy | 0% → 100% | > 80% | Auto-eval or test harness |
| Latency | 0s → ∞ | < 5s | Test harness |
| Tokens/sec | 0 → ∞ | > 20 | Test harness |
| TTFT | 0s → ∞ | < 1s | Test harness |

---

## How to Run

```bash
cd mlops_finetuning
uv venv .venv --python 3.10
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU support
streamlit run app.py
```

**GitHub:** https://github.com/VasuVerma123/mlops_finetuning
