# Getting Started — MLOps Fine-Tuning Hub

> No installation needed. Just open a URL.

---

## Access the App

Open your browser and go to:

```
http://<EC2-PUBLIC-IP>:8501
```

> Ask your DevOps person for the exact URL. It looks like `http://54.123.45.67:8501`

That's it. You're in.

---

## What You Can Do

The app has 4 tabs:

| Tab | What it does |
|---|---|
| **Train** | Upload your dataset, pick a model, set hyperparameters, run fine-tuning |
| **Evaluate** | Run test cases against a trained adapter, see accuracy and latency |
| **Chat** | Load any trained adapter and have a live conversation with it |
| **Experiments** | View all past training runs, compare loss curves, metrics, parameters |

---

## Tab 1: Train a Model

**Step 1 — Pick a model**
Choose from the preset list (Qwen 3-4B, Llama 3-8B, etc.) or paste any HuggingFace model ID.

**Step 2 — Upload your dataset**
Supported formats:
- **JSONL** — each line is `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- **CSV** — any CSV with an input column and an output column (you pick which ones in the UI)

The app validates your file and tells you how many examples it found.

**Step 3 — Adapter stacking (optional)**
If you want to build on a previously trained adapter instead of starting from scratch, check this box and select the base adapter.

**Step 4 — Set hyperparameters**
Sensible defaults are pre-filled. You can adjust:
- Epochs, learning rate, gradient accumulation
- LoRA rank and alpha, quantization (4-bit recommended)

**Step 5 — Upload test cases (optional)**
If you have a test JSONL file, the app will auto-evaluate accuracy right after training finishes.

**Step 6 — Launch Training**
Click **Launch Training**. A live loss curve appears as training runs on the GPU. You'll see train loss, eval loss, and GPU usage update in real time.

When done, results are saved automatically and visible in the Experiments tab.

---

## Tab 2: Evaluate

1. Select a trained adapter from the dropdown
2. Upload a test cases JSONL file
3. Optionally add a system prompt
4. Click **Run Evaluation**

You'll get accuracy, latency, tokens/sec, and per-scenario results.

**Test case format:**
```json
{
  "name": "Basic booking",
  "user_msg": "Book a meeting for 3pm tomorrow",
  "expected": {
    "should_call_tool": true,
    "tool_name": "book_appointment",
    "key_params": ["date", "time_slot"]
  }
}
```

---

## Tab 3: Chat

1. Select a trained adapter from the dropdown
2. Click **Load Model** (takes ~30 seconds)
3. Type your message and press Enter

The model responds live. If it was trained with thinking enabled, you'll see a collapsible reasoning block above each response.

---

## Tab 4: Experiments

Shows all past training runs in a table with:
- Model name, quantization, LoRA settings
- Train loss, eval loss, perplexity, accuracy
- Training duration, number of trainable parameters
- Dataset used

Click any run to expand it and see the full loss curve and all logged parameters.

---

## Dataset Format Reference

### JSONL (recommended)
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]}
{"messages": [
  {"role": "user", "content": "Book a meeting at 3pm"},
  {"role": "assistant", "content": "<tool_call>{\"name\": \"book_meeting\", \"arguments\": {\"time\": \"15:00\"}}</tool_call>"}
]}
```
One example per line. System message is optional.

### CSV
```csv
input,output
"What is the capital of France?","Paris."
"Book a meeting at 3pm","<tool_call>...</tool_call>"
```
Any column names work — you map them in the UI.

---

## Need Help?

Contact your DevOps person or the team lead who set up the hub.
