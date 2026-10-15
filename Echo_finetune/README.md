# Echo_finetune — Apps & Deployment Tooling

The runnable parts of Echo: three ways to talk to the model, plus the script that turns
the trained adapters into a published Ollama model.

Training and evaluation live one level up ([`../README.md`](../README.md),
[`../RESULTS.md`](../RESULTS.md)). Full deployment walkthrough: [`../PIPELINE.md`](../PIPELINE.md).

---

## Files

| File | What it is | Needs |
|---|---|---|
| `ollama_test.py` | Guided CLI chat against the **published Ollama model**. 8 one-keypress sample scenarios, pretty-printed tool calls, multi-turn history. Offers to pull the model on first run. No GPU needed. | Ollama + `ollama`, `colorama` |
| `streamlit_echo.py` | Full web UI: live calendar on the left, chat in the middle, tools actually execute and mutate the calendar. Loads the **local adapters** in 4-bit. | GPU, adapters, `streamlit` |
| `interactive_agent_v47.py` | CLI agent that loads the **local LoRA adapters** directly through PEFT. Same model as the Streamlit app, no UI. | GPU, adapters, `torch`/`transformers`/`peft` |
| `echo_logger.py` | JSONL session logger used by `streamlit_echo.py`. Writes `logs/echo_<timestamp>.jsonl`, one JSON event per line. | — |
| `prepare_ollama_v47.py` | Merge V4.6 → merge V4.7 → save merged model → convert to GGUF → write Modelfile → create and push to Ollama Hub. | GPU, adapters, llama.cpp |
| `output/Modelfile_v4.7` | The Ollama config: GGUF path, Qwen3 chat template, Echo's system prompt and tool definitions, sampling parameters. | — |
| `requirements.txt` | Python dependencies. | — |

---

## Quickest path — chat with the published model

```bash
pip install ollama colorama
python ollama_test.py
```

The script checks whether `vasuverma/echo-v4.7` is installed and offers to pull it
(~4.3 GB) if not. Then type a message, or a number `1`–`8` to fire a sample scenario.
`clear` resets the conversation, `exit` quits.

## The Streamlit UI

```bash
pip install -r requirements.txt
streamlit run streamlit_echo.py
```

This one is the demo to show people. Tool calls are parsed out of the model's response and
executed against an in-memory calendar, so booking a slot visibly fills it in. **Reset
Calendar** puts it back. Every turn is logged to `logs/`.

It loads `Qwen/Qwen3-4B` in 4-bit and stacks the two adapters from
`output/scheduling-agent-v4.{6,7}-aws/`, so it needs a CUDA GPU and the trained adapters
on disk. If you only want to *show* Echo working and have neither, use `ollama_test.py`
instead — it talks to the published model and runs on anything.

## Running the local adapters instead

```bash
python interactive_agent_v47.py
```

Expects the adapters at `output/scheduling-agent-v4.6-aws/final_v4.6` and
`output/scheduling-agent-v4.7-aws/final_v4.7`. Needs a CUDA GPU. Useful when you have just
finished training and want to check the adapters before converting to GGUF.

In-chat commands: `calendar` shows the current state, `calendar 25` shows March 25,
`messages` shows the sent-message log, `quit` exits.

---

## Rebuilding and republishing the model

```bash
python prepare_ollama_v47.py                # merge adapters → merged model
python prepare_ollama_v47.py --skip-merge   # merge already done, just convert + register
python prepare_ollama_v47.py --push         # merge + push to Ollama Hub
```

The merge step takes 1–2 hours and produces ~8 GB in `output/echo-v4.7-merged/`.
See [`../PIPELINE.md`](../PIPELINE.md) for the full six-step walkthrough including the
llama.cpp GGUF conversion.

> **Merge order is not optional.** V4.6 must be merged into the base *first*, then V4.7
> loaded on top. Any other order gives a different, worse model.

> **The `TEMPLATE` block in the Modelfile is not optional either.** Without it Ollama
> falls back to raw `{{ .Prompt }}`, Qwen3's chat tokens are never applied, and the model
> hangs in an infinite thinking loop on every request.

---

## Not in this directory

Model weights. `.gguf`, `.safetensors`, `.bin`, and training checkpoints are gitignored —
they are gigabytes. `output/` holds only the `Modelfile`. Pull the model from
<https://ollama.com/vasuverma/echo-v4.7>, or rebuild it with `prepare_ollama_v47.py`.
