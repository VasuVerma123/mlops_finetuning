# Echo — Benchmark Results & Training Recipes

How the 98% number is produced, what it does and does not mean, and the exact recipes
behind each version.

Everything here is derived from the code in this repository:
[`aws_eval_v4_7.py`](aws_eval_v4_7.py), [`aws_finetune_v4_6.py`](aws_finetune_v4_6.py),
and [`aws_finetune_v4_7.py`](aws_finetune_v4_7.py).

---

## 1. Headline

| Model | Accuracy | Checks |
|---|---:|---:|
| **V4.7-AWS** (published to Ollama Hub) | **98%** | **47 / 48** |
| V4.6-AWS | 90% | 43 / 48 |
| V4.2 (`enable_thinking=True`) | 85% | — |
| V4.5 | 77% | — |
| V4 | 73% | — |
| Qwen3-4B base | 40% | — |

Fine-tuning moved the base model from **40% → 98%**, a 58-point gain on this benchmark.

The V4.5, V4, and V4.2 runs predate this repository — they were trained locally and only
their headline accuracies survive, as the comparison constants in `aws_eval_v4_7.py`.
V4.6 and V4.7 are fully reproducible from the scripts here.

---

## 2. The benchmark

### What it measures

One thing: **does the model pick the right action?** Each scenario is a single incoming
message plus a snapshot of the day's calendar. The model must either emit the correct
tool call with the correct arguments, or correctly emit no tool call at all.

Six tools are offered in the system prompt:

| Tool | Required arguments |
|---|---|
| `check_availability` | `date` |
| `book_appointment` | `date`, `time_slot`, `client_name`, `service` |
| `cancel_appointment` | `date`, `time_slot`, `client_name` |
| `reschedule_appointment` | `date`, `old_time_slot`, `new_time_slot`, `client_name` |
| `send_message` | `recipient`, `message` |
| `get_weather` | `location` |

`get_weather` is a **trap**. It is never the right answer. It is there so the benchmark can
measure whether the model resists calling an available-but-irrelevant tool — the single
most common failure mode in small tool-calling models.

### The 8 scenarios

| # | Scenario | Incoming message | Correct behaviour | Checks |
|---|---|---|---|---:|
| 1 | Direct Booking | Harsh Verma asks to book 3 PM | call `book_appointment` | 7 |
| 2 | Availability | Elena Rodriguez asks what's free | call `check_availability` | 7 |
| 3 | Cancellation | Tom Wilson cancels his 4 PM | call `cancel_appointment` | 7 |
| 4 | FAQ | Kevin Park asks about working hours | answer in text, **no tool** | 5 |
| 5 | Conflict | Lisa Chang asks for 11 AM, already booked | decline **and** offer alternatives | 5 |
| 6 | Privacy | Unknown caller asks who has the 9 AM | refuse to name the client | 5 |
| 7 | Reschedule | Anika Patel wants 2 PM moved to 3 PM | call `reschedule_appointment` | 6 |
| 8 | Distractor | Random person asks about the weather | decline, **no** `get_weather` | 6 |
| | | | **Total** | **48** |

### How a scenario is scored

Each scenario is worth several independent boolean checks, not one pass/fail. The overall
accuracy is `passed checks / 48`.

**For scenarios that should call a tool** (1, 2, 3, 7):

| Check | Passes when |
|---|---|
| `called_tool` | the response contains a `<tool_call>` block |
| `valid_json` | the block parses as JSON |
| `correct_tool` | `name` matches the expected tool |
| `has_params` | every required argument is present |
| `avoided_<tool>` | it did not call a named wrong tool (1–2 of these per scenario) |
| `has_reasoning` | the response contains a `<think>` block |

**For scenarios that should not call a tool** (4, 5, 6, 8):

| Check | Passes when |
|---|---|
| `correctly_no_tool` | no `<tool_call>` block at all |
| `avoided_<tool>` | it did not call a named wrong tool (1–3 of these per scenario) |
| `has_reasoning` | the response contains a `<think>` block |

Plus one scenario-specific check each:

| Scenario | Extra check | Passes when |
|---|---|---|
| 5 Conflict | `detected_conflict` | reply says the slot is unavailable / taken / booked |
| 5 Conflict | `suggested_alternatives` | reply names another time or offers options |
| 6 Privacy | `kept_privacy` | reply contains neither "sarah" nor "kim" |
| 8 Distractor | `declined_off_topic` | reply redirects to scheduling / says it can't help |

Generation settings for every scenario: `temperature=0.3`, `top_p=0.9`, `do_sample=True`,
`repetition_penalty=1.1`, `max_new_tokens=800`, `enable_thinking=True`.

---

## 3. V4.6 → V4.7: what was fixed

V4.6 scored **43/48 (90%)**. All five lost checks were in two scenarios:

**Reschedule — 4/6.** Asked to move an appointment, the model called
`cancel_appointment`. It had learned "change an appointment" but not the distinction
between cancelling and moving. `reschedule_appointment` was only 10 of 130 examples in
the V4.6 dataset.

**Distractor — 3/6.** Asked about the weather, the model called `get_weather`. The tool was
in its tool list, the question matched it, so it fired. Resisting an available tool is a
behaviour that has to be trained in explicitly.

V4.7's dataset was built to attack exactly these two, while keeping enough of everything
else to avoid catastrophic forgetting:

| Category | V4.6 examples | V4.7 examples | Intent |
|---|---:|---:|---|
| Reschedule | 10 | **40** | fix: teach the distinct tool |
| Distractor | 30 | **35** | fix: teach refusal over `get_weather` |
| Booking | 12 | 8 | retention |
| Cancel | 10 | 6 | retention |
| Availability | 40 | 6 | retention |
| Conflict | 8 | 5 | retention |
| Privacy | 15 | 3 | retention |
| FAQ | 5 | 2 | retention |
| **Total** | **130** | **105** | |
| Tool / no-tool balance | 55% / 45% | 57% / 43% | |

Result: **47/48 (98%)**. Four of the five lost checks recovered. Every example in both
datasets includes a `<think>` reasoning block, which is why `has_reasoning` passes across
the board.

> The per-scenario breakdown for V4.7 was not committed — only the 47/48 total, which
> appears in `streamlit_echo.py` and the headers of `interactive_agent_v47.py` and
> `prepare_ollama_v47.py`. Re-run the eval to recover the detail.

---

## 4. Training recipes

Both versions are **self-contained**: the script generates its own synthetic dataset, so
there are no data files to download. `random.seed(42)` is set, so the datasets regenerate
identically.

| | V4.6 | V4.7 |
|---|---|---|
| Base model | `Qwen/Qwen3-4B` | `Qwen/Qwen3-4B` |
| Starts from | base (fresh LoRA) | base + **merged V4.6** |
| Epochs | 4 | 5 |
| Learning rate | 5e-5 | 4e-5 |
| Batch size | 2 | 2 |
| Gradient accumulation | 4 | 4 |
| Effective batch | 8 | 8 |
| Max sequence length | 1536 | 1536 |
| LoRA rank (`r`) | 32 | 16 |
| LoRA alpha | 64 | 32 |
| Precision | bf16, no quantization | bf16, no quantization |
| Dataset size | 130 (114 train / 16 val) | 105 (92 train / 13 val) |
| Hardware | AWS `g5.2xlarge` (A10G 24 GB) | AWS `g5.2xlarge` (A10G 24 GB) |

LoRA target modules, both versions:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

V4.7 uses a **lower rank and a lower learning rate than V4.6 on purpose** — it is a
corrective pass on an already-good model, so it should nudge rather than overwrite. The
stack at inference time is:

```
Qwen/Qwen3-4B
   └── merge V4.6 adapter   (becomes the new base)
         └── load V4.7 adapter on top
```

Merge order matters. Loading V4.7 first, or merging both in the wrong sequence, produces
a different and worse model — `prepare_ollama_v47.py` carries a note about this because it
was originally wrong.

---

## 5. What this number does not mean

Worth being explicit, because "98%" invites over-reading:

- **It is 47 of 48 checks on 8 scenarios.** It is a focused smoke test of tool selection,
  not a general capability benchmark. It says nothing about summarisation, long-context
  behaviour, or robustness to adversarial phrasing.
- **Scoring is lexical, not model-graded.** `detected_conflict` passes on a keyword list;
  `kept_privacy` passes if the strings "sarah" and "kim" are absent. A correct reply with
  unusual wording can score as a miss, and a wrong reply containing the right keyword can
  score as a hit.
- **Single-turn.** Every scenario is one message. Multi-turn consistency is not scored.
- **One persona, one calendar.** The benchmark uses one fixed schedule for March 24, 2026,
  and one therapist's rules. Generalisation to other personas is untested.
- **Sampling is on** (`do_sample=True`, `temperature=0.3`), so runs vary slightly. The
  48 checks are not all independent either — a single wrong tool call fails several at once.
- **The training data is synthetic**, generated by the same author as the benchmark. The
  eval scenarios are held out from training, but they come from the same distribution,
  which flatters the score relative to real traffic.

A fair summary: *on the tool-selection behaviours it was trained for, Echo v4.7 is
reliable, and the fine-tuning clearly worked — base 40% to 98%. Treat it as evidence the
training pipeline works, not as a claim about general ability.*

---

## 6. Reproducing

```bash
pip install torch transformers peft accelerate trl datasets

# Train (each script generates its own data)
python aws_finetune_v4_6.py
python aws_finetune_v4_7.py

# Evaluate
export ECHO_V46_ADAPTER=/path/to/final_v4.6
export ECHO_V47_ADAPTER=/path/to/final_v4.7
export ECHO_EVAL_RESULTS=eval_v47_results.json
python aws_eval_v4_7.py
```

`eval_v47_results.json` will contain the overall accuracy, per-scenario passed/total, the
names of any failed checks, and the full raw response for every scenario — enough to audit
each of the 48 checks by hand.
