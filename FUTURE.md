# MLOps Fine-Tuning Hub — Future Roadmap: AWS S3 + EC2 Integration

> **Prepared for:** Head of Engineering  
> **Date:** April 1, 2026  
> **Author:** Vasu  
> **Purpose:** Strategic proposal to evolve the **general-purpose** GenAI / SLM fine-tuning pipeline into a cloud-native, **multi-team** MLOps platform using AWS S3 and EC2.

---

## Executive Summary

The **MLOps Fine-Tuning Hub** is a **modular, model-agnostic** pipeline for supervised fine-tuning of small and medium language models (SLMs / LLMs) using QLoRA. It is **not tied to a single product or domain**: teams can bring **any Hugging Face–compatible causal LM** (Qwen, Llama, Mistral, Gemma, Phi, custom checkpoints, etc.), standard **JSONL / JSON / CSV** chat or instruction data, and optional **evaluation harnesses** tailored to their task.

Today the hub runs **locally** (Streamlit UI + CLI, MLflow on SQLite, artifacts on disk). That is ideal for onboarding and experimentation.

**As we scale to shared use across teams — larger corpora, bigger models, concurrent experiments, and governed data — local infrastructure becomes a bottleneck.** Integrating **AWS S3** (durable, versioned object storage) and **AWS EC2** (elastic GPU compute) turns the same modular pipeline into a **reusable internal platform**: any squad doing GenAI fine-tuning can plug in their model preset, dataset, and eval logic without rebuilding infrastructure.

---

## Product Vision: A Generalized, Shareable Fine-Tuning Platform

| Principle | What it means in this repo |
|-----------|----------------------------|
| **Model-agnostic** | Presets live under `configs/models/*.yaml`; `_template.yaml` documents how to add a new base model. The UI and trainer consume a unified `ModelConfig` (quantization, LoRA targets, seq length). |
| **Data-format–agnostic (within standard chat/SFT)** | `loader.py` + `validator.py` support JSONL (chat), JSON, and CSV column mapping — suitable for instruction tuning, tool-use traces, or plain Q/A. |
| **Task-agnostic core** | Training is standard **SFT** (TRL `SFTTrainer`). Domain-specific behavior lives in **datasets** and **optional** eval scorers — not in a hard-coded “one product” path. |
| **Pluggable evaluation** | The harness (`src/evaluation/`) can be extended or swapped: one team uses tool-call scoring; another uses BLEU/ROUGE, LLM-as-judge, or custom JSON checks. |
| **Shareable across teams** | Same codebase, different `configs/`, different S3 prefixes or MLflow experiments per team — no fork required for “another use case.” |

**Goal:** Any team doing GenAI fine-tuning (customer support bots, internal copilots, domain adapters, safety tuning, RAG-style instruction data, etc.) can adopt this pipeline **without** maintaining separate training glue code.

---

## Current State: What We Have Today

| Component | Current Implementation | Limitation at Multi-Team / Scale |
|-----------|------------------------|----------------------------------|
| **Dataset storage** | Local filesystem (`outputs/<exp>/data/`) | Not durable; hard to share across squads or regions |
| **Trained adapters** | Local `outputs/<exp>/final_adapter/` | No central registry; another team cannot pull artifacts by URI |
| **Checkpoints** | Local `outputs/<exp>/checkpoint-*/` | Spot / crash = lost progress unless copied manually |
| **MLflow backend** | SQLite (`mlflow.db`) | Single-writer; weak story for concurrent runs from many users |
| **Compute** | Developer GPU (VRAM-limited) | Caps model size and batch size; blocks the machine for hours |
| **Base model cache** | Hugging Face → local disk per machine | Repeated full downloads for every new trainer |
| **Experiment artifacts** | Local `mlruns/` | Not visible org-wide; hard to audit or reproduce |

The pipeline **already** supports multiple families (see `configs/models/`: Qwen 2.5/3, Llama 3/3.1/3.2). Cloud backing **amplifies** that modularity: Team A fine-tunes Llama 3.1-8B on legal Q&A; Team B fine-tunes Qwen3-4B on support tickets — **same platform**, different config + S3 namespace + MLflow experiment.

---

## Proposed Future Architecture: AWS S3 + EC2

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FUTURE CLOUD ARCHITECTURE (MULTI-TEAM)                  │
│                                                                              │
│  ┌──────────────────┐     ┌─────────────────────────────────────────────┐   │
│  │  UI / CLI        │     │               AWS CLOUD                     │   │
│  │  (any engineer)  │────▶│                                             │   │
│  └──────────────────┘     │  ┌──────────────┐    ┌───────────────────┐  │   │
│                           │  │   AWS S3     │    │    AWS EC2        │  │   │
│  ┌──────────────────┐     │  │              │    │  (GPU Instance)   │  │   │
│  │  MLflow Tracking │────▶│  │  datasets/   │    │                   │  │   │
│  │  (S3 + DB)       │     │  │  by-team/    │    │  SFTTrainer       │  │   │
│  └──────────────────┘     │  │  adapters/   │◀──▶│  QLoRA (any HF    │  │   │
│                           │  │  checkpoints/│    │  model in config) │  │   │
│                           │  │  base-models/│    │  Eval harness     │  │   │
│                           │  │  mlruns/     │    │  (team-specific)  │  │   │
│                           │  └──────────────┘    └───────────────────┘  │   │
│                           └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Convention (example):** `s3://<org>-mlops-finetuning/<team-or-project>/datasets/...` and `.../adapters/...` — bucket name is **configurable**; paths are **namespaced** so teams do not step on each other’s artifacts.

---

## Part 1: AWS S3 Integration

### What S3 Would Replace / Augment

S3 provides **durable, versioned, URI-addressable** storage. It plugs into the **same** phases described in `PIPELINE.md` — no change to the *conceptual* pipeline, only to **where bytes live** and **how jobs discover them**.

---

### 1.1 Dataset Storage & Versioning

**Current behavior:** Uploads land under `outputs/<exp>/data/` on one machine.

**With S3:**

```
User or CI uploads dataset
         ↓
upload to s3://<bucket>/<team>/datasets/<content-hash>/train.jsonl
         ↓
PipelineConfig: train_file = "s3://..."
         ↓
EC2 training job streams or copies dataset at job start
```

**Why this matters for GenAI fine-tuning (any team):**

- **Reproducibility:** Every run points at an **immutable** dataset URI. Compliance and research both care that “model M was trained on dataset D,” not “some file on someone’s laptop.”
- **Collaboration:** Curated corpora (PII-scrubbed, licensed) live in one place; annotators and ML engineers use the same source of truth.
- **Scale:** Instruction datasets and synthetic data generation can reach **millions** of rows — local disks and laptops are the wrong tier.
- **Multi-tenant hygiene:** Prefix per team or per project (`legal/`, `support/`, `platform/`) keeps governance and lifecycle policies simple.

---

### 1.2 Adapter & Checkpoint Storage

**Current behavior:** Adapters and checkpoints are local; adapter stacking uses `adapter_path` on disk.

**With S3:**

```
Training completes on EC2
         ↓
Save locally → sync to s3://<bucket>/<team>/adapters/<run-id>/final_adapter/
Checkpoints → s3://<bucket>/<team>/adapters/<run-id>/checkpoint-*/
         ↓
Next run: adapter_path = "s3://.../previous-run/final_adapter/"
```

**Why this matters:**

- **Resume after spot interruption:** Checkpoints in S3 → **standard** for serious GPU training; essential when many teams share a spot-based cost model.
- **Artifact handoff:** Team fine-tunes; **platform or serving** team pulls the same URI for deployment or merges — no manual zip files.
- **Adapter stacking as a product pattern:** Continual improvement (v1 → v2 → v3 LoRA) stays **mechanically** the same for any domain.

---

### 1.3 MLflow Artifact Backend

**Current:** SQLite + local `mlruns/`.

**Future:** Central MLflow server with **artifact root on S3** and a proper **metadata DB** (PostgreSQL or managed equivalent).

**Why this matters:**

- **One leaderboard for the org:** Compare runs across **teams** (different models, different datasets) with consistent metrics.
- **Audit trail:** Who trained what, when, with which params — critical for regulated or customer-facing GenAI.
- **No attachment to one laptop:** Artifacts behave like **build artifacts** in CI/CD.

---

### 1.4 Base Model Cache (Any HF Architecture)

**Current:** Each new machine downloads Qwen, Llama, etc. from the internet.

**With S3:**

```
One-time or automated: HF model → s3://<bucket>/shared/base-models/<sanitized-model-id>/
         ↓
EC2 jobs copy from S3 (fast in-region) instead of throttling on public HF
```

**Why this matters:** Teams switching between **Llama**, **Qwen**, **Mistral**, and internal forks all benefit from **one** cache policy — aligned with the hub’s **YAML-driven model configs**, not a single vendor.

---

## Part 2: AWS EC2 Integration

### Why Local GPU Is a Bottleneck for a Shared Platform

| Constraint | Local (typical) | EC2 GPU (elastic) |
|------------|-----------------|-------------------|
| VRAM | Consumer card (8–24 GB) | Up to very large multi-GPU instances |
| Model class | Practical ceiling for full finetune without cloud | Same **code path**; larger `configs/models/*.yaml` presets become usable |
| Throughput | One person’s machine | **Burst** capacity for many teams (queues + autoscaling patterns) |
| Availability | Tied to individual hardware | Jobs run **when scheduled**, not when a specific laptop is free |
| Cost model | Hidden (amortized hardware) | **Transparent** per job — easier to allocate to cost centers / teams |

---

### 2.1 On-Demand Training Jobs

**Pattern:** UI or CLI submits a job → EC2 (spot or on-demand) runs the **existing** `run.py train` (or equivalent) with config + S3 paths → uploads results → terminates.

**Benefits:** Engineers are not **blocked** on a single GPU machine; **nightly** or **CI-triggered** fine-tunes become realistic for **every** consumer of the hub.

---

### 2.2 Larger Models & Bigger Batches (Still Modular)

The **same** trainer and LoRA stack apply; only **instance size** and **YAML** change. Examples:

| Instance class | Typical use |
|----------------|-------------|
| Single mid GPU | 3B–8B SLMs at 4-bit, default presets |
| Larger single / multi-GPU | 13B–70B+ class models for teams that need them |

**Point for leadership:** We are not building “one model pipeline”; we are building **one engine** whose **limits** are lifted by cloud compute.

---

### 2.3 Parallel Hyperparameter & Ablation Studies

Grid search over `lora_r`, `learning_rate`, `max_seq_length`, etc. becomes **parallel EC2 jobs**, all logging to the **same** MLflow — ideal when **many teams** want quick answers without a centralized GPU cluster on-prem.

---

### 2.4 Inference / Serving on EC2

Decouple **training** from **try-the-model**: a shared inference endpoint (e.g. FastAPI + vLLM) loads **base + adapter from S3** by URI. **Any** team that produced an adapter can share a **stable URL** for PM/QA review — again, **not** product-specific.

---

## Part 3: Upgraded Pipeline Flow (Unchanged Semantics, Cloud-Backed)

The **nine phases** in `PIPELINE.md` stay the same **conceptually**:

1. Configuration (YAML + overrides)  
2. Data validation & loading  
3. Model load + optional adapter merge + LoRA  
4. Training  
5. Post-train metrics  
6. MLflow logging  
7. Optional evaluation harness  
8. Chat / inference  
9. Experiments dashboard  

**Change:** durable storage and compute move to **S3 + EC2**; **UI** may point at remote MLflow and remote inference. **Teams** differ by **config + data + eval**, not by **forking the trainer**.

---

## Part 4: Estimated Cost Model (Illustrative)

Same methodology as before: **pay per GPU-hour** for training + **cheap** S3 for artifacts. Numbers vary by region and instance generation; treat as **order-of-magnitude** for budgeting.

| Scenario | Rough idea |
|----------|------------|
| Small SLM run (e.g. 4B–8B class, modest data) | Low single-digit dollars per run on spot-friendly instances |
| Parallel sweep | Wall-clock ≈ one run; cost ≈ sum of instances × time |
| S3 | Cents to low dollars per month for adapters + datasets at typical scales |

**Platform economics:** One shared S3 + MLflow footprint can serve **many** teams; marginal cost is **per training job**, not **per team repo**.

---

## Part 5: Implementation Roadmap (Modular, Feature-Flagged)

### Phase A — S3 Storage Layer

- Centralize I/O in something like `src/storage/s3_client.py` (boto3 wrapper).
- Teach **loader / trainer / callbacks / adapter_stack** to accept `s3://` URIs alongside local paths.
- **Feature flag** in `configs/defaults.yaml` (or env): `s3.enabled: false` keeps **100% local** behavior for contributors and air-gapped use.

**Example config (bucket is organizational, not product-specific):**

```yaml
s3:
  bucket: "${MLOPS_FINETUNING_BUCKET}"  # e.g. org-mlops-finetuning
  region: us-east-1
  enabled: false
  # Optional: default prefix for a team or project
  key_prefix: "${TEAM_NAME}/"
```

### Phase B — EC2 Training Jobs

- `src/cloud/ec2_launcher.py` (or equivalent): launch, bootstrap, poll, terminate.
- Headless `run.py train` as the **contract** so **CI** and **UI** both submit the same entrypoint.

### Phase C — Remote MLflow

- Tracking URI from environment; artifacts on S3; DB suitable for concurrent users.

### Phase D — Remote Inference (Optional)

- HTTP inference service; Chat tab becomes a **client** of that API.

**Throughout:** Preserve **modularity** — no coupling to a single product name inside core training paths.

---

## Part 6: Why S3 + EC2 Helps **Cross-Team** Adoption

1. **Single platform, many use cases:** Support, legal review assistants, code helpers, internal search-grounded bots — all **SFT + LoRA** with different JSONL and eval — share **one** training, tracking, and artifact story.

2. **Governance:** Data and model artifacts in S3 under IAM, encryption, and lifecycle policies beat **scattered** copies on laptops — important for **any** GenAI program that touches sensitive text.

3. **Reproducibility & handoffs:** New hire or different team can reproduce a run from **URI + commit hash + config** — essential when **multiple teams** touch the same hub.

4. **No reinventing the wheel:** Squads avoid writing their own “download model, train LoRA, log something, save adapter” scripts — they add **YAML + dataset + (optional) scorer**.

5. **Elastic demand:** When several teams experiment the same week, **cloud** absorbs spikes; **on-prem** GPU clusters often sit idle or are **over-subscribed**.

6. **Vendor-neutral models:** The hub already treats **Llama**, **Qwen**, and others uniformly; cloud storage and GPU **scale** that story without locking to one model vendor.

---

## Summary: What We Gain (Platform View)

| Capability | Local hub today | With S3 + EC2 |
|------------|-----------------|---------------|
| **Audience** | Individual / small group | **Org-wide** GenAI fine-tuning |
| **Artifacts** | Per-machine disk | **URI-addressed**, versioned, shareable |
| **Experiments** | SQLite / single user | **Central** MLflow + durable artifacts |
| **Compute** | Fixed local GPU | **Elastic** GPU per job |
| **Models** | YAML presets + custom HF IDs (already) | **Same** — larger instances unlock bigger presets |
| **Teams** | Informal sharing | **Namespaced** buckets, experiments, IAM |
| **Serving** | Local chat only | Optional **shared** inference endpoint |

The current codebase is a **strong modular baseline**: configuration-driven models, standard data loaders, pluggable evaluation, and MLflow hooks. **AWS S3 + EC2** are the natural next step to make that baseline a **general-purpose internal fine-tuning service** — not a single-product script.

---

*Forward-looking engineering proposal. AWS pricing and instance families change; validate current numbers against your org’s AWS account and region.*

---

## Parked Ideas (not yet scoped)

Captured so they are not lost; none of these are committed work.

- **Synthetic data generation in the hub.** Echo's training sets were produced by
  generator functions inside its training scripts. Pulling that into the hub as a first
  class step -- describe the scenarios, get a JSONL -- would remove the biggest manual
  step in starting a new fine-tune.
- **Full config surface in the UI.** The Train tab exposes the common hyperparameters;
  the model YAML exposes more. Closing that gap would remove most reasons to edit YAML
  by hand.
- **Cost tracking per run.** Alongside latency and tokens/sec, record GPU-hours and an
  estimated dollar cost per run, so experiments can be compared on price as well as
  accuracy.
