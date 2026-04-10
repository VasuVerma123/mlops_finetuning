# MLOps Fine-Tuning Hub — Architecture & Deployment Plan

> **Purpose:** Share with team before implementation. Shows the target architecture,
> what is already built, what needs to be added, and exactly what credentials/config
> each person needs to provide.

---

## Target Architecture (Plan B — Recommended)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TEAM MEMBER (Anyone)                            │
│                                                                         │
│   Opens browser → http://<EC2-PUBLIC-IP>:8501                          │
│   No installation. No AWS credentials. Just a URL.                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  Browser (HTTP)
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        AWS EC2 INSTANCE                                 │
│                    (GPU, e.g. g4dn.xlarge)                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   Docker Container                                │  │
│  │                 (built from Dockerfile.aws)                       │  │
│  │                                                                   │  │
│  │   ┌─────────────────────────────────────────────────────────┐   │  │
│  │   │              Streamlit UI  (port 8501)                   │   │  │
│  │   │                                                           │   │  │
│  │   │   Tab: Train    ──► QLoRA Training (runs on this GPU)    │   │  │
│  │   │   Tab: Evaluate ──► Eval Harness                         │   │  │
│  │   │   Tab: Chat     ──► Inference with loaded adapter        │   │  │
│  │   │   Tab: Experiments ──► MLflow Dashboard                  │   │  │
│  │   └────────────┬──────────────────┬────────────────────────-─┘   │  │
│  │                │                  │                                │  │
│  │        Upload  │          Metrics │  Artifacts                    │  │
│  │        Dataset │          & Logs  │  (adapters)                   │  │
│  └────────────────┼──────────────────┼────────────────────────────---┘  │
│                   │                  │                                   │
│    EC2 IAM Role ──┘                  │                                   │
│    (no hardcoded keys needed)        │                                   │
└───────────────────┬──────────────────┼───────────────────────────────────┘
                    │                  │
          boto3 SDK │                  │ boto3 SDK
                    ▼                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            AWS S3 BUCKET                                  │
│                                                                            │
│   s3://<your-bucket>/                                                     │
│   ├── datasets/                                                           │
│   │   ├── train.jsonl          ◄── uploaded from UI                      │
│   │   └── val.jsonl                                                       │
│   │                                                                       │
│   ├── outputs/                                                            │
│   │   └── <experiment-name>/                                              │
│   │       ├── final_adapter/   ◄── LoRA weights saved here after train   │
│   │       └── loss_log.json                                               │
│   │                                                                       │
│   └── mlflow-artifacts/        ◄── MLflow stores all run artifacts here  │
│       └── <experiment-id>/                                                │
│                                                                           │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — Step by Step

```
1. DATASET UPLOAD
   ─────────────
   User picks a file in UI
        │
        ▼
   [MISSING] boto3.upload_file()  →  s3://<bucket>/datasets/train.jsonl
        │
        ▼
   S3 URI stored in DB  (already has s3_link column ✅)


2. TRAINING
   ────────
   User sets hyperparams → clicks Train
        │
        ▼
   [MISSING] Training code pulls dataset FROM S3  →  loads into memory
        │
        ▼
   QLoRA training runs on EC2 GPU  (training engine already built ✅)
        │
        ▼
   [MISSING] Final adapter pushed TO S3  →  s3://<bucket>/outputs/<exp>/final_adapter/
        │
        ▼
   MLflow logs metrics (currently SQLite ✅, needs S3 backend ❌)


3. EXPERIMENTS DASHBOARD
   ─────────────────────
   User opens Experiments tab
        │
        ▼
   [MISSING] MLflow reads from S3 artifact store  →  shows all runs
   (currently reads from local mlflow.db)


4. CHAT / INFERENCE
   ─────────────────
   User selects a trained adapter → clicks Load
        │
        ▼
   [MISSING] App pulls adapter FROM S3  →  loads into GPU memory
        │
        ▼
   Chat works as normal  (inference engine already built ✅)
```

---

## What Is Already Built vs. What Is Missing

### Already Done ✅

| Component | What It Does | File |
|-----------|-------------|------|
| Streamlit UI | Full 4-tab interface (Train, Eval, Chat, Experiments) | `app.py` |
| Training engine | QLoRA fine-tuning, adapter stacking, callbacks | `src/training/` |
| Data pipeline | JSONL/CSV loading, validation, tokenization | `src/data/` |
| Evaluation harness | Tool-call scoring, metrics, latency | `src/evaluation/` |
| Inference engine | Chat with loaded LoRA adapter | `src/serving/` |
| MLflow tracking | Params, metrics, loss curves (local SQLite today) | `src/tracking/` |
| Database | Experiment + dataset registry (has `s3_link` column) | `src/database/db.py` |
| Docker — base | Python 3.11, runs Streamlit, local use | `Dockerfile` |
| Docker — GPU | CUDA 12.4, for local GPU machine | `Dockerfile.gpu` |
| Docker — AWS | Non-root user, healthcheck, dynamic PORT, ECR-ready | `Dockerfile.aws` |
| Docker Compose | GPU compose + local CPU compose | `docker-compose*.yml` |
| Entrypoint script | Starts Streamlit with dynamic port (App Runner safe) | `scripts/docker-entrypoint-aws.sh` |

### Still Missing ❌

| Component | What It Needs to Do | Where It Goes |
|-----------|-------------------|---------------|
| S3 upload on dataset ingest | When user uploads file → `boto3.upload_file()` to S3 | `app.py` upload section |
| S3 download before training | Pull dataset from S3 URI before `run_training()` | `src/data/loader.py` |
| S3 upload after training | Push `final_adapter/` folder to S3 after training ends | `src/training/trainer.py` |
| S3 download for inference | Pull adapter from S3 when user loads it in Chat tab | `app.py` chat section |
| MLflow remote artifact store | Point MLflow at `s3://<bucket>/mlflow-artifacts/` | `app.py` MLflow init |
| `boto3` in requirements | AWS SDK not in requirements.txt yet | `requirements.txt` |
| `.env` / env vars wiring | App reads `S3_BUCKET`, `AWS_REGION` from environment | `app.py` config section |

**Total missing:** ~6 focused code changes. Everything else is already wired.

---

## Configuration — What You Need to Provide

### On the EC2 Machine (one-time setup by DevOps / team lead)

```bash
# These go into a .env file on the EC2 instance
# OR set as EC2 environment variables
# OR use EC2 IAM Role (recommended — no hardcoded keys)

S3_BUCKET=your-bucket-name          # e.g. mlops-finetuning-team
AWS_REGION=us-east-1                # whichever region your bucket is in
MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db   # changes to S3 when ready
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx    # HuggingFace token to download base models
```

### For Each Team Member (nothing — just a URL)

```
Open browser → http://<EC2-PUBLIC-IP>:8501
Done.
```

No AWS credentials. No Python install. No Docker. Just a browser.

### IAM Permissions Needed on EC2 Role (DevOps sets this once)

```
s3:PutObject        → upload datasets and adapters
s3:GetObject        → download datasets and adapters
s3:ListBucket       → browse bucket contents
s3:DeleteObject     → cleanup old runs (optional)
```

---

## Deployment Steps (When Ready)

```
Step 1 — DevOps
  Create S3 bucket
  Create EC2 instance (g4dn.xlarge recommended for GPU)
  Attach IAM Role with S3 permissions above
  Open port 8501 in EC2 Security Group

Step 2 — DevOps
  SSH into EC2
  Pull this repo
  Fill in .env file (S3_BUCKET, AWS_REGION, HF_TOKEN)
  docker build -f Dockerfile.aws -t mlops-hub .
  docker run -p 8501:8501 --env-file .env mlops-hub

Step 3 — Everyone
  Open http://<EC2-PUBLIC-IP>:8501
  Upload dataset → stored in S3
  Train → runs on EC2 GPU → adapter saved to S3
  View experiments → MLflow reads from S3
  Chat → adapter loaded from S3
```

---

## What Changes Between "Works Locally" and "Works on AWS"

| Thing | Local Today | AWS Target |
|-------|------------|------------|
| Where datasets go | `outputs/<exp>/data/` on disk | `s3://<bucket>/datasets/` |
| Where adapters go | `outputs/<exp>/final_adapter/` on disk | `s3://<bucket>/outputs/<exp>/final_adapter/` |
| MLflow artifacts | `mlruns/` folder on disk | `s3://<bucket>/mlflow-artifacts/` |
| MLflow database | `mlflow.db` SQLite file | Same SQLite on EC2 (good enough for single-server) |
| Who runs training | Your laptop's GPU | EC2's GPU |
| Who runs the UI | Your laptop | EC2 (everyone hits the same URL) |

> **Note:** MLflow's tracking database (experiment list, run metadata) stays as SQLite on EC2.
> Only the **artifacts** (model weights, loss logs) move to S3.
> This is standard MLflow practice for single-server setups.

---

## Summary for Team Approval

**What Abid built:** Full Docker packaging — the app can be deployed to EC2 and run as a container. Solid foundation.

**What is missing:** S3 read/write calls in ~6 places in the existing code. The hooks are already there (s3_link column in DB, MLflow artifact config), they just need the actual boto3 calls wired in.

**Effort to complete:** Small — no new architecture, no new tabs, no redesign. Just plug boto3 into existing upload/download/save points.

**Once done:** Any team member opens a URL, uploads their dataset, trains a model on EC2 GPU, views results — all without installing anything or touching AWS credentials.
