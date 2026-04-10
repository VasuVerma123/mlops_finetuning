# DevOps Setup Guide — MLOps Fine-Tuning Hub

> One-time setup. After this, the team uses a browser URL — no installs, no access keys, nothing.

---

## What This App Does (Infrastructure Context)

This is a Streamlit-based fine-tuning platform for large language models. The team uses it through a browser — they upload datasets, launch training runs, and view results, all from the UI.

**Why EC2:**
Training a 70B parameter model (Llama 3.1-70B with QLoRA) requires serious GPU compute. The app runs entirely on the EC2 instance — Streamlit UI, training, inference, everything. The team never installs anything locally; they just open the URL you give them.
Recommended instance: `p5.4xlarge` (8x H100 80GB GPUs).

**Why S3:**
All datasets uploaded through the UI, all trained adapter weights, and all experiment history are stored in S3. This means:
- Nothing is lost if the EC2 instance is stopped or replaced
- Any team member's upload is immediately available to everyone
- Trained models persist independently of the machine that trained them

**Why an IAM Role (not access keys):**
The EC2 instance gets an IAM Role with S3 permissions attached directly. This means the app can read/write S3 without any credentials hardcoded or stored anywhere. No key rotation, no secret management.

---

## What You Need to Set Up

Before running the Docker container, three things need to exist in AWS:

**1. An S3 bucket**
Any name, any region. The app will store datasets, trained adapters, and experiment backups here.

**2. An IAM Role for EC2 with S3 access**
Create a role with `AmazonS3FullAccess`, attach it to the EC2 instance. That's the only permission needed.

**3. A `p5.4xlarge` EC2 instance (Ubuntu 22.04 LTS)**
- Attach the IAM Role above
- Open port `8501` to `0.0.0.0/0` in the Security Group (this is the Streamlit UI port)
- Open port `22` to your IP for SSH
- 200 GB storage minimum (Llama 70B weights alone are ~140 GB)

---

## Prerequisites on the EC2 Machine

Once the instance is running, SSH in and make sure these are installed:

- **Docker Engine + Compose plugin** (docker-ce, not docker.io)
- **NVIDIA Container Toolkit** (so Docker can access the H100 GPUs)

Both have standard install guides. After installing, verify with:
```bash
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
The second command should show all 8 H100s.

---

## Before You Run the Container

Clone the repo, then ask the **team lead / ML engineer** to fill in `.env`:

```bash
cp .env.example .env
# hand this off — the team lead fills in their HF token and S3 bucket name
```

They own the HuggingFace account and the S3 bucket — that config is theirs, not yours.
Once `.env` is filled in and placed in the repo directory, you're ready.

---

## Run the App

```bash
touch mlflow.db
docker compose up -d --build
```

That's it. Share `http://<EC2-PUBLIC-IP>:8501` with the team.

---

## Day-to-Day Operations

```bash
# Check status
docker compose ps

# View logs
docker compose logs -f

# Restart
docker compose restart

# Update after a new code push
git pull && docker compose up -d --build

# Free disk — delete training checkpoints (keeps final adapters)
rm -rf outputs/<experiment-name>/checkpoint-*
```

**Stopping EC2 to save cost:** Use "Stop" in the AWS Console, not "Terminate". Stop preserves the disk. Terminate deletes everything.

**After EC2 restart:** The container starts automatically. MLflow history is backed up to S3 after every training run so nothing is lost even if the disk is wiped.
