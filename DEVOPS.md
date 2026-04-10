# DevOps Setup Guide — MLOps Fine-Tuning Hub

> Do this once. After this, the app runs permanently and the team just uses a URL.

---

## What You Need Before Starting

- AWS account with permission to create EC2, S3, and IAM resources
- Your HuggingFace token (get it from huggingface.co → Settings → Access Tokens)

---

## Step 1 — Create the S3 Bucket

In AWS Console → S3 → Create Bucket

- **Bucket name:** pick any name, e.g. `mlops-finetuning-team`
- **Region:** pick one close to you, e.g. `us-east-1`
- Block public access: leave ON (default)
- Everything else: leave default → Create

Or via CLI:
```bash
aws s3 mb s3://mlops-finetuning-team --region us-east-1
```

---

## Step 2 — Create an IAM Role for EC2

In AWS Console → IAM → Roles → Create Role

- **Trusted entity:** AWS Service → EC2
- **Permissions:** attach `AmazonS3FullAccess`
- **Role name:** `mlops-ec2-role`
- Create Role

> This lets the EC2 machine read/write S3 without any hardcoded credentials.

---

## Step 3 — Launch the EC2 Instance

In AWS Console → EC2 → Launch Instance

| Setting | Value |
|---|---|
| **Name** | `mlops-hub` |
| **AMI** | Ubuntu 22.04 LTS |
| **Instance type** | `g4dn.xlarge` (1x T4 GPU, cheapest GPU option) |
| **Key pair** | Create or use existing — you'll need this to SSH in |
| **Storage** | 100 GB gp3 (models are large) |
| **IAM Instance Profile** | `mlops-ec2-role` (the role you just created) |

**Security Group — open these ports:**

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 22 | TCP | Your IP | SSH access |
| 8501 | TCP | 0.0.0.0/0 | Streamlit UI (everyone) |

Launch the instance. Copy the **Public IPv4 address** — that's the URL your team will use.

---

## Step 4 — SSH Into the Machine

```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

---

## Step 5 — Install Docker

Run these commands on the EC2 machine:

```bash
# Update packages
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Allow running Docker without sudo
sudo usermod -aG docker ubuntu

# Apply group change (or just log out and back in)
newgrp docker

# Verify Docker works
docker --version
```

---

## Step 6 — Install NVIDIA Container Toolkit (for GPU)

```bash
# Add NVIDIA repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify GPU is visible
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Step 7 — Clone the Repo

```bash
git clone https://github.com/VasuVerma123/mlops_finetuning.git
cd mlops_finetuning
```

---

## Step 8 — Create the .env File

```bash
cp .env.example .env
nano .env
```

Fill in these two lines (everything else can stay blank):

```
HF_TOKEN=hf_your_token_here
S3_BUCKET=mlops-finetuning-team
AWS_REGION=us-east-1
```

Save and close (`Ctrl+X` → `Y` → Enter in nano).

---

## Step 9 — Build and Run the Container

```bash
# Build the Docker image (takes 5-10 minutes first time)
docker build -f Dockerfile.aws -t mlops-hub .

# Run it in the background on port 8501
docker run -d \
  --gpus all \
  --name mlops-hub \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  mlops-hub
```

Check it started:
```bash
docker ps
docker logs mlops-hub
```

---

## Done — Share This URL With the Team

```
http://<EC2-PUBLIC-IP>:8501
```

The app is running. The team can now open that URL in any browser and use it.

---

## Day-to-Day Operations

### Check if the app is running
```bash
docker ps
```

### View live logs
```bash
docker logs -f mlops-hub
```

### Restart the app
```bash
docker restart mlops-hub
```

### Update the code (when there's a new push)
```bash
cd mlops_finetuning
git pull
docker build -f Dockerfile.aws -t mlops-hub .
docker stop mlops-hub && docker rm mlops-hub
docker run -d --gpus all --name mlops-hub --restart unless-stopped \
  -p 8501:8501 --env-file .env mlops-hub
```

### Free up disk space (delete old checkpoints)
```bash
# List what's taking space
du -sh ~/mlops_finetuning/outputs/*/

# Delete a specific experiment's checkpoints (keeps final_adapter)
rm -rf ~/mlops_finetuning/outputs/<exp-name>/checkpoint-*
```

### Stop EC2 to save money (when not in use)
In AWS Console → EC2 → Select instance → Instance State → **Stop**
(Stop, not Terminate — Stop keeps the disk. Terminate deletes everything.)

### Restore after EC2 restart
The app starts automatically on Docker restart (`--restart unless-stopped`).
MLflow history is backed up to S3 after every training run — nothing is lost.

---

## Cost Estimate

| Resource | Cost |
|---|---|
| g4dn.xlarge EC2 | ~$0.526/hour (stop when not training) |
| S3 storage | ~$0.023/GB/month (negligible for model files) |
| S3 data transfer | Free within same region |
