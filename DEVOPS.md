# DevOps Setup Guide — MLOps Fine-Tuning Hub

---

## What You Need to Create in AWS (you know how)

- **EC2 instance** — `p5.4xlarge`, Ubuntu 22.04, 200 GB storage, port `8501` open to `0.0.0.0/0`
- **S3 bucket** — any name, any region. Note the bucket name and region down.
- **IAM Role** — attach `AmazonS3FullAccess`, attach the role to the EC2 instance

---

## Once EC2 is Running — Run These in Order

### 1. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

sudo usermod -aG docker ubuntu
newgrp docker
```

### 2. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPUs are visible:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 3. Clone the Repo

```bash
git clone https://github.com/VasuVerma123/mlops_finetuning.git
cd mlops_finetuning
git checkout abid_docker
```

### 4. Create the .env File

```bash
cp .env.example .env
nano .env
```

Fill in these three lines — get `HF_TOKEN` from Vasu, use your S3 bucket name and region:

```
HF_TOKEN=           ← get from Vasu
S3_BUCKET=          ← bucket you just created
AWS_REGION=         ← region of that bucket
```

Save and close (`Ctrl+X` → `Y` → Enter).

### 5. Start the App

```bash
touch mlflow.db
docker compose up -d --build
```

Check it's running:
```bash
docker compose ps
docker compose logs -f
```

---

## Done

Share this URL with the team:
```
http://<EC2-PUBLIC-IP>:8501
```

---

## Day-to-Day

```bash
# Check status
docker compose ps

# View logs
docker compose logs -f

# Restart
docker compose restart

# Pull new code and redeploy
git pull && docker compose up -d --build

# Free disk — remove old training checkpoints (keeps final adapters)
rm -rf outputs/<experiment-name>/checkpoint-*
```

> **Stopping EC2:** Use "Stop" not "Terminate" in AWS Console. Stop keeps the disk. Terminate deletes everything.
