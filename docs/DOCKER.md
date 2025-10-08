# Docker Deployment Guide for LoRA Craft

<div align="center">
  <img src="../static/images/lora_craft.png" alt="LoRA Craft" width="200"/>
  <h2>Docker Deployment & Production Guide</h2>
</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [GPU Setup](#gpu-setup)
6. [Volume Management](#volume-management)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Security Considerations](#security-considerations)

---

## Overview

This guide covers deploying LoRA Craft using Docker and Docker Compose. The Docker deployment provides:

- ✅ **Consistent environment** across different systems
- ✅ **GPU acceleration** with NVIDIA CUDA 12.8
- ✅ **Persistent data storage** for models, configs, and outputs
- ✅ **Easy updates** and rollbacks
- ✅ **Production-ready** configuration
- ✅ **Isolated dependencies** from host system

### Architecture

```
┌─────────────────────────────────────────┐
│          Docker Container               │
│  ┌───────────────────────────────────┐  │
│  │   Flask App (Port 5000)           │  │
│  │   - Web Interface                 │  │
│  │   - WebSocket Server              │  │
│  │   - Training Engine               │  │
│  └───────────────────────────────────┘  │
│                                          │
│  ┌───────────────────────────────────┐  │
│  │   NVIDIA GPU (CUDA 12.8)          │  │
│  │   - PyTorch Training              │  │
│  │   - Model Inference               │  │
│  └───────────────────────────────────┘  │
│                                          │
│  Persistent Volumes:                    │
│  - /app/outputs  (trained models)       │
│  - /app/exports  (GGUF exports)         │
│  - /app/configs  (configurations)       │
│  - /app/cache    (HuggingFace models)   │
└─────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

#### Hardware
- **GPU**: NVIDIA GPU with CUDA compute capability 6.0+ (Maxwell or newer)
- **VRAM**: Minimum 8GB (16GB+ recommended for larger models)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free disk space
  - Base image: ~10GB
  - Models cache: 5-20GB (varies by models used)
  - Training outputs: 1-10GB per model

#### Software
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **NVIDIA Driver**: Version 535+ (for CUDA 12.8)
- **NVIDIA Container Toolkit**: Latest version

### Installation Steps

#### 1. Install Docker

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to docker group (optional, for non-root access)
sudo usermod -aG docker $USER
newgrp docker
```

**Other systems:**
- Windows/macOS: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

#### 2. Install NVIDIA Container Toolkit

```bash
# Configure the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lora_craft.git
cd lora_craft
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional)
nano .env
```

**Key settings to customize:**
- `FLASK_SECRET_KEY`: Generate a secure key for production
- `CUDA_VISIBLE_DEVICES`: Select which GPU(s) to use
- `HF_TOKEN`: Your HuggingFace token (if using private models)

### 3. Start the Application

```bash
# Build and start the container
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### 4. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

### 5. Stop the Application

```bash
# Stop containers (preserves data)
docker compose down

# Stop and remove all data (WARNING: deletes volumes)
docker compose down -v
```

---

## Configuration

### Environment Variables

All configuration is managed through the `.env` file. See `.env.example` for a complete list of options.

**Essential variables:**

```bash
# Flask configuration
FLASK_SECRET_KEY=your-secret-key-here
PORT=5000

# GPU selection
CUDA_VISIBLE_DEVICES=0

# HuggingFace configuration
HF_TOKEN=hf_your_token_here  # Optional, for private models
HF_HOME=/app/cache/huggingface
```

### Docker Compose Customization

Edit `docker-compose.yml` for advanced configuration:

**Limit GPU usage to specific devices:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1']  # Use only GPU 0 and 1
          capabilities: [gpu]
```

**Adjust memory limits:**
```yaml
mem_limit: 64g      # Maximum RAM usage
shm_size: 32g       # Shared memory for PyTorch
```

**Enable development mode (hot reload):**
```yaml
volumes:
  - ./core:/app/core
  - ./routes:/app/routes
  - ./services:/app/services
environment:
  - FLASK_DEBUG=true
```

---

## GPU Setup

### Verifying GPU Access

After starting the container, verify GPU is accessible:

```bash
# Check GPU from container
docker compose exec lora-craft nvidia-smi

# Check PyTorch CUDA availability
docker compose exec lora-craft python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Multi-GPU Configuration

**Use all GPUs:**
```yaml
# docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Use specific GPUs:**
```bash
# .env
CUDA_VISIBLE_DEVICES=0,1,2  # Use first 3 GPUs
```

### GPU Memory Management

**Monitor GPU usage:**
```bash
# Real-time GPU monitoring
watch -n 1 docker compose exec lora-craft nvidia-smi

# Check from host
nvidia-smi -l 1
```

**Optimize memory:**
- Reduce batch size in training config
- Enable gradient checkpointing
- Use smaller models (0.6B-1.7B instead of 7B-8B)

---

## Volume Management

### Understanding Volumes

LoRA Craft uses both bind mounts and named volumes:

**Bind mounts** (host directory → container):
- `./outputs` → `/app/outputs` (trained models)
- `./exports` → `/app/exports` (GGUF exports)
- `./configs` → `/app/configs` (training configs)
- `./uploads` → `/app/uploads` (uploaded datasets)
- `./logs` → `/app/logs` (application logs)

**Named volumes** (managed by Docker):
- `huggingface-cache` → `/app/cache/huggingface` (HF models)
- `transformers-cache` → `/app/cache/transformers`
- `datasets-cache` → `/app/cache/datasets`
- `torch-cache` → `/app/cache/torch`

### Backup Data

**Backup all persistent data:**
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup bind mounts
tar czf backups/$(date +%Y%m%d)/outputs.tar.gz outputs/
tar czf backups/$(date +%Y%m%d)/exports.tar.gz exports/
tar czf backups/$(date +%Y%m%d)/configs.tar.gz configs/

# Backup Docker volumes
docker run --rm \
  -v lora-craft_huggingface-cache:/source \
  -v $(pwd)/backups/$(date +%Y%m%d):/backup \
  alpine tar czf /backup/huggingface-cache.tar.gz -C /source .
```

### Restore Data

```bash
# Restore bind mounts
tar xzf backups/20250108/outputs.tar.gz
tar xzf backups/20250108/exports.tar.gz
tar xzf backups/20250108/configs.tar.gz

# Restore Docker volumes
docker run --rm \
  -v lora-craft_huggingface-cache:/target \
  -v $(pwd)/backups/20250108:/backup \
  alpine sh -c "cd /target && tar xzf /backup/huggingface-cache.tar.gz"
```

### Clean Up Space

```bash
# Remove stopped containers
docker compose down

# Remove unused images
docker image prune -a

# Remove unused volumes (WARNING: may delete data)
docker volume prune

# Clean up old model checkpoints
docker compose exec lora-craft find /app/outputs -name "checkpoint-*" -type d -mtime +30 -exec rm -rf {} +
```

---

## Production Deployment

### Security Hardening

**1. Change default secret key:**
```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_hex(32))" >> .env

# Edit .env and set FLASK_SECRET_KEY to the generated value
```

**2. Restrict CORS origins:**
```bash
# .env
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

**3. Disable debug mode:**
```bash
# .env
FLASK_DEBUG=false
```

**4. Use HTTPS with reverse proxy:**

Example Nginx configuration:
```nginx
server {
    listen 443 ssl http2;
    server_name lora-craft.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### High Availability

**Using Docker Swarm:**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml lora-craft

# Scale services (limited by GPU availability)
docker service scale lora-craft_app=2
```

### Monitoring & Logging

**Centralized logging with ELK stack:**
```yaml
# docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "5"
    labels: "production"
```

**Health monitoring:**
```bash
# Check health status
docker compose ps

# View health check logs
docker inspect lora-craft | jq '.[0].State.Health'
```

### Automated Backups

**Cron job for daily backups:**
```bash
# Add to crontab
0 2 * * * /path/to/backup-script.sh

# backup-script.sh
#!/bin/bash
cd /path/to/lora_craft
mkdir -p backups/$(date +\%Y\%m\%d)
tar czf backups/$(date +\%Y\%m\%d)/outputs.tar.gz outputs/
tar czf backups/$(date +\%Y\%m\%d)/configs.tar.gz configs/
find backups/ -type f -mtime +30 -delete  # Clean old backups
```

---

## Troubleshooting

### Common Issues

#### GPU Not Detected

**Symptoms:**
- "CUDA not available" error
- Training runs on CPU (very slow)

**Solutions:**
```bash
# 1. Verify NVIDIA driver on host
nvidia-smi

# 2. Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 3. Restart Docker daemon
sudo systemctl restart docker

# 4. Check docker-compose.yml has GPU config
grep -A 5 "deploy:" docker-compose.yml
```

#### Out of Memory Errors

**Symptoms:**
- "CUDA out of memory"
- Container crashes during training

**Solutions:**
```bash
# 1. Reduce batch size in training config
# Set batch_size to 1, increase gradient_accumulation_steps

# 2. Increase shared memory
# Edit docker-compose.yml:
shm_size: 16g

# 3. Use smaller model
# Switch from 7B to 1.7B model

# 4. Enable gradient checkpointing
# In training config UI
```

#### Container Won't Start

**Symptoms:**
- Container exits immediately
- "Port already in use" error

**Solutions:**
```bash
# 1. Check logs
docker compose logs lora-craft

# 2. Check port conflicts
sudo lsof -i :5000

# 3. Change port in .env
PORT=5001

# 4. Rebuild image
docker compose build --no-cache
docker compose up -d
```

#### Slow Model Downloads

**Symptoms:**
- HuggingFace downloads are slow
- Timeouts during model loading

**Solutions:**
```bash
# 1. Enable HF transfer acceleration
# Edit .env:
HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Pre-download models on host
# Then mount as volume
docker compose exec lora-craft huggingface-cli download unsloth/Qwen3-1.7B

# 3. Use CDN mirror (if available in your region)
```

#### WebSocket Connection Drops

**Symptoms:**
- Real-time metrics stop updating
- "Connection lost" errors

**Solutions:**
```bash
# 1. Check reverse proxy config (if using)
# Ensure WebSocket upgrade headers are set

# 2. Increase timeout
# In Nginx:
proxy_read_timeout 300s;
proxy_connect_timeout 300s;

# 3. Check firewall rules
sudo ufw allow 5000/tcp
```

---

## Performance Optimization

### Build Optimization

**Multi-stage build caching:**
```bash
# Build with cache
docker compose build

# Build without cache (clean build)
docker compose build --no-cache
```

**Use BuildKit for faster builds:**
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1
docker compose build
```

### Runtime Optimization

**1. GPU Performance:**
- Use latest NVIDIA drivers
- Enable flash attention (for supported models)
- Use mixed precision training (FP16/BF16)

**2. Disk I/O:**
```yaml
# Use tmpfs for temporary data
tmpfs:
  - /tmp:size=10G
```

**3. Network:**
- Use local HuggingFace mirror if available
- Enable HF transfer acceleration

### Resource Limits

**Balanced configuration for 16GB VRAM GPU:**
```yaml
mem_limit: 32g
memswap_limit: 32g
shm_size: 16g

deploy:
  resources:
    limits:
      memory: 32g
    reservations:
      memory: 16g
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Security Considerations

### Network Security

**1. Firewall configuration:**
```bash
# Only allow localhost access
sudo ufw deny 5000/tcp
sudo ufw allow from 127.0.0.1 to any port 5000
```

**2. Use reverse proxy:**
- Nginx or Traefik with TLS termination
- Rate limiting
- Request filtering

### Data Security

**1. Encrypt sensitive data:**
```bash
# Use encrypted volumes
docker volume create --driver local \
  --opt type=none \
  --opt o=bind,encryption=aes-xts-plain64 \
  --opt device=/encrypted/path \
  encrypted-cache
```

**2. Secure secrets:**
```bash
# Use Docker secrets instead of .env
echo "my-secret-key" | docker secret create flask_secret_key -

# Update docker-compose.yml
secrets:
  - flask_secret_key
```

### Access Control

**1. User isolation:**
```dockerfile
# Run as non-root user
RUN useradd -m -u 1000 loracraft
USER loracraft
```

**2. Read-only filesystem:**
```yaml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [LoRA Craft Main Documentation](../README.md)
- [GitHub Repository](https://github.com/yourusername/lora_craft)

---

**For issues or questions, please open an issue on GitHub.**
