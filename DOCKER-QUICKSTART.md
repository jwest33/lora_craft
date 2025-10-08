# Docker Quick Start Guide

Fast track to running LoRA Craft with Docker.

## Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Driver** 535+ installed
3. **Docker** 20.10+ and **Docker Compose** 2.0+ installed
4. **NVIDIA Container Toolkit** installed

## One-Time Setup

### Linux (Ubuntu/Debian)

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Linux (RHEL/CentOS/Fedora)

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install
sudo yum install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Windows (Docker Desktop with WSL2)

1. **Install WSL2**:
   ```powershell
   wsl --install
   ```

2. **Install Docker Desktop**:
   - Download from [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
   - Enable WSL2 integration in Docker Desktop settings

3. **Install NVIDIA Driver** (on Windows host):
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
   - Driver 535+ required
   - No need to install CUDA Toolkit separately

4. **Verify GPU access**:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

**Note**: Docker Desktop automatically includes NVIDIA Container Toolkit support when using WSL2 backend.

### macOS

**GPU acceleration is not supported on macOS** due to lack of NVIDIA GPU support. You can run LoRA Craft in CPU-only mode, but training will be significantly slower and memory-limited.

For macOS users, we recommend:
- Using a cloud GPU instance (AWS, GCP, RunPod, etc.)
- Remote development to a Linux machine with GPU

## Launch LoRA Craft

```bash
# Clone repository
git clone https://github.com/jwest33/lora_craft.git
cd lora_craft

# Optional: Configure environment
cp .env.example .env
# Edit .env if needed

# Start application
docker compose up -d

# View logs
docker compose logs -f
```

**Access:** http://localhost:5000

## Common Commands

```bash
# Start
docker compose up -d

# Stop (preserves data)
docker compose down

# View logs
docker compose logs -f

# Restart
docker compose restart

# Check status
docker compose ps

# Execute command in container
docker compose exec lora-craft bash

# Check GPU
docker compose exec lora-craft nvidia-smi

# Update to latest version
git pull
docker compose build --no-cache
docker compose up -d
```

## Troubleshooting

### GPU Not Detected

**Symptoms**: Container logs show "CUDA Available: False" or "GPU Count: 0"

**Linux Solutions:**
```bash
# Check NVIDIA driver on host
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
docker compose restart
```

**Windows/Docker Desktop Solutions:**

1. **Verify GPU access works** with test container:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Check docker-compose.yml configuration**:
   - Ensure `runtime: nvidia` is set (required for Docker Desktop)
   - The `deploy` section doesn't work reliably on Docker Desktop

3. **Restart Docker Desktop**:
   - Right-click Docker Desktop tray icon → Restart

4. **Check WSL2 integration**:
   - Docker Desktop → Settings → Resources → WSL Integration
   - Ensure your WSL2 distro is enabled

### Container Won't Start / Entrypoint Error

**Symptoms**: "exec /app/src/entrypoint.sh: no such file or directory"

**Cause**: Line ending issues when building on Windows

**Solution**:
The Dockerfile now automatically fixes line endings. If you still see this error:

```bash
# Rebuild image without cache
docker compose build --no-cache

# Restart container
docker compose up -d
```

**Alternative (manual fix)**:
```bash
# Convert line endings in entrypoint.sh
dos2unix entrypoint.sh

# Or use Git to normalize line endings
git config core.autocrlf input
git rm --cached -r .
git reset --hard
```

### Port 5000 Already in Use

```bash
# Option 1: Find and stop conflicting process
sudo lsof -i :5000
sudo kill <PID>

# Option 2: Use different port
echo "PORT=5001" >> .env
docker compose up -d
```

### Container Crashes

```bash
# View logs
docker compose logs lora-craft

# Check resources
docker stats lora-craft

# Rebuild from scratch
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Data Locations

All data is stored in the project directory:

- `outputs/` - Trained models
- `exports/` - GGUF exports
- `configs/` - Saved configurations
- `uploads/` - Uploaded datasets
- `logs/` - Application logs

**Backup:** Simply copy these directories

## Resource Requirements

### Minimum
- 8GB VRAM GPU
- 16GB RAM
- 50GB disk space

### Recommended
- 16GB+ VRAM GPU
- 32GB RAM
- 100GB disk space

## What's Included in the Docker Image

The LoRA Craft Docker image provides a complete, pre-configured environment:

- **NVIDIA CUDA 12.8** runtime with cuDNN
- **Python 3.11** with all dependencies pre-installed
- **nvidia-smi** utility for GPU monitoring
- **Automatic GPU detection** on container startup
- **Persistent volumes** for models, datasets, and outputs
- **Health checks** to monitor application status
- **Optimized dependencies** (PyTorch 2.8.0 with CUDA support)

**Image size**: ~15GB (includes PyTorch, Transformers, and training libraries)

## GPU Configuration Notes

LoRA Craft supports two GPU configuration methods in docker-compose.yml:

### Method 1: Runtime (Recommended for Docker Desktop)
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

**Use when:**
- Running on Docker Desktop (Windows/macOS)
- Simpler configuration needed
- Having issues with `deploy` section

### Method 2: Deploy Resources
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Use when:**
- Running on Linux with Docker Compose v2+
- Need fine-grained GPU control (specific GPU selection)
- Running in swarm mode

**Current configuration**: LoRA Craft uses Method 1 (`runtime: nvidia`) for maximum compatibility.

## Getting Help

- **Detailed docs:** [docs/DOCKER.md](docs/DOCKER.md)
- **Main README:** [README.md](README.md)
- **Issues:** https://github.com/jwest33/lora_craft/issues

## Security Notes

**For production deployments:**

1. Change `FLASK_SECRET_KEY` in `.env`:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. Use reverse proxy (Nginx/Traefik) with HTTPS

3. Restrict CORS origins in `.env`:
   ```bash
   CORS_ORIGINS=https://yourdomain.com
   ```

4. Keep Docker images updated:
   ```bash
   docker compose pull
   docker compose up -d
   ```
