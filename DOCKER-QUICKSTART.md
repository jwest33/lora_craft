# Docker Quick Start Guide

Fast track to running LoRA Craft with Docker.

## Prerequisites

### Required (All Modes)
- **Docker** 20.10+ and **Docker Compose** 2.0+ installed

### For GPU Mode (Optional, but Recommended)
- **NVIDIA GPU** with CUDA support
- **NVIDIA Driver** 535+ installed
- **NVIDIA Container Toolkit** installed

### For CPU-Only Mode
- No GPU required - Docker is sufficient
- **Note:** Training will be 5-10x slower on CPU
- Recommended: 16GB+ RAM (32GB+ preferred)

## One-Time Setup (GPU Mode Only)

**Skip this section if you're running in CPU-only mode.** The NVIDIA Container Toolkit is only needed for GPU acceleration.

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

**GPU acceleration is not supported on macOS** due to lack of NVIDIA GPU support.

However, **LoRA Craft now supports CPU-only mode** and will run on macOS:
- Training will be 5-10x slower than GPU mode
- Recommended for development, testing, or small-scale training
- Use smaller models (e.g., Qwen3-0.6B) and reduce batch sizes

**For production training on macOS**, we recommend:
- Using a cloud GPU instance (AWS, GCP, RunPod, vast.ai, etc.)
- Remote development to a Linux machine with GPU
- Docker Desktop with macOS native installation (no GPU setup needed for CPU mode)

## Launch LoRA Craft

```bash
# Clone repository
git clone https://github.com/jwest33/lora_craft.git
cd lora_craft

# Optional: Configure environment
cp .env.example .env
# Edit .env if needed

# Start application (GPU mode - default)
docker compose up -d

# OR: Start in CPU-only mode (omit --gpus flag in docker-compose.yml)
# Edit docker-compose.yml and comment out the 'runtime: nvidia' line
# Then run: docker compose up -d

# View logs
docker compose logs -f
```

**Access:** http://localhost:5000

### How to Check Which Mode You're Running

Once started, check the logs to see if GPU was detected:

```bash
docker compose logs lora-craft | grep -i "gpu\|cuda\|cpu mode"
```

**GPU Mode Output:**
```
+ GPU detected: NVIDIA GeForce RTX 4090
+ CUDA available: 12.8
+ Unsloth optimizations: ENABLED
```

**CPU Mode Output:**
```
x No GPU detected - running in CPU mode
x Unsloth optimizations: DISABLED (requires CUDA)
x Training will be slower on CPU
```

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

### GPU Not Detected (But You Have a GPU)

**Symptoms**: Container logs show "CPU Mode" or "CUDA Available: False" even though you have an NVIDIA GPU

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

### Running in CPU Mode (Intentionally)

**To force CPU-only mode** (useful for testing or when GPU not needed):

1. Edit `docker-compose.yml` and comment out the GPU runtime:
   ```yaml
   # runtime: nvidia  # Comment this out for CPU mode
   ```

2. Restart the container:
   ```bash
   docker compose down
   docker compose up -d
   ```

3. Verify CPU mode in logs:
   ```bash
   docker compose logs lora-craft | grep "CPU mode"
   ```

### Out of Memory on CPU

**Symptoms**: Container crashes with "Killed" or OOM errors in CPU mode

**Solutions:**

1. **Use smaller models**:
   - Try Qwen3-0.6B instead of larger models
   - Reduce max_sequence_length to 512 or 256

2. **Reduce batch size**:
   - Set batch_size to 1
   - Reduce gradient_accumulation_steps

3. **Increase Docker memory limit** (Docker Desktop):
   - Docker Desktop → Settings → Resources
   - Increase Memory limit to 16GB+ (32GB recommended)

4. **Close other applications** to free system RAM

5. **Use smaller datasets**:
   - Limit dataset to < 1000 samples for testing

## Data Locations

All data is stored in the project directory:

- `outputs/` - Trained models
- `exports/` - GGUF exports
- `configs/` - Saved configurations
- `uploads/` - Uploaded datasets
- `logs/` - Application logs

**Backup:** Simply copy these directories

## Resource Requirements

### GPU Mode (Recommended)

**Minimum:**
- 8GB VRAM GPU
- 16GB RAM
- 50GB disk space

**Recommended:**
- 16GB+ VRAM GPU
- 32GB RAM
- 100GB disk space

### CPU-Only Mode

**Minimum:**
- 16GB RAM
- 50GB disk space
- Modern multi-core CPU (4+ cores)

**Recommended:**
- 32GB+ RAM
- 100GB disk space
- 8+ core CPU (Ryzen 7/9, Intel i7/i9, or equivalent)

**Note:** CPU mode is 5-10x slower than GPU. Best for:
- Development and testing
- Small models (Qwen3-0.6B, Qwen3-1.7B)
- Limited datasets (< 1000 samples)
- Systems without NVIDIA GPU

## What's Included in the Docker Image

The LoRA Craft Docker image provides a complete, pre-configured environment:

- **NVIDIA CUDA 12.8** runtime with cuDNN 9.7 (works on both GPU and CPU)
- **Python 3.11** with all dependencies pre-installed
- **PyTorch 2.8.0** with CUDA 12.8 support
- **nvidia-smi** utility for GPU monitoring (when GPU available)
- **Automatic GPU/CPU detection** on startup
- **CPU fallback** when GPU not available
- **Persistent volumes** for models, datasets, configs, and outputs
- **Health checks** to monitor application status
- **Optimized training libraries** (Unsloth, Transformers, PEFT, TRL)

**Image size**: ~20GB (includes PyTorch, Transformers, and training libraries)

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
