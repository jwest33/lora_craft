# Docker Quick Start Guide

Fast track to running LoRA Craft with Docker.

## Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Driver** 535+ installed
3. **Docker** and **Docker Compose** installed
4. **NVIDIA Container Toolkit** installed

## One-Time Setup

### Install NVIDIA Container Toolkit (Ubuntu/Debian)

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

```bash
# Check NVIDIA driver
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
sudo systemctl restart docker
docker compose restart
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
