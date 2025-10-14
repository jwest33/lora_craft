# ============================================================================
# LoRA Craft - Multi-Stage Dockerfile
#
# This Dockerfile creates a container for running LoRA Craft,
# a web-based GRPO fine-tuning interface for language models.
#
# Supports both GPU and CPU modes:
#   GPU MODE (default):
#     - Requires NVIDIA GPU with CUDA 12.8+ support
#     - Requires NVIDIA Container Toolkit installed on host
#     - Run with: docker run --gpus all ...
#   CPU MODE:
#     - No GPU required
#     - Run with: docker run ... (no --gpus flag needed)
#     - Training will be slower but functional
#
# Requirements:
#   - 64GB+ disk space
#   - GPU: 8GB+ VRAM (16GB+ recommended)
#   - CPU: 16GB+ RAM (32GB+ recommended)
# ============================================================================

# ============================================================================
# Stage 1: Base Image with CUDA 12.8 Runtime (GPU) or Ubuntu (CPU-capable)
# ============================================================================
# Note: This image includes CUDA support but will work on CPU-only systems
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    ca-certificates \
    sed \
    nvidia-utils-580 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Builder - Install Python Dependencies
# ============================================================================
FROM base AS builder

# Set working directory
WORKDIR /build

# Copy requirements file
COPY requirements-docker.txt .

# Install PyTorch with CUDA 12.8 support first (large dependency)
# Note: This will work on both GPU and CPU systems. PyTorch detects available hardware at runtime.
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Build xformers from source for RTX 50X (Blackwell) GPU support
# This provides compatibility with PyTorch 2.8.0 and latest GPU architectures
# Note: ninja is already installed in base image
RUN pip install --no-cache-dir -v --no-build-isolation \
    git+https://github.com/facebookresearch/xformers.git@main

# Install remaining dependencies
# Note: Some packages (unsloth, bitsandbytes, triton) are GPU-only but won't prevent CPU operation
# The application will detect GPU availability and disable these features automatically
RUN pip install --no-cache-dir -r requirements-docker.txt

# ============================================================================
# Stage 3: Runtime - Final Optimized Image
# ============================================================================
FROM base AS runtime

# Set working directory for application code
WORKDIR /app/src

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code first
COPY . .

# Copy and setup entrypoint script last (overwrites any version from COPY . .)
# Convert CRLF to LF line endings to ensure compatibility on Linux
COPY entrypoint.sh /app/src/entrypoint.sh
RUN sed -i 's/\r$//' /app/src/entrypoint.sh && chmod +x /app/src/entrypoint.sh

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/cache \
    /app/configs \
    /app/exports \
    /app/logs \
    /app/outputs \
    /app/uploads \
    /app/src/tools \
    && chmod -R 755 /app

# Clone llama.cpp for model export functionality
# Install only necessary dependencies without touching PyTorch
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /app/src/tools/llama.cpp && \
    pip install --no-cache-dir \
        numpy~=1.26.4 \
        sentencepiece~=0.2.0 \
        gguf>=0.1.0 \
        protobuf~=4.25.0 \
        mistral-common>=1.8.3

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/system/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/src/entrypoint.sh"]

# Default command
CMD ["python", "server.py"]
