#!/bin/bash
# ============================================================================
# LoRA Craft - Docker Entrypoint Script
#
# This script initializes the container environment and starts the Flask app.
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "LoRA Craft - Container Initialization"
echo "============================================================================"

# ============================================================================
# 1. Check GPU Availability
# ============================================================================
echo "[1/5] Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "- GPU detected successfully"
else
    echo "- WARNING: nvidia-smi not found. GPU may not be available!"
    echo "  Make sure NVIDIA Container Toolkit is installed on the host."
fi

# Verify CUDA availability via PyTorch
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" || true
python3 -c "import torch; print(f'CUDA Version: {torch.version.cuda}')" || true
python3 -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')" || true

# ============================================================================
# 2. Create Required Directories
# ============================================================================
echo ""
echo "[2/5] Creating required directories..."

mkdir -p \
    /app/cache/huggingface \
    /app/cache/transformers \
    /app/cache/datasets \
    /app/cache/torch \
    /app/configs \
    /app/exports \
    /app/logs \
    /app/outputs \
    /app/uploads \
    /app/src/presets \
    /app/src/static \
    /app/src/templates

echo "- Directories created"

# ============================================================================
# 3. Verify llama.cpp Installation
# ============================================================================
echo ""
echo "[3/5] Verifying llama.cpp installation..."

if [ ! -d "/app/src/tools/llama.cpp" ]; then
    echo "- llama.cpp not found, cloning repository..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /app/src/tools/llama.cpp
    cd /app/src/tools/llama.cpp
    pip install --no-cache-dir -r requirements.txt
    echo "- llama.cpp installed"
else
    echo "- llama.cpp already installed"
fi

# ============================================================================
# 4. Set Environment Variables
# ============================================================================
echo ""
echo "[4/5] Setting environment variables..."

export HF_HOME=${HF_HOME:-/app/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/app/cache/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/app/cache/datasets}
export TORCH_HOME=${TORCH_HOME:-/app/cache/torch}
export PYTHONUNBUFFERED=1

echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  Environment configured"

# ============================================================================
# 5. Display System Information
# ============================================================================
echo ""
echo "[5/5] System Information:"
echo "  Python Version: $(python --version)"
echo "  PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  CUDA Version: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
echo "  Working Directory: $(pwd)"
echo "  User: $(whoami)"

# ============================================================================
# 6. Start Application
# ============================================================================
echo ""
echo "============================================================================"
echo "Starting LoRA Craft Flask Application"
echo "============================================================================"
echo ""

# Execute the command passed to the container
exec "$@"
