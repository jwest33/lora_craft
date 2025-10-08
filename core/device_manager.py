"""
Device Detection and Management Module

This module provides centralized device detection for CPU/GPU selection.
It checks for CUDA availability and determines whether Unsloth optimizations
can be used, providing a consistent interface for the entire application.
"""

import os
import warnings
import torch

# Global device configuration
_device = None
_is_cuda_available = None
_use_unsloth = None
_device_info = {}


def detect_device():
    """
    Detect available compute device (CUDA GPU or CPU).

    Returns:
        dict: Device information including:
            - device: torch.device object
            - is_cuda_available: bool
            - use_unsloth: bool
            - device_name: str
            - device_count: int
    """
    global _device, _is_cuda_available, _use_unsloth, _device_info

    # Return cached result if already detected
    if _device is not None:
        return _device_info

    # Check CUDA availability
    _is_cuda_available = torch.cuda.is_available()

    if _is_cuda_available:
        # CUDA is available - use GPU
        _device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()

        # Check if Unsloth can be used (requires CUDA)
        # We'll try importing it to verify
        try:
            import unsloth
            _use_unsloth = True
            unsloth_status = "available"
        except ImportError:
            _use_unsloth = False
            unsloth_status = "not installed"
        except Exception as e:
            _use_unsloth = False
            unsloth_status = f"import failed: {e}"
            warnings.warn(f"Unsloth import failed despite CUDA availability: {e}")

        _device_info = {
            'device': _device,
            'is_cuda_available': True,
            'use_unsloth': _use_unsloth,
            'device_name': device_name,
            'device_count': device_count,
            'mode': 'GPU',
            'unsloth_status': unsloth_status
        }

        print(f"[OK] GPU detected: {device_name}")
        print(f"[OK] CUDA available: {torch.version.cuda}")
        if _use_unsloth:
            print(f"[OK] Unsloth optimizations: ENABLED")
        else:
            print(f"[WARN] Unsloth optimizations: DISABLED ({unsloth_status})")
    else:
        # No CUDA - fallback to CPU
        _device = torch.device("cpu")
        _use_unsloth = False

        _device_info = {
            'device': _device,
            'is_cuda_available': False,
            'use_unsloth': False,
            'device_name': 'CPU',
            'device_count': 0,
            'mode': 'CPU',
            'unsloth_status': 'unavailable (no CUDA)'
        }

        print(f"[WARN] No GPU detected - running in CPU mode")
        print(f"[WARN] Unsloth optimizations: DISABLED (requires CUDA)")
        print(f"[WARN] Training will be slower on CPU")

    return _device_info


def get_device():
    """
    Get the current compute device.

    Returns:
        torch.device: The device to use for computations
    """
    if _device is None:
        detect_device()
    return _device


def is_cuda_available():
    """
    Check if CUDA is available.

    Returns:
        bool: True if CUDA is available
    """
    if _is_cuda_available is None:
        detect_device()
    return _is_cuda_available


def use_unsloth():
    """
    Check if Unsloth optimizations should be used.

    Returns:
        bool: True if Unsloth can be used
    """
    if _use_unsloth is None:
        detect_device()
    return _use_unsloth


def get_device_info():
    """
    Get detailed device information.

    Returns:
        dict: Complete device information
    """
    if _device is None:
        detect_device()
    return _device_info


def get_optimal_device_map(model_name: str = None):
    """
    Get optimal device map for model loading.

    Args:
        model_name: Optional model name for specific optimizations

    Returns:
        str or dict: Device map configuration for model loading
    """
    if is_cuda_available():
        # Use "auto" for GPU - HuggingFace will distribute across available GPUs
        return "auto"
    else:
        # Use CPU explicitly
        return "cpu"


# Note: Device detection is lazy - it runs on first access to any device function
# This prevents issues with encoding errors during module import


if __name__ == "__main__":
    # Test the device detection
    info = get_device_info()
    print("\n" + "=" * 60)
    print("Device Detection Test")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 60)
