"""
System routes for system information and status.

This module provides endpoints for:
- System hardware information (CPU, GPU, RAM, VRAM)
- Real-time system status monitoring
"""

import os
import sys
from flask import Blueprint, jsonify
import torch
import psutil

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create blueprint
system_bp = Blueprint('system', __name__, url_prefix='/api')


@system_bp.route('/system/info', methods=['GET'])
def get_system_info():
    """Get comprehensive system information."""
    try:
        info = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cpu_count': os.cpu_count(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'platform': sys.platform
        }

        # Add GPU memory info if available
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            total_vram = gpu_props.total_memory / 1024**3  # GB
            allocated_vram = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved_vram = torch.cuda.memory_reserved(0) / 1024**3  # GB
            free_vram = total_vram - reserved_vram

            info['gpu_memory_total'] = total_vram
            info['gpu_memory_allocated'] = allocated_vram
            info['gpu_memory_free'] = free_vram
            info['gpu_memory_reserved'] = reserved_vram

        # Add system RAM info using psutil
        memory = psutil.virtual_memory()
        info['ram_total'] = memory.total / 1024**3  # GB
        info['ram_available'] = memory.available / 1024**3  # GB
        info['ram_used'] = memory.used / 1024**3  # GB
        info['ram_percent'] = memory.percent

        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500


@system_bp.route('/system_status', methods=['GET'])
def get_system_status():
    """Get real-time system status including GPU, VRAM, and RAM information."""
    try:
        status = {
            'gpu': 'Unknown',
            'vram': 'N/A',
            'ram': 'N/A',
            'cpu': 'N/A',
            'vram_percent': 0,
            'ram_percent': 0
        }

        # Get GPU info if available
        if torch.cuda.is_available():
            status['gpu'] = torch.cuda.get_device_name(0)

            # Try to get actual VRAM usage from GPU hardware using pynvml
            vram_acquired = False
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get actual hardware memory usage
                used = mem_info.used / 1024**3  # Convert to GB
                total = mem_info.total / 1024**3
                status['vram'] = f"{used:.1f}GB / {total:.1f}GB"
                status['vram_percent'] = (used / total * 100) if total > 0 else 0
                vram_acquired = True

                pynvml.nvmlShutdown()
            except (ImportError, Exception) as e:
                # pynvml not available or error, fall back to torch
                pass

            # Fallback to PyTorch memory tracking if pynvml failed
            if not vram_acquired:
                allocated = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                # Use reserved as a proxy for "used" since it's closer to actual usage
                status['vram'] = f"{reserved:.1f}GB / {total:.1f}GB"
                status['vram_percent'] = (reserved / total * 100) if total > 0 else 0
        else:
            status['gpu'] = 'CPU Only'

        # Get RAM info
        ram = psutil.virtual_memory()
        status['ram'] = f"{ram.used / 1024**3:.1f}GB / {ram.total / 1024**3:.1f}GB"
        status['ram_percent'] = ram.percent

        # Get CPU usage
        status['cpu'] = f"{psutil.cpu_percent()}%"

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'gpu': 'Error',
            'vram': 'Error',
            'ram': 'Error',
            'cpu': 'Error',
            'vram_percent': 0,
            'ram_percent': 0
        })


@system_bp.route('/active_sessions', methods=['GET'])
def get_active_sessions():
    """Get list of active training sessions."""
    try:
        from flask import current_app
        training_sessions = current_app.training_sessions

        # Get active sessions
        active = []
        for session_id, session_data in training_sessions.items():
            if session_data.status == 'running':
                active.append({
                    'id': session_id,
                    'name': session_data.display_name,
                    'status': 'running',
                    'progress': getattr(session_data, 'progress', 0),
                    'start_time': session_data.created_at.isoformat() if hasattr(session_data, 'created_at') and session_data.created_at else None
                })
        return jsonify({'sessions': active})
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        return jsonify({'sessions': []})
