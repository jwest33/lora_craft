"""
WebSocket layer for real-time communication.

This module provides WebSocket functionality for real-time updates
during training sessions, including:
- Event handlers for client connections and room management
- Queue processing utilities for training progress
- Periodic background tasks for queue processing
"""

from .handlers import register_socketio_handlers
from .utils import (
    emit_to_session,
    process_training_queue,
    periodic_queue_processor,
)

__all__ = [
    'register_socketio_handlers',
    'emit_to_session',
    'process_training_queue',
    'periodic_queue_processor',
]
