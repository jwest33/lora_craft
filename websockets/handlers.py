"""
SocketIO event handlers for real-time WebSocket communication.

These handlers manage client connections, session room management,
and real-time updates for training sessions.
"""

from flask import request
from flask_socketio import emit, join_room, leave_room

from utils.logging_config import get_logger
from .utils import process_training_queue

logger = get_logger(__name__)


def register_socketio_handlers(socketio, session_queues, training_sessions):
    """
    Register all SocketIO event handlers with the given SocketIO instance.

    Args:
        socketio: SocketIO instance to register handlers with
        session_queues: Dict mapping session IDs to their queues
        training_sessions: Dict mapping session IDs to TrainingSession objects
    """

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'message': 'Connected to GRPO training server'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('join_training_session')
    def handle_join_training_session(data):
        """Handle client joining a training session room."""
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            logger.info(f"Client {request.sid} joined session {session_id}")
            emit('joined_session', {'session_id': session_id})

    @socketio.on('leave_training_session')
    def handle_leave_training_session(data):
        """Handle client leaving a training session room."""
        session_id = data.get('session_id')
        if session_id:
            leave_room(session_id)
            logger.info(f"Client {request.sid} left session {session_id}")
            emit('left_session', {'session_id': session_id})

    @socketio.on('join_session')
    def handle_join_session(data):
        """Join a training session room for updates."""
        session_id = data.get('session_id')
        if session_id and session_id in training_sessions:
            join_room(session_id)
            emit('joined_session', {'session_id': session_id})
            logger.info(f"Client {request.sid} joined session {session_id}")
        else:
            emit('error', {'message': 'Invalid session ID'})

    @socketio.on('leave_session')
    def handle_leave_session(data):
        """Leave a training session room."""
        session_id = data.get('session_id')
        if session_id:
            leave_room(session_id)
            emit('left_session', {'session_id': session_id})
            logger.info(f"Client {request.sid} left session {session_id}")

    @socketio.on('request_update')
    def handle_request_update(data):
        """Handle request for training update."""
        session_id = data.get('session_id')
        if session_id in training_sessions:
            process_training_queue(session_id, session_queues, training_sessions, socketio)

    @socketio.on('join_dataset_session')
    def handle_join_dataset_session(data):
        """Join a dataset download session for updates."""
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            emit('joined_dataset_session', {'session_id': session_id})
            logger.info(f"Client {request.sid} joined dataset session {session_id}")
