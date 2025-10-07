"""
WebSocket utility functions for queue processing and event emission.

These functions handle the real-time communication between training sessions
and connected clients via SocketIO.
"""

import gc
import queue
from datetime import datetime
from typing import Any, Dict
import torch

from utils.logging_config import get_logger

logger = get_logger(__name__)


def emit_to_session(socketio, session_id: str, event: str, data: Any):
    """
    Emit event to specific session via SocketIO.

    Args:
        socketio: SocketIO instance
        session_id: Unique session identifier
        event: Event name to emit
        data: Data to send with the event
    """
    # Add session_id to data for client-side filtering
    data_with_id = data.copy() if isinstance(data, dict) else {'data': data}
    data_with_id['session_id'] = session_id

    # Broadcast to all clients (they'll filter by session_id)
    # This ensures updates work even if room joining hasn't completed
    socketio.emit(event, data_with_id)


def process_training_queue(session_id: str, session_queues: Dict, training_sessions: Dict, socketio):
    """
    Process messages from training queue and emit to client.

    This function reads messages from the training queue for a specific session
    and emits appropriate SocketIO events based on the message type. It handles:
    - Progress updates
    - Training metrics
    - Reward samples
    - Log messages
    - Completion/error states

    Args:
        session_id: Unique session identifier
        session_queues: Dict mapping session IDs to their queues
        training_sessions: Dict mapping session IDs to TrainingSession objects
        socketio: SocketIO instance for emitting events
    """
    if session_id not in session_queues:
        return

    q = session_queues[session_id]
    session_obj = training_sessions.get(session_id)

    try:
        while True:
            try:
                msg_type, msg_data = q.get_nowait()

                if msg_type == 'progress':
                    if session_obj:
                        session_obj.metrics['progress'] = msg_data
                        session_obj.progress = msg_data  # Store progress for reconnection
                    emit_to_session(socketio, session_id, 'training_progress', {
                        'progress': msg_data,
                        'session_id': session_id
                    })

                elif msg_type == 'metrics':
                    if session_obj:
                        # Detect phase transitions (pre-training -> training)
                        current_phase = msg_data.get('training_phase')
                        previous_phase = getattr(session_obj, 'current_training_phase', None)

                        # If transitioning to GRPO training phase, reset metrics
                        if current_phase == 'training' and previous_phase != 'training':
                            logger.info(f"Transitioning to GRPO training phase - resetting metrics for session {session_id}")
                            session_obj.step_counter = 0
                            session_obj.current_training_phase = 'training'
                            # Emit reset event to frontend
                            emit_to_session(socketio, session_id, 'reset_metrics', {
                                'phase': 'training',
                                'session_id': session_id
                            })
                        elif current_phase:
                            session_obj.current_training_phase = current_phase

                        # If no step provided or step is 0, use and increment our step counter
                        if not msg_data.get('step'):
                            session_obj.step_counter += 1
                            msg_data['step'] = session_obj.step_counter

                        session_obj.metrics.update(msg_data)
                        # Update current_step if present
                        if 'step' in msg_data and msg_data['step'] > 0:
                            session_obj.metrics['current_step'] = msg_data['step']
                            # Debug: Log that we're processing metrics
                            logger.debug(f"Processing metrics for session {session_id}, step {msg_data['step']}: loss={msg_data.get('loss', 'N/A')}, reward={msg_data.get('mean_reward', 'N/A')}")
                        # Store metrics history for reconnection
                        session_obj.metrics_history.append(msg_data.copy())

                    # Debug: Log all metric keys before emitting
                    logger.info(f"Emitting metrics with keys: {list(msg_data.keys())}")
                    logger.debug(f"Sample values - kl: {msg_data.get('kl', 'N/A')}, epoch: {msg_data.get('epoch', 'N/A')}, completions/mean_length: {msg_data.get('completions/mean_length', 'N/A')}")

                    # Emit to frontend via socket.io
                    emit_to_session(socketio, session_id, 'training_metrics', {
                        **msg_data,
                        'session_id': session_id
                    })
                    logger.debug(f"Emitted training_metrics event for session {session_id}")

                elif msg_type == 'reward_sample':
                    # Emit reward sample for real-time analysis
                    emit_to_session(socketio, session_id, 'reward_sample', {
                        **msg_data,
                        'session_id': session_id
                    })
                    logger.debug(f"Emitted reward_sample event for session {session_id}")

                elif msg_type == 'log':
                    if session_obj:
                        session_obj.logs.append(msg_data)
                    emit_to_session(socketio, session_id, 'training_log', {'message': msg_data})

                elif msg_type == 'complete':
                    if session_obj:
                        session_obj.status = 'completed'
                        session_obj.completed_at = datetime.now()

                    # Clean up training models from memory
                    try:
                        # Clear any Unsloth models that might be in GPU memory
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"Cleared GPU memory after training completion for session {session_id}")
                    except Exception as e:
                        logger.warning(f"Error clearing GPU memory: {e}")

                    emit_to_session(socketio, session_id, 'training_complete', {
                        'message': msg_data,
                        'session_id': session_id
                    })

                elif msg_type == 'error':
                    if session_obj:
                        session_obj.status = 'error'
                        session_obj.logs.append(f"ERROR: {msg_data}")
                    emit_to_session(socketio, session_id, 'training_error', {
                        'error': msg_data,
                        'session_id': session_id
                    })

            except queue.Empty:
                break
    except Exception as e:
        logger.error(f"Error processing queue for session {session_id}: {e}")


def periodic_queue_processor(session_queues: Dict, training_sessions: Dict, socketio):
    """
    Process all training queues periodically in a background task.

    This function runs continuously in a background thread, processing
    messages from all active training session queues every 500ms.

    Args:
        session_queues: Dict mapping session IDs to their queues
        training_sessions: Dict mapping session IDs to TrainingSession objects
        socketio: SocketIO instance for sleeping between iterations
    """
    while True:
        try:
            for session_id in list(session_queues.keys()):
                process_training_queue(session_id, session_queues, training_sessions, socketio)
            socketio.sleep(0.5)  # Process every 500ms
        except Exception as e:
            logger.error(f"Error in queue processor: {e}")
            socketio.sleep(1)
