"""
Training routes for model training session management.

This module provides endpoints for:
- Starting and stopping training sessions
- Monitoring training status and progress
- Accessing training logs and metrics
- Managing training history
"""

import gc
import threading
from flask import Blueprint, request, jsonify, current_app
import torch

from services import TrainingSession, create_session_id, run_training
from utils.logging_config import get_logger
from utils.validators import validate_training_config

logger = get_logger(__name__)

# Create blueprint
training_bp = Blueprint('training', __name__, url_prefix='/api/training')


@training_bp.route('/start', methods=['POST'])
def start_training():
    """Start a new training session."""
    try:
        config = request.json

        # Normalize dataset path - preserve forward slashes for HuggingFace datasets
        dataset_path = config.get('dataset_path', '')
        if dataset_path:
            # Check if this is a HuggingFace dataset (not a local file path)
            # HuggingFace datasets don't start with . / C: or contain uploads
            if not any(dataset_path.startswith(prefix) for prefix in ('.', '/', 'C:', 'uploads')) and '\\uploads\\' not in dataset_path:
                # This looks like a HuggingFace dataset (e.g., 'tatsu-lab/alpaca')
                # Ensure forward slashes are preserved (Windows may convert them)
                config['dataset_path'] = dataset_path.replace('\\', '/')
                logger.info(f"Normalized HuggingFace dataset path: {config['dataset_path']}")

        # Validate configuration
        valid, errors = validate_training_config(config)
        if not valid:
            return jsonify({'error': 'Invalid configuration', 'errors': errors}), 400

        # Get global state from current_app
        training_sessions = current_app.training_sessions
        session_queues = current_app.session_queues
        training_threads = current_app.training_threads
        system_config = current_app.system_config
        session_registry = current_app.session_registry
        socketio = current_app.socketio
        upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')

        # Clean up any stale training sessions before starting new one
        # This helps prevent the hanging issue when training multiple models
        for sid, session in list(training_sessions.items()):
            if session.status in ['completed', 'error', 'stopped']:
                # Clean up trainer if it exists
                if hasattr(session, 'trainer') and session.trainer:
                    try:
                        if hasattr(session.trainer, 'cleanup'):
                            session.trainer.cleanup()
                        del session.trainer
                    except:
                        pass
                    finally:
                        session.trainer = None

        # Force garbage collection before starting new training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create new session
        session_id = create_session_id()
        session_obj = TrainingSession(session_id, config)

        # Store session
        training_sessions[session_id] = session_obj
        session_queues[session_id] = session_obj.queue

        # Start training in background thread
        thread = threading.Thread(
            target=run_training,
            args=(session_id, config, training_sessions, session_queues,
                  system_config, session_registry, socketio, upload_folder),
            daemon=True
        )
        thread.start()
        training_threads[session_id] = thread

        return jsonify({
            'success': True,
            'session_id': session_id,
            'status': 'started',
            'message': 'Training started successfully'
        })

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/<session_id>/status', methods=['GET'])
def get_training_status(session_id):
    """Get status of a training session."""
    training_sessions = current_app.training_sessions

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_obj = training_sessions[session_id]
    return jsonify({'success': True, 'status': session_obj.to_dict()})


@training_bp.route('/<session_id>/stop', methods=['POST'])
def stop_training(session_id):
    """Stop a training session."""
    training_sessions = current_app.training_sessions

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    try:
        session_obj = training_sessions[session_id]
        session_obj.status = 'stopping'

        # Signal trainer to stop if available
        if session_obj.trainer:
            # Trainer should have a stop method
            pass

        return jsonify({
            'session_id': session_id,
            'status': 'stopping',
            'message': 'Training stop requested'
        })
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/<session_id>/metrics', methods=['GET'])
def get_training_metrics(session_id):
    """Get current metrics for a training session."""
    training_sessions = current_app.training_sessions

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_obj = training_sessions[session_id]
    return jsonify(session_obj.metrics)


@training_bp.route('/<session_id>/logs', methods=['GET'])
def get_training_logs(session_id):
    """Get logs for a training session."""
    training_sessions = current_app.training_sessions

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_obj = training_sessions[session_id]
    limit = request.args.get('limit', 100, type=int)
    return jsonify({'logs': session_obj.logs[-limit:]})


@training_bp.route('/sessions', methods=['GET'])
def list_training_sessions():
    """List all training sessions."""
    training_sessions = current_app.training_sessions

    sessions = []
    for session_id, session_obj in training_sessions.items():
        sessions.append({
            'session_id': session_id,
            'status': session_obj.status,
            'created_at': session_obj.created_at.isoformat(),
            'model': session_obj.config.get('model_name', 'Unknown'),
            'display_name': session_obj.display_name
        })
    return jsonify(sessions)


@training_bp.route('/session/<session_id>/history', methods=['GET'])
def get_training_history(session_id):
    """Get training history for reconnection."""
    training_sessions = current_app.training_sessions

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_obj = training_sessions[session_id]

    # Gather historical data
    history = {
        'session_id': session_id,
        'status': session_obj.status,
        'progress': getattr(session_obj, 'progress', 0),
        'logs': session_obj.logs[-100:],  # Last 100 logs
        'metrics': getattr(session_obj, 'metrics_history', [])
    }

    return jsonify(history)


@training_bp.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Manually cleanup model resources for a session to free memory."""
    training_sessions = current_app.training_sessions

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    try:
        session_obj = training_sessions[session_id]
        if hasattr(session_obj, 'trainer') and session_obj.trainer is not None:
            session_obj.trainer.cleanup()
            return jsonify({
                'success': True,
                'message': 'Model resources cleaned up successfully'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No resources to cleanup'
            })
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return jsonify({'error': str(e)}), 500


# Legacy route aliases are now handled in app_factory.py for better compatibility
