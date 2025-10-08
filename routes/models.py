"""
Model routes for model testing, comparison, and management.

This module provides endpoints for:
- Listing available and trained models
- Model loading and testing
- Model comparison (single and batch)
- Test history tracking
- Batch comparison operations
"""

import os
import json
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from core import ModelTester, SessionRegistry
from core.test_history import get_test_history_manager
from core.batch_tester import get_batch_test_runner
from utils.logging_config import get_logger
from constants import MODEL_DEFINITIONS

logger = get_logger(__name__)

# Create blueprint
models_bp = Blueprint('models', __name__, url_prefix='/api')

# Initialize model tester
model_tester = ModelTester()


# Model listing routes

@models_bp.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available base models."""
    # Flatten into single array for compatibility with frontend
    all_models = []
    for category, models in MODEL_DEFINITIONS.items():
        all_models.extend(models)

    return jsonify({'models': all_models})


@models_bp.route('/models/definitions', methods=['GET'])
def get_model_definitions():
    """Get model definitions organized by family for frontend use."""
    return jsonify({'modelsByFamily': MODEL_DEFINITIONS})


@models_bp.route('/models/trained', methods=['GET'])
def list_trained_models():
    """List all trained models from the session registry."""
    try:
        session_registry = current_app.session_registry

        # Get completed sessions from registry (fast O(1) lookup)
        sessions = session_registry.list_completed_sessions()

        trained_models = []
        sessions_to_cleanup = []

        for session in sessions:
            # Handle infinity values in best_reward
            best_reward = session.best_reward
            if best_reward is not None and (best_reward == float('-inf') or best_reward == float('inf')):
                best_reward = None

            # Convert SessionInfo to API response format
            model_info = {
                'session_id': session.session_id,
                'path': f"./outputs/{session.session_id}",
                'has_final_checkpoint': True,  # Registry only stores completed sessions
                'checkpoints': [],
                'created_at': session.created_at or datetime.now().isoformat(),
                'modified_at': session.completed_at or datetime.now().isoformat(),
                'model_name': session.model_name,
                'epochs': session.epochs_trained or 0,
                'best_reward': best_reward
            }

            # Check if checkpoint still exists (in case of manual deletion)
            checkpoint_path = session_registry.get_checkpoint_path(session.session_id)
            if checkpoint_path and Path(checkpoint_path).exists():
                trained_models.append(model_info)
            else:
                sessions_to_cleanup.append(session.session_id)

        # Clean up orphaned sessions
        for session_id in sessions_to_cleanup:
            session_registry.remove_session(session_id)

        return jsonify({'models': trained_models})

    except Exception as e:
        logger.error(f"Failed to list trained models: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/models/<session_id>/info', methods=['GET'])
def get_model_info(session_id):
    """Get detailed information about a trained model."""
    try:
        session_registry = current_app.session_registry
        session_info = session_registry.get_session(session_id)

        if not session_info:
            return jsonify({'error': 'Model not found'}), 404

        return jsonify({
            'session_id': session_info.session_id,
            'model_name': session_info.model_name,
            'status': session_info.status,
            'created_at': session_info.created_at,
            'completed_at': session_info.completed_at,
            'epochs_trained': session_info.epochs_trained,
            'best_reward': session_info.best_reward,
            'checkpoint_path': session_registry.get_checkpoint_path(session_id)
        })

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return jsonify({'error': str(e)}), 500


# Trained models alias
@models_bp.route('/trained_models', methods=['GET'])
def get_trained_models_alias():
    """Alias for /api/models/trained."""
    return list_trained_models()


# Test history routes

@models_bp.route('/test_history', methods=['GET'])
def get_test_history():
    """Get test history."""
    try:
        # Get test history manager
        test_manager = get_test_history_manager()

        # Get limit and offset from query params
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)

        # Retrieve test history
        tests = test_manager.get_test_history(limit=limit, offset=offset)

        return jsonify({'tests': tests})
    except Exception as e:
        logger.error(f"Error getting test history: {e}")
        return jsonify({'tests': []})


@models_bp.route('/batch_tests/active', methods=['GET'])
def get_active_batch_tests():
    """Get active batch tests."""
    try:
        # Get batch test runner
        runner = get_batch_test_runner()

        # Get active test if any
        active_test = runner.get_active_test()

        return jsonify({'active_test': active_test})
    except Exception as e:
        logger.error(f"Error getting active batch tests: {e}")
        return jsonify({'active_test': None})


# Model testing routes

@models_bp.route('/test/models', methods=['GET'])
def get_testable_models():
    """Get list of models available for testing."""
    try:
        session_registry = current_app.session_registry

        # Get trained models
        trained_sessions = session_registry.list_completed_sessions()
        trained_models = []

        for session in trained_sessions:
            checkpoint_path = session_registry.get_checkpoint_path(session.session_id)
            if checkpoint_path and Path(checkpoint_path).exists():
                trained_models.append({
                    'id': session.session_id,
                    'name': f"{session.model_name} (Trained)",
                    'type': 'trained',
                    'base_model': session.model_name,
                    'checkpoint_path': checkpoint_path
                })

        # Get base models from constants (just a few popular ones for testing)
        base_models = []
        for category, models in MODEL_DEFINITIONS.items():
            for model in models[:2]:  # Take first 2 from each category
                base_models.append({
                    'id': model['id'],
                    'name': model['name'],
                    'type': 'base'
                })

        return jsonify({
            'trained': trained_models,
            'base': base_models
        })

    except Exception as e:
        logger.error(f"Failed to get testable models: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/test/load', methods=['POST'])
def load_test_model():
    """Load a model for testing."""
    try:
        data = request.json
        model_id = data.get('model_id')
        model_type = data.get('type', 'base')

        if not model_id:
            return jsonify({'error': 'Model ID required'}), 400

        if model_type == 'trained':
            session_registry = current_app.session_registry
            checkpoint_path = session_registry.get_checkpoint_path(model_id)

            if not checkpoint_path:
                return jsonify({'error': f'No checkpoint found for session {model_id}'}), 404

            success, error = model_tester.load_trained_model(checkpoint_path, model_id)
        else:
            success, error = model_tester.load_base_model(model_id)

        if success:
            return jsonify({
                'success': True,
                'message': f'Model {model_id} loaded successfully',
                'model_id': model_id
            })
        else:
            return jsonify({'error': error}), 500

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/test/generate', methods=['POST'])
def generate_test_response():
    """Generate a response from a loaded model."""
    try:
        data = request.json
        model_id = data.get('model_id')
        prompt = data.get('prompt')
        config = data.get('config', {})

        if not model_id or not prompt:
            return jsonify({'error': 'Model ID and prompt required'}), 400

        # Generate response
        response = model_tester.generate(
            model_id=model_id,
            prompt=prompt,
            max_length=config.get('max_length', 512),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.9)
        )

        if response.get('success'):
            return jsonify(response)
        else:
            return jsonify({'error': response.get('error', 'Generation failed')}), 500

    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/test/compare', methods=['POST'])
def compare_models():
    """Compare multiple models on the same prompt."""
    try:
        data = request.json
        model_ids = data.get('model_ids', [])
        prompt = data.get('prompt')
        config = data.get('config', {})

        if not model_ids or not prompt:
            return jsonify({'error': 'Model IDs and prompt required'}), 400

        results = []
        for model_id in model_ids:
            response = model_tester.generate(
                model_id=model_id,
                prompt=prompt,
                max_length=config.get('max_length', 512),
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 0.9)
            )

            results.append({
                'model_id': model_id,
                'response': response.get('response', ''),
                'success': response.get('success', False),
                'error': response.get('error')
            })

        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/test/clear-cache', methods=['POST'])
def clear_test_cache():
    """Clear loaded models from cache."""
    try:
        model_tester.clear_cache()
        return jsonify({
            'success': True,
            'message': 'Model cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({'error': str(e)}), 500


# Batch comparison routes

@models_bp.route('/test/batch-compare', methods=['POST'])
def batch_compare():
    """Start a batch comparison test."""
    try:
        data = request.json
        model_ids = data.get('model_ids', [])
        prompts = data.get('prompts', [])
        config = data.get('config', {})

        if not model_ids or not prompts:
            return jsonify({'error': 'Model IDs and prompts required'}), 400

        # Get batch test runner
        runner = get_batch_test_runner()

        # Start batch test
        batch_id = runner.start_batch_test(
            model_ids=model_ids,
            prompts=prompts,
            config=config
        )

        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'message': 'Batch comparison started'
        })

    except Exception as e:
        logger.error(f"Failed to start batch comparison: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/test/batch-status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id):
    """Get status of a batch comparison."""
    try:
        runner = get_batch_test_runner()
        status = runner.get_batch_status(batch_id)

        if status:
            return jsonify(status)
        else:
            return jsonify({'error': 'Batch test not found'}), 404

    except Exception as e:
        logger.error(f"Failed to get batch status: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/test/batch-results/<batch_id>', methods=['GET'])
def get_batch_results(batch_id):
    """Get results of a completed batch comparison."""
    try:
        runner = get_batch_test_runner()
        results = runner.get_batch_results(batch_id)

        if results:
            return jsonify(results)
        else:
            return jsonify({'error': 'Batch test not found or not completed'}), 404

    except Exception as e:
        logger.error(f"Failed to get batch results: {e}")
        return jsonify({'error': str(e)}), 500


@models_bp.route('/test/batch-list', methods=['GET'])
def list_batch_tests():
    """List all batch tests."""
    try:
        runner = get_batch_test_runner()
        tests = runner.list_batch_tests()
        return jsonify({'tests': tests})
    except Exception as e:
        logger.error(f"Failed to list batch tests: {e}")
        return jsonify({'tests': []})


@models_bp.route('/test/batch-cancel/<batch_id>', methods=['POST'])
def cancel_batch_test(batch_id):
    """Cancel a running batch test."""
    try:
        runner = get_batch_test_runner()
        success = runner.cancel_batch_test(batch_id)

        if success:
            return jsonify({
                'success': True,
                'message': f'Batch test {batch_id} cancelled'
            })
        else:
            return jsonify({'error': 'Batch test not found or already completed'}), 404

    except Exception as e:
        logger.error(f"Failed to cancel batch test: {e}")
        return jsonify({'error': str(e)}), 500


# Additional test routes

@models_bp.route('/test_model', methods=['POST'])
def test_model_simple():
    """Simple model testing endpoint (legacy)."""
    try:
        data = request.json
        session_id = data.get('session_id')
        prompt = data.get('prompt')

        if not session_id or not prompt:
            return jsonify({'error': 'Session ID and prompt required'}), 400

        session_registry = current_app.session_registry
        checkpoint_path = session_registry.get_checkpoint_path(session_id)

        if not checkpoint_path:
            return jsonify({'error': f'No checkpoint found for session {session_id}'}), 404

        # Load and test model
        success, error = model_tester.load_trained_model(checkpoint_path, session_id)
        if not success:
            return jsonify({'error': error}), 500

        response = model_tester.generate(
            model_id=session_id,
            prompt=prompt,
            max_length=512
        )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Failed to test model: {e}")
        return jsonify({'error': str(e)}), 500
