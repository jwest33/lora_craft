"""
Application factory for creating and configuring the Flask app.

This module provides the create_app() function that:
- Initializes Flask app with configuration
- Creates required directories
- Sets up CORS and SocketIO
- Initializes global state and services
- Registers all blueprints
- Registers WebSocket handlers
- Starts background tasks
"""

import os
import sys
import threading
from pathlib import Path

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core import SystemConfig, SessionRegistry
from utils.logging_config import setup_logging, get_logger

# Import route blueprints
from routes import (
    system_bp,
    configs_bp,
    rewards_bp,
    exports_bp,
    templates_bp,
    datasets_bp,
    training_bp,
    models_bp
)

# Import WebSocket handlers
from websockets import register_socketio_handlers
from websockets.utils import periodic_queue_processor


def create_app(config=None):
    """
    Create and configure the Flask application.

    Args:
        config: Optional configuration dictionary

    Returns:
        Tuple of (app, socketio)
    """
    # Setup logging
    setup_logging(log_level="INFO", log_dir="logs")
    logger = get_logger(__name__)

    # Initialize Flask app
    app = Flask(__name__)

    # Apply configuration
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB max file size for large datasets
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    app.config['ALLOWED_EXTENSIONS'] = {'json', 'jsonl', 'csv', 'parquet'}

    # Apply custom config if provided
    if config:
        app.config.update(config)

    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('exports', exist_ok=True)

    logger.info("Created required directories")

    # Enable CORS for API access
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize SocketIO for real-time updates
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    # Initialize global state
    training_sessions = {}
    training_threads = {}
    session_queues = {}

    # System configuration (shared)
    system_config = SystemConfig()

    # Session registry for fast model lookup
    session_registry = SessionRegistry()

    # Store in app context for routes
    app.training_sessions = training_sessions
    app.training_threads = training_threads
    app.session_queues = session_queues
    app.system_config = system_config
    app.session_registry = session_registry
    app.socketio = socketio

    logger.info("Initialized global state and services")

    # Register blueprints
    app.register_blueprint(system_bp)
    app.register_blueprint(configs_bp)
    app.register_blueprint(rewards_bp)
    app.register_blueprint(exports_bp)
    app.register_blueprint(templates_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(models_bp)

    logger.info("Registered all blueprints")

    # Register WebSocket handlers
    register_socketio_handlers(socketio, session_queues, training_sessions)

    logger.info("Registered WebSocket handlers")

    # Register legacy API routes for backward compatibility
    # These provide flat /api/<endpoint> routes that some frontend code still expects
    from services import DatasetService
    legacy_dataset_service = DatasetService()

    @app.route('/api/upload_dataset', methods=['POST'])
    def upload_dataset_legacy():
        """Legacy route for /api/upload_dataset (redirects to /api/datasets/upload)."""
        from flask import request as flask_request
        # Forward to the actual blueprint route
        with app.test_request_context(
            path='/api/datasets/upload',
            method='POST',
            data=flask_request.get_data(),
            headers=flask_request.headers
        ):
            from routes.datasets import datasets_bp
            return app.view_functions['datasets.upload_dataset']()

    @app.route('/api/list_datasets', methods=['GET'])
    def list_datasets_legacy():
        """Legacy route for /api/list_datasets."""
        from core import DatasetHandler
        from constants import POPULAR_DATASETS

        try:
            handler = DatasetHandler()
            datasets_with_status = []

            for dataset_path, info in POPULAR_DATASETS.items():
                # Extract dataset_config if it exists (for multi-config datasets like GSM8K)
                dataset_config = info.get('dataset_config')

                # Check cache status with config to differentiate between different configs
                is_cached = handler.is_cached(dataset_path, dataset_config)
                cache_info = handler.get_cache_info(dataset_path, dataset_config) if is_cached else None

                dataset_entry = {
                    'path': dataset_path,
                    'name': info['name'],
                    'size': info['size'],
                    'category': info['category'],
                    'is_cached': is_cached,
                    'cache_info': cache_info.to_dict() if cache_info else None
                }

                # Include dataset_config if present (for multi-config datasets like GSM8K)
                if dataset_config:
                    dataset_entry['dataset_config'] = dataset_config

                datasets_with_status.append(dataset_entry)

            from flask import jsonify
            return jsonify({'datasets': datasets_with_status})
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            from flask import jsonify
            return jsonify({'error': str(e)}), 500

    @app.route('/api/upload_dataset_info', methods=['POST'])
    def upload_dataset_info_legacy():
        """Legacy route for /api/upload_dataset_info."""
        try:
            import os
            from flask import request as flask_request, jsonify
            data = flask_request.json
            dataset_path = data.get('path')

            if not dataset_path:
                return jsonify({'error': 'Path required'}), 400

            # Convert relative path to absolute if needed
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path)

            if not os.path.exists(dataset_path):
                return jsonify({'error': 'File not found'}), 404

            # Get dataset info using service
            dataset_info = legacy_dataset_service.get_upload_info(dataset_path)

            # Rename 'name' to 'columns' if needed for compatibility
            if 'columns' not in dataset_info and dataset_info.get('columns'):
                dataset_info['columns'] = dataset_info.get('columns', [])

            # Add detected fields if available from suggested_mappings
            if 'suggested_mappings' in dataset_info:
                mappings = dataset_info['suggested_mappings']
                if 'instruction' in mappings:
                    dataset_info['detected_instruction_field'] = mappings['instruction']
                if 'response' in mappings:
                    dataset_info['detected_response_field'] = mappings['response']

            return jsonify(dataset_info)
        except FileNotFoundError as e:
            from flask import jsonify
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"Failed to get upload info: {e}")
            from flask import jsonify
            return jsonify({'error': str(e)}), 500

    # Training routes
    @app.route('/api/start_training', methods=['POST'])
    def start_training_legacy():
        """Legacy route for /api/start_training."""
        return app.view_functions['training.start_training']()

    @app.route('/api/stop_training', methods=['POST'])
    def stop_training_legacy():
        """Legacy route for /api/stop_training."""
        from flask import request as flask_request, jsonify
        try:
            data = flask_request.json or {}
            session_id = data.get('session_id')

            if not session_id:
                return jsonify({'error': 'Session ID required', 'success': False}), 400

            # Call the training blueprint's stop function
            return app.view_functions['training.stop_training'](session_id)

        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    # Model/Testing routes
    @app.route('/api/test_model', methods=['POST'])
    def test_model_legacy():
        """Legacy route for /api/test_model."""
        return app.view_functions['models.generate_test_response']()

    @app.route('/api/save_test', methods=['POST'])
    def save_test_legacy():
        """Legacy route for /api/save_test."""
        return app.view_functions['models.save_test_result']()

    @app.route('/api/start_batch_test', methods=['POST'])
    def start_batch_test_legacy():
        """Legacy route for /api/start_batch_test."""
        return app.view_functions['models.start_batch_test']()

    @app.route('/api/trained_models', methods=['GET'])
    def trained_models_legacy():
        """Legacy route for /api/trained_models."""
        return app.view_functions['models.list_trained_models']()

    @app.route('/api/evaluate', methods=['POST'])
    def evaluate_legacy():
        """Legacy route for /api/evaluate."""
        return app.view_functions['models.evaluate_model']()

    # Configuration routes
    @app.route('/api/save_configuration', methods=['POST'])
    def save_configuration_legacy():
        """Legacy route for /api/save_configuration."""
        return app.view_functions['configs.save_config']()

    # Dataset routes
    @app.route('/api/dataset_info', methods=['POST'])
    def dataset_info_legacy():
        """Legacy route for /api/dataset_info."""
        from flask import request as flask_request, jsonify
        try:
            data = flask_request.json
            dataset_name = data.get('path') or data.get('dataset_name')
            dataset_config = data.get('dataset_config')  # Extract dataset_config if provided

            if not dataset_name:
                return jsonify({'error': 'Dataset path required'}), 400

            # Call the datasets blueprint's detect_fields function
            # which provides similar functionality
            result = legacy_dataset_service.detect_fields(
                dataset_name=dataset_name,
                dataset_config=dataset_config,  # Pass dataset_config to service
                is_local='uploads' in dataset_name or dataset_name.endswith(('.json', '.jsonl', '.csv', '.parquet'))
            )

            return jsonify(result)
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/preview_dataset', methods=['POST'])
    def preview_dataset_legacy():
        """Legacy route for /api/preview_dataset."""
        from flask import request as flask_request, jsonify
        try:
            data = flask_request.json
            dataset_name = data.get('path') or data.get('dataset_name')
            dataset_config = data.get('dataset_config')  # Extract dataset_config if provided
            num_samples = data.get('samples', 5)

            if not dataset_name:
                return jsonify({'error': 'Dataset path required'}), 400

            # Call the datasets blueprint's sample function
            result = legacy_dataset_service.sample_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                sample_size=num_samples
            )

            return jsonify(result)
        except Exception as e:
            logger.error(f"Error previewing dataset: {e}")
            return jsonify({'error': str(e), 'samples': []}), 500

    # Export/Model info routes
    @app.route('/api/model_info', methods=['POST'])
    def model_info_legacy():
        """Legacy route for /api/model_info."""
        from flask import request as flask_request, jsonify
        from pathlib import Path
        try:
            data = flask_request.json
            model_path = data.get('path')

            if not model_path:
                return jsonify({'error': 'Model path required'}), 400

            # Get model info from path
            path = Path(model_path)

            info = {
                'name': path.name if path.exists() else model_path,
                'size': path.stat().st_size if path.exists() else 0,
                'timestamp': path.stat().st_mtime * 1000 if path.exists() else None,
                'metrics': {}
            }

            # Try to load metrics from session info if available
            session_registry = current_app.session_registry
            session_id = path.name if path.exists() else None
            if session_id:
                session = session_registry.get_session(session_id)
                if session:
                    info['metrics'] = {
                        'final_loss': session.best_reward,
                        'epochs': session.epochs_trained
                    }

            return jsonify(info)
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return jsonify({'error': str(e)}), 500

    logger.info("Registered legacy API routes for backward compatibility")

    # Start background tasks
    def start_queue_processor():
        """Start the periodic queue processor in a background thread."""
        processor_thread = threading.Thread(
            target=periodic_queue_processor,
            args=(session_queues, training_sessions, socketio),
            daemon=True
        )
        processor_thread.start()
        logger.info("Started periodic queue processor")

    # Start queue processor after app is ready
    start_queue_processor()

    # Register root route
    @app.route('/')
    def index():
        """Serve the main application page."""
        from flask import render_template
        return render_template('index.html')

    # Register manifest.json for PWA support
    @app.route('/manifest.json')
    def manifest():
        """Serve PWA manifest file."""
        from flask import jsonify
        return jsonify({
            'name': 'LoRA Craft',
            'short_name': 'LoRA Craft',
            'start_url': '/',
            'display': 'standalone',
            'background_color': '#ffffff',
            'theme_color': '#000000',
            'description': 'GRPO Fine-tuning Interface'
        })

    logger.info("Flask application created successfully")

    return app, socketio
