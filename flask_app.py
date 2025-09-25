#!/usr/bin/env python3
"""
GRPO Fine-Tuner Flask Server
A web-based interface for GRPO (Group Relative Policy Optimization) fine-tuning
"""

import os
import sys
import json
import uuid
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from flask import Flask, render_template, request, jsonify, session, Response, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import existing core modules
from core import (
    SystemConfig,
    GRPOTrainer,
    GRPOConfig,
    DatasetHandler,
    DatasetConfig,
    PromptTemplate,
    TemplateConfig,
    CustomRewardBuilder
)
from utils.logging_config import setup_logging, get_logger
from utils.validators import validate_training_config

# Setup logging
setup_logging(log_level="INFO", log_dir="logs")
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SESSION_TYPE'] = 'filesystem'

# Enable CORS for API access
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global storage for training sessions
training_sessions = {}
training_threads = {}
session_queues = {}

# System configuration (shared)
system_config = SystemConfig()


class TrainingSession:
    """Represents a training session for a user."""

    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.status = "initialized"
        self.trainer = None
        self.thread = None
        self.queue = queue.Queue()
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.metrics = {
            'current_epoch': 0,
            'total_epochs': config.get('num_epochs', 3),
            'current_step': 0,
            'total_steps': 0,
            'loss': None,
            'reward': None,
            'learning_rate': config.get('learning_rate', 2e-4),
            'samples_processed': 0
        }
        self.logs = []

    def to_dict(self):
        """Convert session to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'config': self.config,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'logs': self.logs[-50:]  # Last 50 log entries
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def emit_to_session(session_id: str, event: str, data: Any):
    """Emit event to specific session via SocketIO."""
    socketio.emit(event, data, room=session_id)


def process_training_queue(session_id: str):
    """Process messages from training queue and emit to client."""
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
                    emit_to_session(session_id, 'training_progress', {
                        'progress': msg_data,
                        'session_id': session_id
                    })

                elif msg_type == 'metrics':
                    if session_obj:
                        session_obj.metrics.update(msg_data)
                    emit_to_session(session_id, 'training_metrics', msg_data)

                elif msg_type == 'log':
                    if session_obj:
                        session_obj.logs.append(msg_data)
                    emit_to_session(session_id, 'training_log', {'message': msg_data})

                elif msg_type == 'complete':
                    if session_obj:
                        session_obj.status = 'completed'
                        session_obj.completed_at = datetime.now()
                    emit_to_session(session_id, 'training_complete', {
                        'message': msg_data,
                        'session_id': session_id
                    })

                elif msg_type == 'error':
                    if session_obj:
                        session_obj.status = 'error'
                        session_obj.logs.append(f"ERROR: {msg_data}")
                    emit_to_session(session_id, 'training_error', {
                        'error': msg_data,
                        'session_id': session_id
                    })

            except queue.Empty:
                break
    except Exception as e:
        logger.error(f"Error processing queue for session {session_id}: {e}")


def run_training(session_id: str, config: Dict[str, Any]):
    """Run training in background thread."""
    try:
        session_obj = training_sessions[session_id]
        session_obj.status = 'running'
        session_obj.started_at = datetime.now()

        q = session_queues[session_id]

        # Send initial status
        q.put(('log', f"Initializing training for session {session_id}"))

        # Create GRPO configuration with algorithm support
        grpo_config = GRPOConfig(
            model_name=config.get('model_name', 'unsloth/Qwen3-0.6B'),
            num_train_epochs=config.get('num_epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 4),
            learning_rate=config.get('learning_rate', 2e-4),
            lora_r=config.get('lora_rank', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.0),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.95),
            num_generations_per_prompt=config.get('num_generations', 4),
            kl_penalty=config.get('kl_penalty', 0.01),
            clip_range=config.get('clip_range', 0.2),
            # Algorithm selection
            loss_type=config.get('loss_type', 'grpo'),
            importance_sampling_level='sequence' if config.get('loss_type') == 'gspo' else 'token',
            epsilon=config.get('epsilon', 3e-4),
            epsilon_high=config.get('epsilon_high', 4e-4),
            # Hardware optimizations
            use_flash_attention=config.get('use_flash_attention', False),
            gradient_checkpointing=config.get('gradient_checkpointing', False),
            fp16=config.get('mixed_precision', True) and not config.get('bf16', False),
            bf16=config.get('bf16', False),
            output_dir=f"./outputs/{session_id}",
            checkpoint_dir=f"./checkpoints/{session_id}"
        )

        # Create trainer
        trainer = GRPOTrainer(grpo_config, system_config)
        session_obj.trainer = trainer

        # Setup callbacks
        def progress_callback(progress):
            q.put(('progress', progress))

        def metrics_callback(metrics):
            q.put(('metrics', metrics))

        trainer.progress_callback = progress_callback
        trainer.metrics_callback = metrics_callback

        # Load model
        q.put(('log', "Loading model..."))
        trainer.setup_model()
        q.put(('log', "Model loaded successfully"))

        # Load dataset
        q.put(('log', "Loading dataset..."))
        dataset_config = DatasetConfig(
            source_type=config.get('dataset_source', 'huggingface'),
            source_path=config.get('dataset_path', 'tatsu-lab/alpaca'),
            split=config.get('dataset_split', 'train[:100]'),  # Limit for demo
            instruction_field=config.get('instruction_field', 'instruction'),
            response_field=config.get('response_field', 'output'),
            max_samples=config.get('max_samples', 100)
        )

        dataset_handler = DatasetHandler(dataset_config)
        dataset = dataset_handler.load()
        q.put(('log', f"Loaded {len(dataset)} samples"))

        # Setup prompt template
        template_config = TemplateConfig(
            name="training",
            description="Training template",
            instruction_template=config.get('instruction_template', "{instruction}"),
            response_template=config.get('response_template', "{response}"),
            reasoning_start_marker=config.get('reasoning_start', '<start_working_out>'),
            reasoning_end_marker=config.get('reasoning_end', '<end_working_out>'),
            solution_start_marker=config.get('solution_start', '<SOLUTION>'),
            solution_end_marker=config.get('solution_end', '</SOLUTION>'),
            system_prompt=config.get('system_prompt', 'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>')
        )
        template = PromptTemplate(template_config)

        # Setup reward function based on configuration
        reward_builder = CustomRewardBuilder()
        reward_config = config.get('reward_config', {'type': 'preset', 'preset': 'math'})

        if reward_config.get('type') == 'preset':
            preset = reward_config.get('preset', 'math')
            if preset == 'math':
                # Mathematical problems reward
                reward_builder.add_numerical_reward("numerical_accuracy", tolerance=1e-6, weight=0.7)
                reward_builder.add_format_reward("answer_format",
                    pattern=r"\\boxed\{[^}]+\}|Final Answer:.*|Answer:.*", weight=0.2)
                reward_builder.add_length_reward("length", min_length=10, max_length=500,
                    optimal_length=100, weight=0.1)
            elif preset == 'code':
                # Code generation reward
                reward_builder.add_format_reward("code_block",
                    pattern=r"```[\w]*\n.*?\n```", weight=0.3)
                reward_builder.add_format_reward("function_def",
                    pattern=r"def\s+\w+\s*\(|function\s+\w+\s*\(|const\s+\w+\s*=", weight=0.3)
                reward_builder.add_format_reward("comments",
                    pattern=r"#.*|//.*|/\*.*?\*/", weight=0.2)
                reward_builder.add_length_reward("length", min_length=20, max_length=1000,
                    optimal_length=200, weight=0.2)
            elif preset == 'binary':
                reward_builder.add_binary_reward("exact_match", weight=1.0)
            elif preset == 'numerical':
                reward_builder.add_numerical_reward("numerical", tolerance=1e-6, weight=1.0)
            elif preset == 'length':
                reward_builder.add_length_reward("length", min_length=50, max_length=500, weight=1.0)
            elif preset == 'format':
                reward_builder.add_format_reward("format", pattern=r".*", weight=1.0)
        else:
            # Custom reward configuration
            components = reward_config.get('components', [])
            for comp in components:
                comp_type = comp.get('type')
                weight = comp.get('weight', 1.0)

                if comp_type == 'binary':
                    pattern = comp.get('pattern')
                    reward_builder.add_binary_reward(f"binary_{len(components)}",
                        regex_pattern=pattern, weight=weight)
                elif comp_type == 'numerical':
                    tolerance = comp.get('tolerance', 1e-6)
                    reward_builder.add_numerical_reward(f"numerical_{len(components)}",
                        tolerance=tolerance, weight=weight)
                elif comp_type == 'length':
                    min_len = comp.get('min_length')
                    max_len = comp.get('max_length')
                    reward_builder.add_length_reward(f"length_{len(components)}",
                        min_length=min_len, max_length=max_len, weight=weight)
                elif comp_type == 'format':
                    pattern = comp.get('pattern', r".*")
                    reward_builder.add_format_reward(f"format_{len(components)}",
                        pattern=pattern, weight=weight)

        # Log algorithm selection
        loss_type = config.get('loss_type', 'grpo').upper()
        q.put(('log', f"Using {loss_type} algorithm for training"))

        # Training phases
        if config.get('pre_finetune', False):
            q.put(('log', "Starting pre-fine-tuning phase..."))
            trainer.pre_fine_tune(dataset, template, epochs=1)
            q.put(('log', "Pre-fine-tuning completed"))

        # GRPO/GSPO training
        q.put(('log', f"Starting {loss_type} training..."))
        metrics = trainer.grpo_train(dataset, template, reward_builder)

        # Update final metrics
        session_obj.metrics.update(metrics)

        q.put(('log', "Training completed successfully"))
        q.put(('complete', "Training finished!"))

    except Exception as e:
        logger.error(f"Training error for session {session_id}: {e}")
        q.put(('error', str(e)))
        if session_id in training_sessions:
            training_sessions[session_id].status = 'error'
    finally:
        # Cleanup
        if session_id in training_sessions:
            session_obj = training_sessions[session_id]
            if session_obj.trainer:
                session_obj.trainer.cleanup()
            session_obj.completed_at = datetime.now()
            if session_obj.status == 'running':
                session_obj.status = 'completed'


# ============================================================================
# API Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')

@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    """Get system information."""
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
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB

        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models."""
    models = {
        'qwen': [
            {'id': 'unsloth/Qwen3-0.6B', 'name': 'Qwen3 0.6B', 'size': '600M', 'vram': '~1.2GB'},
            {'id': 'unsloth/Qwen3-1.7B', 'name': 'Qwen3 1.7B', 'size': '1.7B', 'vram': '~3.4GB'},
            {'id': 'unsloth/Qwen3-4B', 'name': 'Qwen3 4B', 'size': '4B', 'vram': '~8GB'},
            {'id': 'unsloth/Qwen3-8B', 'name': 'Qwen3 8B', 'size': '8B', 'vram': '~16GB'}
        ],
        'llama': [
            {'id': 'unsloth/Llama-3.2-1B-Instruct', 'name': 'LLaMA 3.2 1B', 'size': '1B', 'vram': '~2GB'},
            {'id': 'unsloth/Llama-3.2-3B-Instruct', 'name': 'LLaMA 3.2 3B', 'size': '3B', 'vram': '~6GB'}
        ],
        'phi': [
            {'id': 'unsloth/phi-4-reasoning', 'name': 'Phi-4 Reasoning', 'size': '15B', 'vram': '~30GB'}
        ]
    }
    return jsonify(models)


@app.route('/api/config/validate', methods=['POST'])
def validate_config():
    """Validate training configuration."""
    try:
        config = request.json
        valid, errors = validate_training_config(config)

        if valid:
            return jsonify({'valid': True, 'message': 'Configuration is valid'})
        else:
            return jsonify({'valid': False, 'errors': errors}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/save', methods=['POST'])
def save_config():
    """Save configuration to file."""
    try:
        config = request.json
        filename = config.get('filename', f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        filepath = Path('configs') / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        return jsonify({'success': True, 'filename': filename, 'path': str(filepath)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/load/<filename>', methods=['GET'])
def load_config(filename):
    """Load configuration from file."""
    try:
        filepath = Path('configs') / filename

        if not filepath.exists():
            return jsonify({'error': 'Configuration file not found'}), 404

        with open(filepath, 'r') as f:
            config = json.load(f)

        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start a new training session."""
    try:
        config = request.json

        # Validate configuration
        valid, errors = validate_training_config(config)
        if not valid:
            return jsonify({'error': 'Invalid configuration', 'errors': errors}), 400

        # Create new session
        session_id = create_session_id()
        session_obj = TrainingSession(session_id, config)

        # Store session
        training_sessions[session_id] = session_obj
        session_queues[session_id] = session_obj.queue

        # Start training in background thread
        thread = threading.Thread(
            target=run_training,
            args=(session_id, config),
            daemon=True
        )
        thread.start()
        training_threads[session_id] = thread

        return jsonify({
            'session_id': session_id,
            'status': 'started',
            'message': 'Training started successfully'
        })

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/<session_id>/status', methods=['GET'])
def get_training_status(session_id):
    """Get status of a training session."""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_obj = training_sessions[session_id]
    return jsonify(session_obj.to_dict())


@app.route('/api/training/<session_id>/stop', methods=['POST'])
def stop_training(session_id):
    """Stop a training session."""
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
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/<session_id>/metrics', methods=['GET'])
def get_training_metrics(session_id):
    """Get current metrics for a training session."""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_obj = training_sessions[session_id]
    return jsonify(session_obj.metrics)


@app.route('/api/training/<session_id>/logs', methods=['GET'])
def get_training_logs(session_id):
    """Get logs for a training session."""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_obj = training_sessions[session_id]
    limit = request.args.get('limit', 100, type=int)
    return jsonify({'logs': session_obj.logs[-limit:]})


@app.route('/api/training/sessions', methods=['GET'])
def list_training_sessions():
    """List all training sessions."""
    sessions = []
    for session_id, session_obj in training_sessions.items():
        sessions.append({
            'session_id': session_id,
            'status': session_obj.status,
            'created_at': session_obj.created_at.isoformat(),
            'model': session_obj.config.get('model_name', 'Unknown')
        })
    return jsonify(sessions)


@app.route('/api/export/<session_id>', methods=['POST'])
def export_model(session_id):
    """Export trained model."""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    try:
        session_obj = training_sessions[session_id]
        if session_obj.status != 'completed':
            return jsonify({'error': 'Training not completed'}), 400

        export_config = request.json
        export_format = export_config.get('format', 'safetensors')
        export_path = f"./exports/{session_id}/model"

        # Would call trainer export method here
        # session_obj.trainer.export_model(export_path, format=export_format)

        return jsonify({
            'success': True,
            'path': export_path,
            'format': export_format
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Dataset Management Routes
# ============================================================================

@app.route('/api/datasets/status/<dataset_name>', methods=['GET'])
def get_dataset_status(dataset_name):
    """Check if a dataset is cached and get its info."""
    try:
        # Replace forward slash with safe separator
        safe_name = dataset_name.replace('/', '__')

        # Create temporary handler to check cache
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name.replace('__', '/'),
            use_cache=True
        )
        handler = DatasetHandler(config)

        is_cached = handler.is_cached(dataset_name.replace('__', '/'))
        cache_info = handler.get_cache_info(dataset_name.replace('__', '/'))

        response = {
            'dataset_name': dataset_name.replace('__', '/'),
            'is_cached': is_cached,
            'cache_info': cache_info.to_dict() if cache_info else None
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Failed to get dataset status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/download', methods=['POST'])
def download_dataset():
    """Download a dataset with progress tracking."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        force_download = data.get('force_download', False)

        if not dataset_name:
            return jsonify({'error': 'Dataset name required'}), 400

        # Create session for progress tracking
        session_id = create_session_id()

        def progress_callback(info):
            # Emit progress via SocketIO
            emit_to_session(session_id, 'dataset_progress', info)

        # Create dataset config
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            use_cache=True,
            force_download=force_download
        )

        # Start download in background
        def download_task():
            try:
                handler = DatasetHandler(config, progress_callback=progress_callback)
                dataset = handler.load()

                # Get cache info
                cache_info = handler.get_cache_info(dataset_name)

                emit_to_session(session_id, 'dataset_complete', {
                    'dataset_name': dataset_name,
                    'samples': len(dataset) if dataset else 0,
                    'cache_info': cache_info.to_dict() if cache_info else None
                })
            except Exception as e:
                logger.error(f"Dataset download failed: {e}")
                emit_to_session(session_id, 'dataset_error', {
                    'error': str(e)
                })

        # Start download thread
        thread = threading.Thread(target=download_task, daemon=True)
        thread.start()

        return jsonify({
            'session_id': session_id,
            'message': 'Download started'
        })

    except Exception as e:
        logger.error(f"Failed to start dataset download: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/sample', methods=['POST'])
def sample_dataset():
    """Get a sample of a dataset for preview."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        sample_size = data.get('sample_size', 5)

        if not dataset_name:
            return jsonify({'error': 'Dataset name required'}), 400

        # Create config for sampling
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            sample_size=sample_size,
            use_cache=True  # Use cache if available
        )

        handler = DatasetHandler(config)
        dataset = handler.load()

        # Get sample data
        samples = handler.get_preview(sample_size)

        # Get statistics if available
        stats = None
        if handler.statistics:
            stats = {
                'total_samples': handler.statistics.total_samples,
                'avg_instruction_length': handler.statistics.avg_instruction_length,
                'avg_response_length': handler.statistics.avg_response_length
            }

        return jsonify({
            'dataset_name': dataset_name,
            'samples': samples,
            'statistics': stats
        })

    except Exception as e:
        logger.error(f"Failed to sample dataset: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/cache/info', methods=['GET'])
def get_cache_info():
    """Get information about all cached datasets."""
    try:
        # Create handler to access cache
        handler = DatasetHandler()

        # Get all cache info
        cache_items = []
        total_size = 0

        for cache_key, info in handler.cache_info.items():
            cache_items.append(info.to_dict())
            total_size += info.size_bytes

        return jsonify({
            'cache_items': cache_items,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(handler._cache_dir)
        })

    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/cache/clear', methods=['POST'])
def clear_dataset_cache():
    """Clear cache for specific dataset or all datasets."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')  # Optional, if not provided clears all

        handler = DatasetHandler()
        handler.clear_cache(dataset_name)

        message = f"Cleared cache for {dataset_name}" if dataset_name else "Cleared all dataset cache"

        return jsonify({
            'success': True,
            'message': message
        })

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/list', methods=['GET'])
def list_popular_datasets():
    """Get list of popular datasets with their status."""
    try:
        # Define popular datasets (matching the frontend catalog)
        popular_datasets = {
            'tatsu-lab/alpaca': {
                'name': 'Alpaca',
                'size': '52K samples',
                'category': 'general'
            },
            'openai/gsm8k': {
                'name': 'GSM8K',
                'size': '8.5K problems',
                'category': 'math'
            },
            'open-r1/DAPO-Math-17k-Processed': {
                'name': 'DAPO Math 17k',
                'size': '17K problems',
                'category': 'math'
            },
            'nvidia/OpenMathReasoning': {
                'name': 'OpenMath Reasoning',
                'size': '100K+ problems',
                'category': 'math'
            },
            'sahil2801/CodeAlpaca-20k': {
                'name': 'Code Alpaca',
                'size': '20K examples',
                'category': 'coding'
            },
            'databricks/databricks-dolly-15k': {
                'name': 'Dolly 15k',
                'size': '15K samples',
                'category': 'general'
            },
            'microsoft/orca-math-word-problems-200k': {
                'name': 'Orca Math',
                'size': '200K problems',
                'category': 'math'
            },
            'squad_v2': {
                'name': 'SQuAD v2',
                'size': '150K questions',
                'category': 'qa'
            }
        }

        # Check cache status for each dataset
        handler = DatasetHandler()

        datasets_with_status = []
        for dataset_path, info in popular_datasets.items():
            is_cached = handler.is_cached(dataset_path)
            cache_info = handler.get_cache_info(dataset_path) if is_cached else None

            datasets_with_status.append({
                'path': dataset_path,
                'name': info['name'],
                'size': info['size'],
                'category': info['category'],
                'is_cached': is_cached,
                'cache_info': cache_info.to_dict() if cache_info else None
            })

        return jsonify({
            'datasets': datasets_with_status
        })

    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Template Management Routes
# ============================================================================

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get available prompt templates."""
    try:
        # Default GRPO templates
        default_templates = {
            'grpo-default': {
                'name': 'GRPO Default',
                'description': 'Default GRPO template for reasoning tasks',
                'reasoning_start': '<start_working_out>',
                'reasoning_end': '<end_working_out>',
                'solution_start': '<SOLUTION>',
                'solution_end': '</SOLUTION>',
                'system_prompt': 'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>'
            },
            'qwen': {
                'name': 'Qwen GRPO',
                'description': 'GRPO template optimized for Qwen models',
                'reasoning_start': '<start_working_out>',
                'reasoning_end': '<end_working_out>',
                'solution_start': '<SOLUTION>',
                'solution_end': '</SOLUTION>',
                'system_prompt': 'You are given a problem to solve.\nProvide your reasoning within <start_working_out> and <end_working_out> tags.\nThen provide the final answer within <SOLUTION></SOLUTION> tags.'
            },
            'llama': {
                'name': 'LLaMA GRPO',
                'description': 'GRPO template optimized for LLaMA models',
                'reasoning_start': '[THINKING]',
                'reasoning_end': '[/THINKING]',
                'solution_start': '[ANSWER]',
                'solution_end': '[/ANSWER]',
                'system_prompt': 'You are a helpful assistant.\nShow your work within [THINKING] and [/THINKING].\nProvide the final answer within [ANSWER] and [/ANSWER].'
            }
        }

        # Load custom templates from file storage
        custom_templates = {}
        templates_dir = Path('./templates')
        if templates_dir.exists():
            for template_file in templates_dir.glob('*.json'):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        custom_templates[template_file.stem] = template_data
                except:
                    pass

        return jsonify({
            'default': default_templates,
            'custom': custom_templates
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/save', methods=['POST'])
def save_template():
    """Save a custom prompt template."""
    try:
        template_data = request.json
        template_name = template_data.get('name')

        if not template_name:
            return jsonify({'error': 'Template name required'}), 400

        # Sanitize template name for filesystem
        safe_name = "".join(c for c in template_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

        # Save template to file
        templates_dir = Path('./templates')
        templates_dir.mkdir(exist_ok=True)

        template_path = templates_dir / f"{safe_name}.json"
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Template "{template_name}" saved successfully',
            'template_id': safe_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/<template_id>', methods=['DELETE'])
def delete_template(template_id):
    """Delete a custom template."""
    try:
        template_path = Path(f'./templates/{template_id}.json')
        if template_path.exists():
            template_path.unlink()
            return jsonify({'success': True, 'message': 'Template deleted'})
        else:
            return jsonify({'error': 'Template not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/test', methods=['POST'])
def test_template():
    """Test a template with sample data."""
    try:
        config = request.json

        # Create template config
        template_config = TemplateConfig(
            name="test",
            description="Test template",
            reasoning_start_marker=config.get('reasoning_start', '<start_working_out>'),
            reasoning_end_marker=config.get('reasoning_end', '<end_working_out>'),
            solution_start_marker=config.get('solution_start', '<SOLUTION>'),
            solution_end_marker=config.get('solution_end', '</SOLUTION>'),
            system_prompt=config.get('system_prompt')
        )

        template = PromptTemplate(template_config)

        # Test with sample data
        sample_instruction = config.get('sample_instruction', 'What is 2 + 2?')
        sample_response = config.get('sample_response', '2 + 2 = 4')

        # Format the sample
        messages = [
            {'role': 'system', 'content': template_config.system_prompt} if template_config.system_prompt else None,
            {'role': 'user', 'content': sample_instruction},
            {'role': 'assistant', 'content': sample_response}
        ]
        messages = [m for m in messages if m]  # Remove None entries

        # Generate preview
        preview = f"System: {template_config.system_prompt}\n\n" if template_config.system_prompt else ""
        preview += f"User: {sample_instruction}\n\n"
        preview += f"Assistant: {template_config.reasoning_start_marker}\n"
        preview += f"[Reasoning would go here]\n"
        preview += f"{template_config.reasoning_end_marker}\n"
        preview += f"{template_config.solution_start_marker}\n"
        preview += f"{sample_response}\n"
        preview += f"{template_config.solution_end_marker}"

        return jsonify({
            'success': True,
            'preview': preview,
            'formatted': preview
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WebSocket Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to GRPO training server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


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
        process_training_queue(session_id)


@socketio.on('join_dataset_session')
def handle_join_dataset_session(data):
    """Join a dataset download session for updates."""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('joined_dataset_session', {'session_id': session_id})
        logger.info(f"Client {request.sid} joined dataset session {session_id}")


# ============================================================================
# Background Tasks
# ============================================================================

def periodic_queue_processor():
    """Process all training queues periodically."""
    while True:
        try:
            for session_id in list(session_queues.keys()):
                process_training_queue(session_id)
            socketio.sleep(0.5)  # Process every 500ms
        except Exception as e:
            logger.error(f"Error in queue processor: {e}")
            socketio.sleep(1)


# Start background task for processing queues
socketio.start_background_task(periodic_queue_processor)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Create necessary directories
    for dir_name in ['configs', 'logs', 'outputs', 'checkpoints', 'exports', 'cache', 'templates', 'static']:
        Path(dir_name).mkdir(exist_ok=True)

    # Run the Flask app with SocketIO
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print(f"Starting GRPO Flask server on port {port}")
    print(f"Access the web interface at http://localhost:{port}")

    socketio.run(app, host='0.0.0.0', port=port, debug=debug)