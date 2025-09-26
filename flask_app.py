#!/usr/bin/env python3
"""
LoRA Craft Flask Server
A web-based interface for GRPO (Group Relative Policy Optimization) fine-tuning
"""

import os
import sys
import json
import uuid
import threading
import queue
import logging
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import psutil

from flask import Flask, render_template, request, jsonify, session, Response, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import existing core modules
from core import (
    SystemConfig,
    GRPOModelTrainer,
    GRPOTrainingConfig,
    DatasetHandler,
    DatasetConfig,
    PromptTemplate,
    TemplateConfig,
    CustomRewardBuilder,
    ModelExporter,
    SessionRegistry,
    SessionInfo,
    ModelTester,
    TestConfig
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

# Session registry for fast model lookup
session_registry = SessionRegistry()


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

        # Store key info for display
        self.model_name = config.get('model_name', 'model').split('/')[-1]
        self.dataset_path = config.get('dataset_path', 'dataset').split('/')[-1]
        self.num_epochs = config.get('num_epochs', 3)

        # Generate display name
        self.display_name = self._generate_display_name(config)

        self.metrics = {
            'current_epoch': 0,
            'total_epochs': self.num_epochs,
            'current_step': 0,
            'total_steps': 0,
            'loss': None,
            'reward': None,
            'learning_rate': config.get('learning_rate', 2e-4),
            'samples_processed': 0
        }
        self.logs = []

    def _generate_display_name(self, config):
        """Generate a display name for the model."""
        # Check if user provided a custom name
        if config.get('display_name'):
            return config.get('display_name')

        # Generate automatic name: model_dataset_MMDD_HHMM
        model_short = config.get('model_name', 'model').split('/')[-1].replace('-', '')
        dataset_short = config.get('dataset_name', 'dataset').split('/')[-1].replace('-', '_')[:20]
        timestamp = self.created_at.strftime('%m%d_%H%M')

        return f"{model_short}_{dataset_short}_{timestamp}"

    def to_dict(self):
        """Convert session to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'config': self.config,
            'model': self.config.get('model_name', 'Unknown'),
            'model_name': self.model_name,
            'display_name': self.display_name,
            'dataset': self.config.get('dataset_path', 'Unknown'),
            'dataset_path': self.dataset_path,
            'epochs': self.num_epochs,
            'num_epochs': self.num_epochs,
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

    # Create a custom output capture for console logs
    class OutputCapture:
        def __init__(self, queue, original_stream):
            self.queue = queue
            self.original = original_stream
            self.buffer = io.StringIO()

        def write(self, text):
            # Write to original stream
            self.original.write(text)
            self.original.flush()

            # Also capture for processing
            self.buffer.write(text)

            # Check if we have a complete line
            if '\n' in text:
                content = self.buffer.getvalue()
                lines = content.split('\n')

                # Process complete lines
                for line in lines[:-1]:
                    if line.strip():
                        # Try to parse as a metrics dictionary
                        if line.startswith('{') and ('loss' in line or 'reward' in line):
                            try:
                                # Use ast.literal_eval for safer parsing
                                import ast
                                metrics_dict = ast.literal_eval(line)

                                # Extract key metrics
                                processed_metrics = {
                                    'step': metrics_dict.get('step', 0),
                                    'epoch': metrics_dict.get('epoch', 0),
                                    'loss': metrics_dict.get('loss', 0),
                                    'mean_reward': metrics_dict.get('reward', metrics_dict.get('rewards/reward_wrapper/mean', 0)),
                                    'learning_rate': metrics_dict.get('learning_rate', config.get('learning_rate', 2e-4)),
                                    'grad_norm': metrics_dict.get('grad_norm', 0),
                                    'reward_std': metrics_dict.get('reward_std', metrics_dict.get('rewards/reward_wrapper/std', 0))
                                }

                                # Send metrics update
                                self.queue.put(('metrics', processed_metrics))
                            except Exception:
                                # Not a valid metrics dict, ignore
                                pass

                # Keep the incomplete line in buffer
                self.buffer = io.StringIO()
                if lines[-1]:
                    self.buffer.write(lines[-1])

        def flush(self):
            self.original.flush()

        def fileno(self):
            return self.original.fileno()

    # Capture stdout
    original_stdout = sys.stdout

    try:
        session_obj = training_sessions[session_id]
        session_obj.status = 'running'
        session_obj.started_at = datetime.now()

        q = session_queues[session_id]

        # Set up output capture
        sys.stdout = OutputCapture(q, original_stdout)

        # Send initial status
        q.put(('log', f"Initializing training for session {session_id}"))

        # Create GRPO configuration with algorithm support
        grpo_config = GRPOTrainingConfig(
            # Model configuration
            model_name=config.get('model_name', 'unsloth/Qwen3-0.6B'),

            # LoRA configuration
            lora_r=config.get('lora_rank', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.0),
            lora_target_modules=config.get('lora_target_modules', None),
            lora_bias=config.get('lora_bias', 'none'),

            # Training configuration
            num_train_epochs=config.get('num_epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 4),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            learning_rate=config.get('learning_rate', 2e-4),
            warmup_steps=config.get('warmup_steps', 10),
            weight_decay=config.get('weight_decay', 0.001),
            max_grad_norm=config.get('max_grad_norm', 0.3),
            lr_scheduler_type=config.get('lr_scheduler_type', 'constant'),
            optim=config.get('optim', 'paged_adamw_32bit'),
            logging_steps=config.get('logging_steps', 10),
            save_steps=config.get('save_steps', 100),
            eval_steps=config.get('eval_steps', 100),
            seed=config.get('seed', 42),

            # Generation configuration
            max_sequence_length=config.get('max_sequence_length', 2048),
            max_new_tokens=config.get('max_new_tokens', 256),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.95),
            top_k=config.get('top_k', 50),
            repetition_penalty=config.get('repetition_penalty', 1.0),
            num_generations_per_prompt=config.get('num_generations', 4),
            num_generations=config.get('num_generations', 4),  # For TRL compatibility

            # GRPO/Algorithm specific
            loss_type=config.get('loss_type', 'grpo'),
            importance_sampling_level=config.get('importance_sampling_level', 'token'),
            kl_penalty=config.get('kl_penalty', 0.01),
            clip_range=config.get('clip_range', 0.2),
            value_coefficient=config.get('value_coefficient', 1.0),
            epsilon=config.get('epsilon', 3e-4),
            epsilon_high=config.get('epsilon_high', 4e-4),

            # Quantization configuration
            use_4bit=config.get('use_4bit', False),
            use_8bit=config.get('use_8bit', False),
            load_in_4bit=config.get('use_4bit', False),  # Same as use_4bit
            bnb_4bit_compute_dtype=config.get('bnb_4bit_compute_dtype', 'float16'),
            bnb_4bit_quant_type=config.get('bnb_4bit_quant_type', 'nf4'),
            use_nested_quant=config.get('use_nested_quant', False),

            # Optimization flags
            use_flash_attention=config.get('use_flash_attention', False),
            gradient_checkpointing=config.get('gradient_checkpointing', False),
            fp16=config.get('fp16', False),
            bf16=config.get('bf16', False),

            # Paths
            output_dir=f"./outputs/{session_id}",
            checkpoint_dir=f"./outputs/{session_id}/checkpoints"
        )

        # Create trainer
        trainer = GRPOModelTrainer(grpo_config, system_config)
        session_obj.trainer = trainer

        # Setup callbacks
        def progress_callback(progress):
            q.put(('progress', progress))

        def metrics_callback(metrics):
            q.put(('metrics', metrics))

        def log_callback(log_entry):
            # Send log to frontend via queue
            q.put(('log', f"[{log_entry['level']}] {log_entry['message']}"))
            # Also emit via socketio if session is known
            socketio.emit('training_log', {
                'session_id': session_id,
                'message': f"[{log_entry['level']}] {log_entry['message']}"
            }, room=session_id)

        trainer.progress_callback = progress_callback
        trainer.metrics_callback = metrics_callback
        trainer.set_log_callback(log_callback)

        # Load model
        q.put(('log', "Loading model..."))
        trainer.setup_model()
        q.put(('log', "Model loaded successfully"))

        # Load dataset
        q.put(('log', "Loading dataset..."))
        dataset_config = DatasetConfig(
            source_type=config.get('dataset_source', 'huggingface'),
            source_path=config.get('dataset_path', 'tatsu-lab/alpaca'),
            subset=config.get('dataset_config', None),  # Config for multi-config datasets
            split=config.get('dataset_split', 'train[:100]'),  # Limit for demo
            instruction_field=config.get('instruction_field', 'instruction'),
            response_field=config.get('response_field', 'output'),
            max_samples=config.get('max_samples', 100)
        )

        dataset_handler = DatasetHandler(dataset_config)
        dataset = dataset_handler.load()
        q.put(('log', f"Loaded {len(dataset)} samples"))

        # Setup prompt template with chat template
        template_config = TemplateConfig(
            name="training",
            description="Training template",
            instruction_template=config.get('instruction_template', "{instruction}"),
            response_template=config.get('response_template', "{response}"),
            reasoning_start_marker=config.get('reasoning_start', '<start_working_out>'),
            reasoning_end_marker=config.get('reasoning_end', '<end_working_out>'),
            solution_start_marker=config.get('solution_start', '<SOLUTION>'),
            solution_end_marker=config.get('solution_end', '</SOLUTION>'),
            system_prompt=config.get('system_prompt', 'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>'),
            chat_template=config.get('chat_template'),  # Add chat template from config
            prepend_reasoning_start=True,  # For GRPO templates
            model_type=config.get('chat_template_type', 'grpo')
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

        # Log algorithm selection based on importance_sampling_level
        importance_level = config.get('importance_sampling_level', 'token')
        algorithm_name = 'GSPO' if importance_level == 'sequence' else 'GRPO'
        q.put(('log', f"Using {algorithm_name} algorithm for training (importance_sampling_level: {importance_level})"))

        # Pre-training phase for format learning (if enabled)
        if config.get('enable_pre_training', True):
            q.put(('log', "Starting pre-training phase for format learning..."))

            # Get pre-training configuration
            pre_training_epochs = config.get('pre_training_epochs', 1)
            pre_training_samples = config.get('pre_training_samples', 100)
            validate_format = config.get('validate_format', True)

            # Subset dataset for pre-training
            pre_train_dataset = dataset.select(range(min(pre_training_samples, len(dataset))))

            # Configure tokenizer with chat template
            if hasattr(trainer, 'tokenizer') and trainer.tokenizer:
                template.setup_for_unsloth(trainer.tokenizer)
                q.put(('log', f"Applied {config.get('chat_template_type', 'grpo')} chat template to tokenizer"))

            # Run pre-fine-tuning
            pre_metrics = trainer.pre_fine_tune(pre_train_dataset, template, epochs=pre_training_epochs)
            q.put(('log', f"Pre-training completed: {pre_training_epochs} epochs on {len(pre_train_dataset)} samples"))

            # Validate format compliance if requested
            if validate_format:
                q.put(('log', "Testing format compliance..."))
                test_prompt = "What is 2 + 2?"
                test_messages = [
                    {'role': 'user', 'content': test_prompt}
                ]

                # Generate a sample response to check format
                try:
                    formatted_prompt = template.apply_chat_template(
                        test_messages,
                        add_generation_prompt=True
                    )

                    # Log the formatted prompt for debugging
                    q.put(('log', f"Test prompt formatted: {formatted_prompt[:200]}..."))

                    # Here you could add actual generation and validation
                    # For now, just log that validation was attempted
                    q.put(('log', "Format validation check completed"))
                except Exception as e:
                    q.put(('log', f"Format validation skipped: {str(e)}"))

        # GRPO/GSPO training
        q.put(('log', f"Starting {algorithm_name} training..."))
        metrics = trainer.grpo_train(dataset, template, reward_builder)

        # Update final metrics
        session_obj.metrics.update(metrics)

        # Confirm checkpoint was saved (already done in grpo_train)
        q.put(('log', "Final checkpoint saved successfully"))

        # Add to session registry
        # Handle infinity values in best_reward
        final_reward = metrics.get('final_reward')
        if final_reward is not None and (final_reward == float('-inf') or final_reward == float('inf')):
            final_reward = None

        session_info = SessionInfo(
            session_id=session_id,
            model_name=config.get('model_name', 'Unknown'),
            status='completed',
            checkpoint_path=f"outputs/{session_id}/checkpoints/final",
            created_at=session_obj.created_at.isoformat() if session_obj.created_at else None,
            completed_at=datetime.now().isoformat(),
            best_reward=final_reward,
            epochs_trained=config.get('num_epochs', 0),
            training_config=config,
            display_name=session_obj.display_name
        )
        session_registry.add_session(session_info)
        q.put(('log', "Session added to registry"))

        q.put(('log', "Training completed successfully"))
        q.put(('complete', "Training finished!"))

    except Exception as e:
        logger.error(f"Training error for session {session_id}: {e}")
        q.put(('error', str(e)))
        if session_id in training_sessions:
            training_sessions[session_id].status = 'error'
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

        # Mark session as completed
        if session_id in training_sessions:
            session_obj = training_sessions[session_id]
            # Note: We don't automatically cleanup the model anymore to allow exports
            # Users can still manually trigger cleanup if needed to free memory
            # Uncomment the following line to enable auto-cleanup:
            # if session_obj.trainer:
            #     session_obj.trainer.cleanup()
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


@app.route('/manifest.json')
def manifest():
    """Serve the PWA manifest."""
    return send_file('static/manifest.json', mimetype='application/manifest+json')

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


@app.route('/api/configs/list', methods=['GET'])
def list_configs():
    """List all saved configurations."""
    try:
        configs_dir = Path('configs')
        configs = []

        if configs_dir.exists():
            for config_file in configs_dir.glob('*.json'):
                # Get file info
                stat = config_file.stat()
                configs.append({
                    'name': config_file.stem,
                    'filename': config_file.name,
                    'modified': stat.st_mtime,
                    'size': stat.st_size
                })

        # Sort by modified time (newest first)
        configs.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify(configs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/configs/delete/<filename>', methods=['DELETE'])
def delete_config(filename):
    """Delete a configuration file."""
    try:
        filepath = Path('configs') / filename

        if not filepath.exists():
            return jsonify({'error': 'Configuration file not found'}), 404

        filepath.unlink()

        return jsonify({'message': f'Configuration {filename} deleted successfully'})
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
            'model': session_obj.config.get('model_name', 'Unknown'),
            'display_name': session_obj.display_name
        })
    return jsonify(sessions)


@app.route('/api/export/<session_id>', methods=['POST'])
def export_model(session_id):
    """Export trained model."""
    # First check if session exists in memory
    if session_id in training_sessions:
        session_obj = training_sessions[session_id]
        if session_obj.status != 'completed':
            return jsonify({'error': 'Training not completed'}), 400
    else:
        session_obj = None

    try:
        export_config = request.json or {}
        export_format = export_config.get('format', 'huggingface')
        export_name = export_config.get('name', None)
        quantization = export_config.get('quantization', 'q4_k_m')
        merge_lora = export_config.get('merge_lora', False)

        # Progress callback for WebSocket updates
        def progress_callback(message, progress):
            emit_to_session(session_id, 'export_progress', {
                'message': message,
                'progress': progress
            })

        # Try to use trainer's export method if available and model is loaded
        if (session_obj and
            hasattr(session_obj, 'trainer') and
            session_obj.trainer is not None and
            hasattr(session_obj.trainer, 'model') and
            session_obj.trainer.model is not None):

            logger.info(f"Exporting using in-memory model for session {session_id}")
            success, export_path, metadata = session_obj.trainer.export_model(
                export_format=export_format,
                export_name=export_name,
                quantization=quantization if export_format == 'gguf' else None,
                merge_lora=merge_lora,
                progress_callback=progress_callback
            )
        else:
            # Export directly from checkpoint files
            logger.info(f"Exporting from checkpoint files for session {session_id}")

            # Check if checkpoint exists
            checkpoint_path = Path("./outputs") / session_id / "checkpoints" / "final"
            if not checkpoint_path.exists():
                # Try to find any checkpoint
                checkpoints_dir = Path("./outputs") / session_id / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
                    if checkpoints:
                        # Use the most recent checkpoint
                        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Using checkpoint: {checkpoint_path}")
                    else:
                        return jsonify({'error': 'No checkpoint found for this session'}), 404
                else:
                    return jsonify({'error': 'No checkpoints directory found for this session'}), 404

            # Use ModelExporter directly
            exporter = ModelExporter()
            success, export_path, metadata = exporter.export_model(
                model_path=str(checkpoint_path),
                session_id=session_id,
                export_format=export_format,
                export_name=export_name,
                quantization=quantization if export_format == 'gguf' else None,
                merge_lora=merge_lora,
                progress_callback=progress_callback
            )

        if success:
            return jsonify({
                'success': True,
                'path': export_path,
                'format': export_format,
                'metadata': metadata
            })
        else:
            return jsonify({
                'success': False,
                'error': metadata.get('error', 'Export failed')
            }), 500

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/formats', methods=['GET'])
def get_export_formats():
    """Get available export formats."""
    return jsonify({
        'formats': ModelExporter.SUPPORTED_FORMATS,
        'gguf_quantizations': ModelExporter.GGUF_QUANTIZATIONS
    })


@app.route('/api/export/list/<session_id>', methods=['GET'])
def list_exports(session_id):
    """List all exports for a session."""
    try:
        exporter = ModelExporter()
        exports = exporter.list_exports(session_id)
        return jsonify(exports)
    except Exception as e:
        logger.error(f"Error listing exports: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/download/<session_id>/<path:export_path>', methods=['GET'])
def download_export(session_id, export_path):
    """Download an exported model."""
    try:
        # Construct the full path
        full_path = Path("./exports") / session_id / export_path

        if not full_path.exists():
            return jsonify({'error': 'Export not found'}), 404

        # If it's a directory, create a zip
        if full_path.is_dir():
            exporter = ModelExporter()
            success, archive_path = exporter.create_archive(str(full_path))
            if success:
                return send_file(
                    archive_path,
                    as_attachment=True,
                    download_name=f"{full_path.name}.zip"
                )
            else:
                return jsonify({'error': 'Failed to create archive'}), 500
        else:
            # Send single file
            return send_file(
                str(full_path),
                as_attachment=True,
                download_name=full_path.name
            )

    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/checkpoints/<session_id>', methods=['GET'])
def list_checkpoints(session_id):
    """List available checkpoints for a session."""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    try:
        session_obj = training_sessions[session_id]
        if not hasattr(session_obj, 'trainer') or session_obj.trainer is None:
            return jsonify({'error': 'Trainer not available'}), 400

        checkpoints = session_obj.trainer.list_checkpoints()
        return jsonify({'checkpoints': checkpoints})
    except Exception as e:
        logger.error(f"Error listing checkpoints: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Manually cleanup model resources for a session to free memory."""
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
        logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/trained', methods=['GET'])
def list_trained_models():
    """List all trained models from the session registry."""
    try:
        # Get completed sessions from registry (fast O(1) lookup)
        sessions = session_registry.list_completed_sessions()

        trained_models = []
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
            if session.checkpoint_path:
                checkpoint_path = Path(session.checkpoint_path)
                if checkpoint_path.exists():
                    # List available checkpoints
                    checkpoint_dir = checkpoint_path.parent
                    for checkpoint in checkpoint_dir.iterdir():
                        if checkpoint.is_dir() and (checkpoint / "training_state.json").exists():
                            model_info['checkpoints'].append({
                                'name': checkpoint.name,
                                'path': str(checkpoint)
                            })

                    trained_models.append(model_info)
                else:
                    # Checkpoint deleted, remove from registry
                    logger.warning(f"Checkpoint missing for session {session.session_id}, cleaning up")
                    # Note: cleanup could be done in a background task

        return jsonify({
            'models': trained_models,
            'total': len(trained_models)
        })

    except Exception as e:
        logger.error(f"Error listing trained models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/registry/rebuild', methods=['POST'])
def rebuild_registry():
    """Rebuild the session registry by scanning output directories."""
    try:
        sessions_found = session_registry.rebuild_from_directories()
        return jsonify({
            'success': True,
            'message': f'Registry rebuilt with {sessions_found} sessions',
            'sessions_found': sessions_found
        })
    except Exception as e:
        logger.error(f"Error rebuilding registry: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/registry/cleanup', methods=['POST'])
def cleanup_registry():
    """Remove invalid sessions from the registry."""
    try:
        removed = session_registry.cleanup_invalid_sessions()
        return jsonify({
            'success': True,
            'message': f'Removed {removed} invalid sessions',
            'sessions_removed': removed
        })
    except Exception as e:
        logger.error(f"Error cleaning up registry: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<session_id>/info', methods=['GET'])
def get_model_info(session_id):
    """Get detailed information about a specific model."""
    try:
        model_path = Path("./outputs") / session_id
        if not model_path.exists():
            return jsonify({'error': 'Model not found'}), 404

        info = {
            'session_id': session_id,
            'path': str(model_path),
            'checkpoints': [],
            'exports': []
        }

        # Get checkpoint information
        checkpoint_dir = model_path / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint in checkpoint_dir.iterdir():
                if checkpoint.is_dir():
                    checkpoint_info = {
                        'name': checkpoint.name,
                        'path': str(checkpoint)
                    }

                    # Get training state if available
                    state_file = checkpoint / "training_state.json"
                    if state_file.exists():
                        with open(state_file, 'r') as f:
                            checkpoint_info['training_state'] = json.load(f)

                    # Check for model files
                    checkpoint_info['has_model'] = (checkpoint / "adapter_model.safetensors").exists() or \
                                                   (checkpoint / "pytorch_model.bin").exists()

                    info['checkpoints'].append(checkpoint_info)

        # Get export history
        exporter = ModelExporter()
        export_data = exporter.list_exports(session_id)
        info['exports'] = export_data['exports']

        # Get configuration if available
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                info['config'] = json.load(f)

        return jsonify(info)

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exports/batch', methods=['POST'])
def batch_export():
    """Export multiple models in batch."""
    try:
        data = request.json
        session_ids = data.get('session_ids', [])
        export_format = data.get('format', 'huggingface')
        quantization = data.get('quantization', 'q4_k_m')

        if not session_ids:
            return jsonify({'error': 'No session IDs provided'}), 400

        results = []
        exporter = ModelExporter()

        for session_id in session_ids:
            # Check if model exists
            model_path = Path("./outputs") / session_id / "checkpoints" / "final"
            if not model_path.exists():
                results.append({
                    'session_id': session_id,
                    'success': False,
                    'error': 'Model checkpoint not found'
                })
                continue

            # Export the model
            success, export_path, metadata = exporter.export_model(
                model_path=str(model_path),
                session_id=session_id,
                export_format=export_format,
                quantization=quantization if export_format == 'gguf' else None
            )

            results.append({
                'session_id': session_id,
                'success': success,
                'export_path': export_path if success else None,
                'metadata': metadata if success else None,
                'error': metadata.get('error') if not success else None
            })

        return jsonify({
            'results': results,
            'total': len(results),
            'successful': sum(1 for r in results if r['success'])
        })

    except Exception as e:
        logger.error(f"Batch export error: {str(e)}")
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
        dataset_config = data.get('config', None)  # Config for multi-config datasets
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
            subset=dataset_config,  # This maps to 'name' parameter in HF load_dataset
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
        dataset_config = data.get('config', None)  # Config for multi-config datasets
        sample_size = data.get('sample_size', 5)

        if not dataset_name:
            return jsonify({'error': 'Dataset name required'}), 400

        # Create config for sampling
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            subset=dataset_config,  # This maps to 'name' parameter in HF load_dataset
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
    """Get list of Public Datasets with their status."""
    try:
        # Define Public Datasets (matching the frontend catalog)
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

@app.route('/api/templates/chat-templates', methods=['GET'])
def get_chat_templates():
    """Get available chat templates."""
    try:
        # Built-in chat templates
        builtin_templates = {
            'grpo': {
                'name': 'GRPO Default',
                'template': "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ system_prompt + eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ reasoning_start }}{% endif %}",
                'description': 'GRPO template optimized for reasoning tasks'
            }
        }

        # Load custom templates from file storage
        custom_templates = {}
        chat_templates_dir = Path('./chat_templates')
        if chat_templates_dir.exists():
            for template_file in chat_templates_dir.glob('*.json'):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        custom_templates[template_file.stem] = template_data
                except:
                    pass

        return jsonify({
            'builtin': builtin_templates,
            'custom': custom_templates
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/chat-template/save', methods=['POST'])
def save_chat_template():
    """Save a custom chat template."""
    try:
        data = request.json
        name = data.get('name')
        template = data.get('template')
        description = data.get('description', '')

        if not name or not template:
            return jsonify({'error': 'Name and template required'}), 400

        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()

        # Save template to file
        chat_templates_dir = Path('./chat_templates')
        chat_templates_dir.mkdir(exist_ok=True)

        template_path = chat_templates_dir / f"{safe_name}.json"
        with open(template_path, 'w') as f:
            json.dump({
                'name': name,
                'template': template,
                'description': description
            }, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Chat template "{name}" saved successfully',
            'template_id': safe_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/chat-template/validate', methods=['POST'])
def validate_chat_template():
    """Validate a Jinja2 chat template."""
    try:
        data = request.json
        template = data.get('template')

        if not template:
            return jsonify({'valid': False, 'error': 'No template provided'}), 400

        # Try to validate with Jinja2
        from jinja2 import Environment, TemplateSyntaxError
        env = Environment()

        try:
            env.from_string(template)
            return jsonify({'valid': True, 'message': 'Template is valid'})
        except TemplateSyntaxError as e:
            return jsonify({'valid': False, 'error': str(e)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/chat-template/preview', methods=['POST'])
def preview_chat_template():
    """Preview a chat template with sample data."""
    try:
        data = request.json
        template = data.get('template')
        messages = data.get('messages', [])
        system_prompt = data.get('system_prompt', '')
        reasoning_start = data.get('reasoning_start', '<start_working_out>')
        eos_token = data.get('eos_token', '</s>')
        add_generation_prompt = data.get('add_generation_prompt', False)

        if not template:
            return jsonify({'error': 'No template provided'}), 400

        # Render template
        from jinja2 import Environment
        env = Environment()

        try:
            tmpl = env.from_string(template)
            preview = tmpl.render(
                messages=messages,
                system_prompt=system_prompt,
                reasoning_start=reasoning_start,
                eos_token=eos_token,
                add_generation_prompt=add_generation_prompt
            )
            return jsonify({'preview': preview})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
# Model Testing Routes
# ============================================================================

# Initialize model tester
model_tester = ModelTester()


@app.route('/api/test/models', methods=['GET'])
def get_testable_models():
    """Get list of trained models available for testing."""
    try:
        # Get completed sessions from registry
        sessions = session_registry.list_completed_sessions()

        testable_models = []
        for session in sessions:
            # Get base model name from training config
            base_model = session.training_config.get('model_name', 'Unknown')

            # Extract dataset name from config
            dataset_path = session.training_config.get('dataset_path', 'unknown')

            model_info = {
                'session_id': session.session_id,
                'model_name': session.model_name,
                'display_name': session.display_name or session.model_name,
                'base_model': base_model,
                'checkpoint_path': session.checkpoint_path,
                'created_at': session.created_at,
                'epochs': session.epochs_trained or 0,
                'num_epochs': session.epochs_trained or 0,
                'dataset_path': dataset_path,
                'best_reward': session.best_reward if session.best_reward != float('-inf') else None
            }

            # Check if checkpoint exists
            if session.checkpoint_path:
                checkpoint_path = Path(session.checkpoint_path)
                if checkpoint_path.exists():
                    testable_models.append(model_info)

        return jsonify({
            'models': testable_models,
            'loaded': model_tester.get_loaded_models()
        })

    except Exception as e:
        logger.error(f"Failed to get testable models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/load', methods=['POST'])
def load_test_models():
    """Load models for testing."""
    try:
        data = request.json
        session_id = data.get('session_id')
        base_model = data.get('base_model')

        if not session_id or not base_model:
            return jsonify({'error': 'session_id and base_model required'}), 400

        # Get checkpoint path from registry
        session_info = session_registry.get_session(session_id)
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404

        checkpoint_path = session_info.checkpoint_path
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return jsonify({'error': 'Checkpoint not found'}), 404

        results = {}

        # Load trained model
        success, error = model_tester.load_trained_model(checkpoint_path, session_id)
        results['trained'] = {'success': success, 'error': error}

        # Load base model
        success, error = model_tester.load_base_model(base_model)
        results['base'] = {'success': success, 'error': error}

        return jsonify({
            'results': results,
            'loaded_models': model_tester.get_loaded_models()
        })

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/generate', methods=['POST'])
def generate_test_response():
    """Generate response from a model."""
    try:
        data = request.json
        prompt = data.get('prompt')
        model_type = data.get('model_type', 'trained')  # 'trained' or 'base'
        model_key = data.get('model_key')  # session_id or base_model_name

        # Generation config
        config_data = data.get('config', {})
        config = TestConfig(
            temperature=config_data.get('temperature', 0.7),
            max_new_tokens=config_data.get('max_new_tokens', 512),
            top_p=config_data.get('top_p', 0.95),
            top_k=config_data.get('top_k', 50),
            repetition_penalty=config_data.get('repetition_penalty', 1.0),
            do_sample=config_data.get('do_sample', True)
        )

        use_chat_template = data.get('use_chat_template', True)

        if not prompt or not model_key:
            return jsonify({'error': 'prompt and model_key required'}), 400

        # Generate response
        result = model_tester.generate_response(
            prompt=prompt,
            model_type=model_type,
            model_key=model_key,
            config=config,
            use_chat_template=use_chat_template
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/compare', methods=['POST'])
def compare_models():
    """Compare responses from trained and base models."""
    try:
        data = request.json
        prompt = data.get('prompt')
        session_id = data.get('session_id')
        base_model = data.get('base_model')

        # Generation config
        config_data = data.get('config', {})
        config = TestConfig(
            temperature=config_data.get('temperature', 0.7),
            max_new_tokens=config_data.get('max_new_tokens', 512),
            top_p=config_data.get('top_p', 0.95),
            top_k=config_data.get('top_k', 50),
            repetition_penalty=config_data.get('repetition_penalty', 1.0),
            do_sample=config_data.get('do_sample', True)
        )

        use_chat_template = data.get('use_chat_template', True)

        if not prompt or not session_id or not base_model:
            return jsonify({'error': 'prompt, session_id, and base_model required'}), 400

        # First ensure models are loaded
        session_info = session_registry.get_session(session_id)
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404

        checkpoint_path = session_info.checkpoint_path
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return jsonify({'error': 'Checkpoint not found'}), 404

        # Load models if not already loaded
        trained_cache_key = f"trained_{session_id}"
        base_cache_key = f"base_{base_model.replace('/', '_')}"

        if trained_cache_key not in model_tester.loaded_models:
            success, error = model_tester.load_trained_model(checkpoint_path, session_id)
            if not success:
                return jsonify({'error': f'Failed to load trained model: {error}'}), 500

        if base_cache_key not in model_tester.loaded_models:
            success, error = model_tester.load_base_model(base_model)
            if not success:
                return jsonify({'error': f'Failed to load base model: {error}'}), 500

        # Compare models
        results = model_tester.compare_models(
            prompt=prompt,
            trained_session_id=session_id,
            base_model_name=base_model,
            config=config,
            use_chat_template=use_chat_template
        )

        return jsonify(results)

    except Exception as e:
        logger.error(f"Failed to compare models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/chat-template', methods=['GET', 'POST'])
def manage_chat_template():
    """Get or set the chat template for base model."""
    try:
        if request.method == 'GET':
            return jsonify({
                'template': model_tester.base_chat_template
            })
        else:
            data = request.json
            template = data.get('template')

            if not template or '{prompt}' not in template:
                return jsonify({'error': 'Invalid template, must contain {prompt} placeholder'}), 400

            model_tester.set_chat_template(template)

            return jsonify({
                'success': True,
                'message': 'Chat template updated'
            })

    except Exception as e:
        logger.error(f"Failed to manage chat template: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/clear-cache', methods=['POST'])
def clear_model_cache():
    """Clear loaded models from memory."""
    try:
        data = request.json or {}
        model_key = data.get('model_key')

        model_tester.clear_model_cache(model_key)

        return jsonify({
            'success': True,
            'message': f'Cleared cache for {model_key}' if model_key else 'Cleared all model cache',
            'loaded_models': model_tester.get_loaded_models()
        })

    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
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
    for dir_name in ['configs', 'logs', 'outputs', 'exports', 'cache', 'templates', 'static']:
        Path(dir_name).mkdir(exist_ok=True)

    # Initialize or rebuild registry if needed
    if not Path("./outputs/sessions.json").exists():
        print("Session registry not found, rebuilding from directories...")
        sessions_found = session_registry.rebuild_from_directories()
        print(f"Found {sessions_found} existing training sessions")
    else:
        print(f"Loaded {len(session_registry.sessions)} sessions from registry")

    # Run the Flask app with SocketIO
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print(f"Starting GRPO Flask server on port {port}")
    print(f"Access the web interface at http://localhost:{port}")

    socketio.run(app, host='0.0.0.0', port=port, debug=debug)
