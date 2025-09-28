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
from werkzeug.utils import secure_filename
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
from core.custom_rewards import (
    RewardPresetLibrary,
    RewardTester,
    RewardTemplateLibrary
)
from utils.logging_config import setup_logging, get_logger
from utils.validators import validate_training_config

# Setup logging
setup_logging(log_level="INFO", log_dir="logs")
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB max file size for large datasets
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'json', 'jsonl', 'csv', 'parquet'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Define Popular Datasets catalog (matching the frontend catalog)
POPULAR_DATASETS = {
    'tatsu-lab/alpaca': {
        'name': 'Alpaca',
        'size': '52K samples',
        'estimated_mb': 45,
        'sample_count': 52000,
        'category': 'general'
    },
    'openai/gsm8k': {
        'name': 'GSM8K',
        'size': '8.5K problems',
        'estimated_mb': 12,
        'sample_count': 8500,
        'category': 'math',
        'field_mapping': {
            'instruction': 'question',
            'response': 'answer'
        }
    },
    'nvidia/OpenMathReasoning': {
        'name': 'OpenMath Reasoning',
        'size': '100K problems',
        'estimated_mb': 200,
        'sample_count': 100000,
        'category': 'math',
        'default_split': 'cot',
        'field_mapping': {
            'instruction': 'problem',
            'response': 'generated_solution'
        }
    },
    'sahil2801/CodeAlpaca-20k': {
        'name': 'Code Alpaca',
        'size': '20K examples',
        'estimated_mb': 35,
        'sample_count': 20000,
        'category': 'coding',
        'field_mapping': {
            'instruction': 'prompt',
            'response': 'completion'
        }
    },
    'databricks/databricks-dolly-15k': {
        'name': 'Dolly 15k',
        'size': '15K samples',
        'estimated_mb': 30,
        'sample_count': 15000,
        'category': 'general',
        'default_split': 'train',
        'field_mapping': {
            'instruction': 'instruction',
            'response': 'response'
        }
    },
    'microsoft/orca-math-word-problems-200k': {
        'name': 'Orca Math',
        'size': '200K problems',
        'estimated_mb': 350,
        'sample_count': 200000,
        'category': 'math',
        'field_mapping': {
            'instruction': 'question',
            'response': 'answer'
        }
    },
    'squad': {
        'name': 'SQuAD v2',
        'size': '130K questions',
        'estimated_mb': 120,
        'sample_count': 130000,
        'category': 'qa',
        'field_mapping': {
            'instruction': 'question',
            'response': 'answers'
        }
    }
}


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
        # Extract dataset name properly from path
        dataset_path = config.get('dataset_path', 'dataset')
        if '\\uploads\\' in dataset_path or '/uploads/' in dataset_path:
            # Local file - get filename without extension
            self.dataset_path = os.path.basename(dataset_path).replace('.csv', '').replace('.json', '')
        else:
            # HuggingFace dataset - get dataset name
            self.dataset_path = dataset_path.split('/')[-1] if '/' in dataset_path else dataset_path
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
        self.metrics_history = []  # Store historical metrics for reconnection
        self.progress = 0  # Training progress percentage

    def _generate_display_name(self, config):
        """Generate a display name for the model."""
        # Check if user provided a custom name
        if config.get('display_name'):
            return config.get('display_name')

        # Generate automatic name: model_dataset_MMDD_HHMM
        model_short = config.get('model_name', 'model').split('/')[-1].replace('-', '')
        # Use dataset_path instead of dataset_name as that's what's sent from frontend
        dataset_path = config.get('dataset_path', 'dataset')
        # Extract dataset name from path (could be local file or HF dataset)
        if '/' in dataset_path:
            dataset_short = dataset_path.split('/')[-1].replace('-', '_')[:20]
        else:
            dataset_short = os.path.basename(dataset_path).replace('.csv', '').replace('.json', '').replace('-', '_')[:20]
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
    # Add session_id to data for client-side filtering
    data_with_id = data.copy() if isinstance(data, dict) else {'data': data}
    data_with_id['session_id'] = session_id

    # Broadcast to all clients (they'll filter by session_id)
    # This ensures updates work even if room joining hasn't completed
    socketio.emit(event, data_with_id)


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
                        session_obj.progress = msg_data  # Store progress for reconnection
                    emit_to_session(session_id, 'training_progress', {
                        'progress': msg_data,
                        'session_id': session_id
                    })

                elif msg_type == 'metrics':
                    if session_obj:
                        session_obj.metrics.update(msg_data)
                        # Update current_step if present
                        if 'step' in msg_data and msg_data['step'] > 0:
                            session_obj.metrics['current_step'] = msg_data['step']
                        # Store metrics history for reconnection
                        session_obj.metrics_history.append(msg_data.copy())
                    emit_to_session(session_id, 'training_metrics', msg_data)

                elif msg_type == 'log':
                    if session_obj:
                        session_obj.logs.append(msg_data)
                    emit_to_session(session_id, 'training_log', {'message': msg_data})

                elif msg_type == 'complete':
                    if session_obj:
                        session_obj.status = 'completed'
                        session_obj.completed_at = datetime.now()

                    # Clean up training models from memory
                    try:
                        # Clear any Unsloth models that might be in GPU memory
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"Cleared GPU memory after training completion for session {session_id}")
                    except Exception as e:
                        logger.warning(f"Error clearing GPU memory: {e}")

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
                        # Send the line as a log message first
                        self.queue.put(('log', line))

                        # Try to parse as a metrics dictionary
                        if '{' in line and ('loss' in line or 'reward' in line or 'step' in line):
                            try:
                                # Extract JSON/dict from the line (it might have other text)
                                import re
                                import ast

                                # Try to find a dictionary pattern in the line
                                dict_match = re.search(r'\{[^}]+\}', line)
                                if dict_match:
                                    dict_str = dict_match.group(0)
                                    metrics_dict = ast.literal_eval(dict_str)
                                else:
                                    metrics_dict = ast.literal_eval(line)

                                # Debug: Log all keys in metrics dict to find step key
                                if 'loss' in metrics_dict or 'reward' in metrics_dict:
                                    logger.info(f"Found metrics: {list(metrics_dict.keys())}")

                                # Extract key metrics - check multiple possible key names
                                step_value = (metrics_dict.get('step') or
                                            metrics_dict.get('global_step') or
                                            metrics_dict.get('iteration') or
                                            metrics_dict.get('steps') or 0)

                                processed_metrics = {
                                    'step': step_value,
                                    'epoch': metrics_dict.get('epoch', 0),
                                    'loss': metrics_dict.get('loss', 0),
                                    'mean_reward': metrics_dict.get('reward', metrics_dict.get('rewards/reward_wrapper/mean', 0)),
                                    'learning_rate': metrics_dict.get('learning_rate', config.get('learning_rate', 2e-4)),
                                    'grad_norm': metrics_dict.get('grad_norm', 0),
                                    'reward_std': metrics_dict.get('reward_std', metrics_dict.get('rewards/reward_wrapper/std', 0))
                                }

                                # Log the extracted step for debugging
                                if step_value > 0:
                                    logger.info(f"Training step {step_value} - loss: {processed_metrics['loss']:.4f}")

                                # Send metrics update
                                self.queue.put(('metrics', processed_metrics))
                            except Exception as e:
                                # Not a valid metrics dict, but log for debugging
                                logger.debug(f"Could not parse metrics from line: {line[:100]}... Error: {e}")

                        # Also check for step information in regular log output
                        elif 'Step' in line or 'step' in line or 'Epoch' in line:
                            import re

                            # Try to extract step number from patterns like "Step 10/100" or "step: 10"
                            step_patterns = [
                                r'[Ss]tep[:\s]+(\d+)',  # Step: 10 or Step 10
                                r'[Ss]tep[:\s]+\[?(\d+)/\d+\]?',  # Step [10/100]
                                r'\[(\d+)/\d+\]',  # [10/100] format
                                r'global_step[:\s]+(\d+)',  # global_step: 10
                            ]

                            for pattern in step_patterns:
                                match = re.search(pattern, line)
                                if match:
                                    step_num = int(match.group(1))
                                    logger.info(f"Extracted step {step_num} from log line")
                                    # Send a minimal metrics update with just the step
                                    self.queue.put(('metrics', {'step': step_num}))
                                    break

                            # Also check for percentage progress
                            progress_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                            if progress_match:
                                progress = float(progress_match.group(1))
                                self.queue.put(('progress', progress))

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
    original_stderr = sys.stderr

    try:
        session_obj = training_sessions[session_id]
        session_obj.status = 'running'
        session_obj.started_at = datetime.now()

        q = session_queues[session_id]

        # Set up output capture for both stdout and stderr
        sys.stdout = OutputCapture(q, original_stdout)
        sys.stderr = OutputCapture(q, original_stderr)

        # Configure transformers logging to be more verbose
        import transformers
        transformers.logging.set_verbosity_info()

        # Enable progress bars in transformers
        import os
        os.environ['TRANSFORMERS_NO_PROGRESS'] = '0'

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
            per_device_train_batch_size=config.get('batch_size', 1),
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
            # Use batch_size as default for num_generations if not provided or None
            num_generations_per_prompt=config.get('num_generations') or config.get('batch_size', 1),
            num_generations=config.get('num_generations') or config.get('batch_size', 1),  # For TRL compatibility

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

        # Determine source type based on dataset_path
        dataset_path = config.get('dataset_path', 'tatsu-lab/alpaca')
        source_type = config.get('dataset_source', 'huggingface')

        # Check if this is an uploaded file
        if dataset_path and (dataset_path.startswith(app.config['UPLOAD_FOLDER']) or
                            '\\uploads\\' in dataset_path or '/uploads/' in dataset_path):
            source_type = 'local'
            q.put(('log', f"Using uploaded dataset file: {os.path.basename(dataset_path)}"))

        dataset_config = DatasetConfig(
            source_type=source_type,
            source_path=dataset_path,
            subset=config.get('dataset_config', None),  # Config for multi-config datasets
            split=config.get('dataset_split', 'train[:100]') if source_type == 'huggingface' else 'train',  # No split for local files
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

        # Force flush to ensure output is captured
        sys.stdout.flush()

        # Add progress callback to trainer if possible
        def training_callback(step, total_steps, logs):
            """Callback to report training progress."""
            if logs:
                # Send metrics update
                metrics_update = {
                    'step': step,
                    'total_steps': total_steps
                }
                # Add any available metrics from logs
                if isinstance(logs, dict):
                    metrics_update.update(logs)
                q.put(('metrics', metrics_update))

            # Calculate and send progress
            if total_steps > 0:
                progress = (step / total_steps) * 100
                q.put(('progress', progress))

        # Check if we can add a callback
        if hasattr(trainer, 'set_callback'):
            trainer.set_callback(training_callback)

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
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Mark session as completed and clean up resources
        if session_id in training_sessions:
            session_obj = training_sessions[session_id]

            # Clean up the trainer model from memory after training
            # The trained model is saved to disk and can be loaded separately for testing
            if hasattr(session_obj, 'trainer') and session_obj.trainer:
                try:
                    # Call cleanup method if available
                    if hasattr(session_obj.trainer, 'cleanup'):
                        session_obj.trainer.cleanup()
                        logger.info(f"Cleaned up trainer resources for session {session_id}")

                    # Delete the trainer object reference
                    del session_obj.trainer
                    session_obj.trainer = None
                    logger.info(f"Deleted trainer reference for session {session_id}")

                except Exception as e:
                    logger.warning(f"Error cleaning up trainer: {e}")
                finally:
                    # Ensure trainer is set to None even if cleanup fails
                    session_obj.trainer = None

            # Force garbage collection and CUDA cache clearing
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"Cleared CUDA cache after training session {session_id}")

            # Clear the session queue to prevent memory leaks
            if session_id in session_queues:
                q = session_queues[session_id]
                # Empty the queue
                while not q.empty():
                    try:
                        q.get_nowait()
                    except:
                        break
                logger.info(f"Cleared session queue for {session_id}")

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

        configs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
        configs_dir.mkdir(exist_ok=True)
        filepath = configs_dir / filename

        logger.info(f"Saving config to: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved successfully: {filename}")

        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath),
            'message': f'Configuration saved as {filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/load/<filename>', methods=['GET'])
def load_config(filename):
    """Load configuration from file."""
    try:
        configs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
        filepath = configs_dir / filename

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
        configs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
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
        configs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
        filepath = configs_dir / filename

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

        # Clean up any stale training sessions before starting new one
        # This helps prevent the hanging issue when training multiple models
        import gc
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


@app.route('/api/training/session/<session_id>/history', methods=['GET'])
def get_training_history(session_id):
    """Get training history for reconnection."""
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


@app.route('/api/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate model on test dataset."""
    try:
        data = request.json
        session_id = data.get('session_id')
        test_cases = data.get('test_cases', [])
        prompt_template = data.get('prompt_template', '{input}')
        config = data.get('config', {})

        if not session_id or not test_cases:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Load model
        checkpoint_path = f"outputs/{session_id}/checkpoints/final"
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': 'Model checkpoint not found'}), 404

        # Initialize model tester
        from core.model_tester import ModelTester
        tester = ModelTester()

        # Load the model
        tester.load_model(checkpoint_path)

        # Run evaluation
        results = []
        correct = 0
        total = len(test_cases)

        for test_case in test_cases:
            input_text = prompt_template.replace('{input}', test_case['input'])

            # Generate response
            output = tester.generate(
                input_text,
                temperature=config.get('temperature', 0.1),
                max_new_tokens=config.get('max_new_tokens', 256),
                top_p=config.get('top_p', 0.95)
            )

            # Check if output matches expected
            match = output.strip().lower() == test_case['expected'].strip().lower()
            if match:
                correct += 1

            results.append({
                'input': test_case['input'],
                'expected': test_case['expected'],
                'output': output,
                'match': match
            })

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        precision = correct / total if total > 0 else 0  # Simplified for demo
        recall = correct / total if total > 0 else 0  # Simplified for demo
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Clean up model
        tester.cleanup()

        return jsonify({
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total': total,
                'correct': correct
            },
            'details': results
        })

    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
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


@app.route('/api/models/<session_id>', methods=['DELETE'])
def delete_model(session_id):
    """Delete a model and all its associated data."""
    try:
        import shutil

        # Check if model is currently training
        if session_id in training_sessions and training_sessions[session_id].get('status') == 'running':
            return jsonify({'error': 'Cannot delete model while training is in progress'}), 400

        deleted_items = []

        # Delete model outputs directory
        output_path = Path(f"./outputs/{session_id}")
        if output_path.exists():
            shutil.rmtree(output_path)
            deleted_items.append(f"outputs/{session_id}")
            logger.info(f"Deleted outputs for session {session_id}")

        # Delete exports directory
        export_path = Path(f"./exports/{session_id}")
        if export_path.exists():
            shutil.rmtree(export_path)
            deleted_items.append(f"exports/{session_id}")
            logger.info(f"Deleted exports for session {session_id}")

        # Remove from training sessions if exists
        if session_id in training_sessions:
            del training_sessions[session_id]
            deleted_items.append("session data")

        return jsonify({
            'success': True,
            'message': f'Model {session_id} deleted successfully',
            'deleted': deleted_items
        })

    except Exception as e:
        logger.error(f"Failed to delete model {session_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<session_id>/<export_format>/<export_name>', methods=['DELETE'])
def delete_export(session_id, export_format, export_name):
    """Delete a specific export."""
    try:
        import shutil

        # Construct the export path
        export_path = Path(f"./exports/{session_id}/{export_format}/{export_name}")

        if not export_path.exists():
            return jsonify({'error': 'Export not found'}), 404

        # Delete the export (file or directory)
        if export_path.is_dir():
            shutil.rmtree(export_path)
        else:
            export_path.unlink()

        logger.info(f"Deleted export: {export_path}")

        # Check if this was the last export in the format directory
        format_dir = export_path.parent
        if format_dir.exists() and not any(format_dir.iterdir()):
            format_dir.rmdir()
            logger.info(f"Removed empty format directory: {format_dir}")

        # Check if this was the last export for the session
        session_dir = format_dir.parent
        if session_dir.exists() and not any(session_dir.iterdir()):
            session_dir.rmdir()
            logger.info(f"Removed empty session directory: {session_dir}")

        return jsonify({
            'success': True,
            'message': f'Export {export_name} deleted successfully'
        })

    except Exception as e:
        logger.error(f"Failed to delete export {export_name}: {str(e)}")
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
        custom_field_mapping = data.get('field_mapping', None)  # Custom field mapping from user

        if not dataset_name:
            return jsonify({'error': 'Dataset name required'}), 400

        # Create session for progress tracking
        session_id = create_session_id()

        def progress_callback(info):
            # Emit progress via SocketIO
            emit_to_session(session_id, 'dataset_progress', info)

        # Check if dataset has a custom default split or field mapping
        default_split = 'train'
        dataset_info = POPULAR_DATASETS.get(dataset_name, {})
        if 'default_split' in dataset_info:
            default_split = dataset_info['default_split']

        # Get field mappings - prefer custom, then predefined, then default
        if custom_field_mapping:
            # Use custom field mapping from user
            instruction_field = custom_field_mapping.get('instruction', 'instruction')
            response_field = custom_field_mapping.get('response', 'response')
        else:
            # Use predefined mapping if available
            field_mapping = dataset_info.get('field_mapping', {})
            instruction_field = field_mapping.get('instruction', 'instruction')
            response_field = field_mapping.get('response', 'response')

        # Create dataset config
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            split=default_split,  # Use custom split if specified
            subset=dataset_config,  # This maps to 'name' parameter in HF load_dataset
            instruction_field=instruction_field,
            response_field=response_field,
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

        # Check if dataset has a custom default split or field mapping
        default_split = 'train'
        dataset_info = POPULAR_DATASETS.get(dataset_name, {})
        if 'default_split' in dataset_info:
            default_split = dataset_info['default_split']

        # Get field mappings if specified
        field_mapping = dataset_info.get('field_mapping', {})

        # Create config for sampling
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            split=default_split,  # Use custom split if specified
            subset=dataset_config,  # This maps to 'name' parameter in HF load_dataset
            instruction_field=field_mapping.get('instruction', 'instruction'),
            response_field=field_mapping.get('response', 'response'),
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


@app.route('/api/datasets/detect-fields', methods=['POST'])
def detect_dataset_fields():
    """Detect available fields in a dataset for mapping."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        dataset_config = data.get('config', None)
        is_local = data.get('is_local', False)

        if not dataset_name:
            return jsonify({'error': 'Dataset name required'}), 400

        # Check if this is an uploaded file
        if is_local or dataset_name.startswith(app.config['UPLOAD_FOLDER']) or '\\uploads\\' in dataset_name or '/uploads/' in dataset_name:
            # Handle local uploaded file
            filepath = Path(dataset_name)

            # Ensure file exists
            if not filepath.exists():
                return jsonify({'error': 'File not found'}), 404

            # Read the file to detect columns
            columns = []
            suggested_mappings = {'instruction': None, 'response': None}
            sample_data = []

            extension = filepath.suffix[1:].lower()

            try:
                if extension == 'csv':
                    import pandas as pd
                    df = pd.read_csv(filepath, nrows=5)
                    columns = list(df.columns)
                    sample_data = df.head(3).to_dict('records')

                elif extension == 'json':
                    import json
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data_json = json.load(f)
                        if isinstance(data_json, list) and len(data_json) > 0:
                            columns = list(data_json[0].keys())
                            sample_data = data_json[:3]

                elif extension == 'jsonl':
                    import json
                    with open(filepath, 'r', encoding='utf-8') as f:
                        first_lines = []
                        for i, line in enumerate(f):
                            if i >= 3:
                                break
                            first_lines.append(json.loads(line))
                        if first_lines:
                            columns = list(first_lines[0].keys())
                            sample_data = first_lines

                elif extension == 'parquet':
                    import pandas as pd
                    df = pd.read_parquet(filepath, engine='auto')
                    columns = list(df.columns)
                    sample_data = df.head(3).to_dict('records')

                # Detect instruction/response fields
                instruction_patterns = ['instruction', 'prompt', 'question', 'input', 'text', 'problem', 'query', 'user']
                response_patterns = ['response', 'answer', 'output', 'completion', 'label', 'generated_solution', 'solution', 'reply', 'assistant']

                for col in columns:
                    col_lower = col.lower()
                    if not suggested_mappings['instruction']:
                        for pattern in instruction_patterns:
                            if pattern in col_lower:
                                suggested_mappings['instruction'] = col
                                break
                    if not suggested_mappings['response']:
                        for pattern in response_patterns:
                            if pattern in col_lower:
                                suggested_mappings['response'] = col
                                break

                return jsonify({
                    'columns': columns,
                    'suggested_mappings': suggested_mappings,
                    'sample_data': sample_data,
                    'is_local': True
                })

            except Exception as e:
                logger.error(f"Failed to read local file: {e}")
                return jsonify({'error': f'Failed to read file: {str(e)}'}), 500

        # Check if dataset has predefined field mapping
        dataset_info = POPULAR_DATASETS.get(dataset_name, {})
        default_split = dataset_info.get('default_split', 'train')

        # Create minimal config to load just column names
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            split=default_split,
            subset=dataset_config,
            sample_size=1,  # Load minimal data
            use_cache=False
        )

        try:
            # Load a tiny sample to get column names
            from datasets import load_dataset

            # Special handling for known problematic datasets
            if 'squad' in dataset_name.lower():
                # Use the standard 'squad' dataset instead of variations
                dataset_name = 'squad'
                default_split = 'train'

            kwargs = {
                'path': dataset_name,
                'split': f"{default_split}[:1]",  # Load just 1 sample
                'streaming': False,
                'trust_remote_code': False
            }

            if dataset_config:
                kwargs['name'] = dataset_config

            # Try to load
            try:
                dataset = load_dataset(**kwargs)
            except Exception as load_error:
                # If loading fails, try without the slice notation
                kwargs['split'] = default_split
                dataset = load_dataset(**kwargs)
                if hasattr(dataset, 'select'):
                    dataset = dataset.select(range(1))  # Get just one sample

            # Get column names
            columns = []
            if hasattr(dataset, 'column_names'):
                columns = dataset.column_names
            elif hasattr(dataset, 'features'):
                columns = list(dataset.features.keys())

            # Suggest mappings based on common patterns
            suggested_mappings = {
                'instruction': None,
                'response': None
            }

            # Check for instruction field
            instruction_patterns = ['instruction', 'prompt', 'question', 'input', 'text', 'problem', 'query']
            for col in columns:
                col_lower = col.lower()
                for pattern in instruction_patterns:
                    if pattern in col_lower:
                        suggested_mappings['instruction'] = col
                        break
                if suggested_mappings['instruction']:
                    break

            # Check for response field
            response_patterns = ['response', 'answer', 'output', 'completion', 'label', 'generated_solution', 'solution', 'reply']
            for col in columns:
                col_lower = col.lower()
                for pattern in response_patterns:
                    if pattern in col_lower:
                        suggested_mappings['response'] = col
                        break
                if suggested_mappings['response']:
                    break

            # Get a sample to show data preview
            sample_data = {}
            if len(dataset) > 0:
                sample = dataset[0]
                for col in columns[:5]:  # Show first 5 columns
                    if col in sample:
                        value = str(sample[col])[:100]  # Truncate long values
                        sample_data[col] = value

            return jsonify({
                'columns': columns,
                'suggested_mappings': suggested_mappings,
                'sample_data': sample_data,
                'predefined_mapping': dataset_info.get('field_mapping', {})
            })

        except ValueError as e:
            # Handle split errors
            error_msg = str(e)
            if "Unknown split" in error_msg or "Should be one of" in error_msg:
                # Try to extract available splits and retry with first one
                import re
                splits_match = re.search(r"Should be one of \[(.+?)\]", error_msg)
                if splits_match:
                    available_splits = [s.strip().strip("'") for s in splits_match.group(1).split(',')]
                    # Retry with first available split
                    kwargs['split'] = f"{available_splits[0]}[:1]"
                    dataset = load_dataset(**kwargs)

                    columns = []
                    if hasattr(dataset, 'column_names'):
                        columns = dataset.column_names
                    elif hasattr(dataset, 'features'):
                        columns = list(dataset.features.keys())

                    return jsonify({
                        'columns': columns,
                        'available_splits': available_splits,
                        'used_split': available_splits[0]
                    })
            raise

    except Exception as e:
        logger.error(f"Failed to detect dataset fields: {e}")
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
        # Check cache status for each dataset
        handler = DatasetHandler()

        datasets_with_status = []
        for dataset_path, info in POPULAR_DATASETS.items():
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


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """Upload a dataset file (JSON, JSONL, CSV, or Parquet)."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400

        # Generate secure filename with timestamp to avoid conflicts
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = original_filename.rsplit('.', 1)[0]
        extension = original_filename.rsplit('.', 1)[1]
        filename = f"{base_name}_{timestamp}.{extension}"

        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get file info
        file_size = os.path.getsize(filepath)

        # Try to detect dataset structure
        dataset_info = {}
        try:
            # Create temporary dataset handler to analyze the file
            temp_config = DatasetConfig(
                source_type='local',
                source_path=filepath,
                max_samples=5  # Only load a few samples for preview
            )
            temp_handler = DatasetHandler(temp_config)
            temp_dataset = temp_handler.load()

            # Get field information
            if hasattr(temp_dataset, 'column_names'):
                dataset_info['columns'] = temp_dataset.column_names
                dataset_info['num_samples'] = len(temp_dataset)

                # Get sample data
                samples = []
                for i, item in enumerate(temp_dataset):
                    if i >= 3:  # Only get 3 samples
                        break
                    samples.append(dict(item))
                dataset_info['samples'] = samples

                # Try to auto-detect instruction/response fields
                columns_lower = [col.lower() for col in temp_dataset.column_names]
                instruction_candidates = ['instruction', 'prompt', 'question', 'input', 'text', 'problem']
                response_candidates = ['response', 'answer', 'output', 'completion', 'label', 'solution']

                detected_instruction = None
                detected_response = None

                for candidate in instruction_candidates:
                    if candidate in columns_lower:
                        idx = columns_lower.index(candidate)
                        detected_instruction = temp_dataset.column_names[idx]
                        break

                for candidate in response_candidates:
                    if candidate in columns_lower:
                        idx = columns_lower.index(candidate)
                        detected_response = temp_dataset.column_names[idx]
                        break

                if detected_instruction:
                    dataset_info['detected_instruction_field'] = detected_instruction
                if detected_response:
                    dataset_info['detected_response_field'] = detected_response

        except Exception as e:
            logger.warning(f"Could not analyze dataset structure: {e}")
            dataset_info['error'] = str(e)

        logger.info(f"Dataset uploaded successfully: {filename} ({file_size / 1024 / 1024:.2f} MB)")

        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'original_filename': original_filename,
            'size_mb': round(file_size / 1024 / 1024, 2),
            'dataset_info': dataset_info
        })

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/uploaded', methods=['GET'])
def list_uploaded_datasets():
    """List all uploaded dataset files."""
    try:
        uploaded_files = []
        upload_dir = Path(app.config['UPLOAD_FOLDER'])

        if upload_dir.exists():
            for filepath in upload_dir.iterdir():
                if filepath.is_file() and filepath.suffix[1:] in app.config['ALLOWED_EXTENSIONS']:
                    file_info = {
                        'filename': filepath.name,
                        'filepath': str(filepath),
                        'relative_path': f"uploads/{filepath.name}",
                        'size_mb': round(filepath.stat().st_size / 1024 / 1024, 2),
                        'uploaded_at': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                        'extension': filepath.suffix[1:]
                    }
                    uploaded_files.append(file_info)

        # Sort by upload date (newest first)
        uploaded_files.sort(key=lambda x: x['uploaded_at'], reverse=True)

        return jsonify({'files': uploaded_files})

    except Exception as e:
        logger.error(f"Failed to list uploaded datasets: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/uploaded/<filename>', methods=['DELETE'])
def delete_uploaded_dataset(filename):
    """Delete an uploaded dataset file."""
    try:
        # Validate filename
        filename = secure_filename(filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename

        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404

        # Check if file is in upload directory (security check)
        if not str(filepath).startswith(app.config['UPLOAD_FOLDER']):
            return jsonify({'error': 'Invalid file path'}), 403

        # Delete the file
        filepath.unlink()
        logger.info(f"Deleted uploaded dataset: {filename}")

        return jsonify({
            'success': True,
            'message': f'File {filename} deleted successfully'
        })

    except Exception as e:
        logger.error(f"Failed to delete uploaded dataset: {e}")
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
        configs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
        configs_dir.mkdir(exist_ok=True)

        template_path = configs_dir / f"chat_template_{safe_name}.json"
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
        configs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
        if configs_dir.exists():
            # Look for template files (those with template_ prefix)
            for template_file in configs_dir.glob('template_*.json'):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        # Remove the 'template_' prefix from the key
                        template_key = template_file.stem.replace('template_', '')
                        custom_templates[template_key] = template_data
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
        configs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
        configs_dir.mkdir(exist_ok=True)

        template_path = configs_dir / f"template_{safe_name}.json"
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


@app.route('/api/test/compare-models', methods=['POST'])
def compare_two_trained_models():
    """Compare responses from two trained models."""
    try:
        data = request.json
        prompt = data.get('prompt')
        model1_session_id = data.get('model1_session_id')
        model2_session_id = data.get('model2_session_id')

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

        if not prompt or not model1_session_id or not model2_session_id:
            return jsonify({'error': 'prompt, model1_session_id, and model2_session_id required'}), 400

        # Load first model
        session1_info = session_registry.get_session(model1_session_id)
        if not session1_info:
            return jsonify({'error': f'Session {model1_session_id} not found'}), 404

        checkpoint1_path = session1_info.checkpoint_path
        if not checkpoint1_path or not Path(checkpoint1_path).exists():
            return jsonify({'error': f'Checkpoint for model 1 not found'}), 404

        # Load second model
        session2_info = session_registry.get_session(model2_session_id)
        if not session2_info:
            return jsonify({'error': f'Session {model2_session_id} not found'}), 404

        checkpoint2_path = session2_info.checkpoint_path
        if not checkpoint2_path or not Path(checkpoint2_path).exists():
            return jsonify({'error': f'Checkpoint for model 2 not found'}), 404

        # Load models if not already loaded
        model1_cache_key = f"trained_{model1_session_id}"
        model2_cache_key = f"trained_{model2_session_id}"

        if model1_cache_key not in model_tester.loaded_models:
            success, error = model_tester.load_trained_model(checkpoint1_path, model1_session_id)
            if not success:
                return jsonify({'error': f'Failed to load model 1: {error}'}), 500

        if model2_cache_key not in model_tester.loaded_models:
            success, error = model_tester.load_trained_model(checkpoint2_path, model2_session_id)
            if not success:
                return jsonify({'error': f'Failed to load model 2: {error}'}), 500

        # Compare the two models
        results = model_tester.compare_two_models(
            prompt=prompt,
            model1_session_id=model1_session_id,
            model2_session_id=model2_session_id,
            config=config,
            use_chat_template=use_chat_template
        )

        # Add session info to results
        results['model1']['session_info'] = {
            'name': session1_info.name,
            'model_name': session1_info.model_name,
            'dataset_name': session1_info.dataset_name
        }
        results['model2']['session_info'] = {
            'name': session2_info.name,
            'model_name': session2_info.model_name,
            'dataset_name': session2_info.dataset_name
        }

        return jsonify(results)

    except Exception as e:
        logger.error(f"Failed to compare two models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/compare/stream', methods=['POST'])
def compare_models_stream():
    """Stream model comparison responses using Server-Sent Events."""
    try:
        data = request.json
        prompt = data.get('prompt')
        session_id = data.get('session_id')
        base_model = data.get('base_model')
        config_data = data.get('config', {})
        use_chat_template = data.get('use_chat_template', True)

        def generate():
            try:
                # Load models first if needed
                checkpoint_path = Path(f"outputs/{session_id}/checkpoints/final")
                if not checkpoint_path.exists():
                    checkpoint_path = Path(f"outputs/{session_id}/checkpoint-final")

                success, error = model_tester.load_trained_model(str(checkpoint_path), session_id)
                if not success:
                    yield f"data: {json.dumps({'type': 'error', 'error': error})}\n\n"
                    return

                success, error = model_tester.load_base_model(base_model)
                if not success:
                    yield f"data: {json.dumps({'type': 'error', 'error': error})}\n\n"
                    return

                # Generate from trained model with streaming
                config = TestConfig(
                    temperature=config_data.get('temperature', 0.7),
                    max_new_tokens=config_data.get('max_new_tokens', 512),
                    top_p=config_data.get('top_p', 0.95),
                    top_k=config_data.get('top_k', 50),
                    repetition_penalty=config_data.get('repetition_penalty', 1.0),
                    do_sample=config_data.get('do_sample', True)
                )

                # Use a queue to collect streaming tokens
                from queue import Queue
                import threading
                import time

                trained_queue = Queue()
                base_queue = Queue()

                def trained_callback(token, is_end):
                    trained_queue.put(('token', token))
                    if is_end:
                        trained_queue.put(('end', None))

                def base_callback(token, is_end):
                    base_queue.put(('token', token))
                    if is_end:
                        base_queue.put(('end', None))

                # Start generation in threads
                trained_result = {'success': False}
                base_result = {'success': False}

                def generate_trained():
                    nonlocal trained_result
                    trained_result = model_tester.generate_response(
                        prompt=prompt,
                        model_type="trained",
                        model_key=session_id,
                        config=config,
                        use_chat_template=use_chat_template,
                        streaming_callback=trained_callback
                    )

                def generate_base():
                    nonlocal base_result
                    base_result = model_tester.generate_response(
                        prompt=prompt,
                        model_type="base",
                        model_key=base_model,
                        config=config,
                        use_chat_template=use_chat_template,
                        streaming_callback=base_callback
                    )

                trained_thread = threading.Thread(target=generate_trained)
                base_thread = threading.Thread(target=generate_base)

                trained_thread.start()
                base_thread.start()

                # Stream tokens as they arrive
                trained_done = False
                base_done = False

                while not (trained_done and base_done):
                    # Check trained model queue
                    if not trained_done and not trained_queue.empty():
                        msg_type, token = trained_queue.get()
                        if msg_type == 'token':
                            yield f"data: {json.dumps({'type': 'trained', 'token': token})}\n\n"
                        else:
                            trained_done = True
                            yield f"data: {json.dumps({'type': 'trained_complete'})}\n\n"

                    # Check base model queue
                    if not base_done and not base_queue.empty():
                        msg_type, token = base_queue.get()
                        if msg_type == 'token':
                            yield f"data: {json.dumps({'type': 'base', 'token': token})}\n\n"
                        else:
                            base_done = True
                            yield f"data: {json.dumps({'type': 'base_complete'})}\n\n"

                    # Small delay to avoid busy-waiting
                    time.sleep(0.01)

                # Wait for threads to complete
                trained_thread.join()
                base_thread.join()

                # Send final metadata
                yield f"data: {json.dumps({'type': 'complete', 'trained': trained_result, 'base': base_result})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Stream comparison failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/compare-models/stream', methods=['POST'])
def compare_two_models_stream():
    """Stream comparison between two trained models using SSE."""
    try:
        data = request.json
        prompt = data.get('prompt')
        model1_session_id = data.get('model1_session_id')
        model2_session_id = data.get('model2_session_id')
        config_data = data.get('config', {})
        use_chat_template = data.get('use_chat_template', True)

        def generate():
            try:
                # Load both models first if needed
                checkpoint1_path = Path(f"outputs/{model1_session_id}/checkpoints/final")
                if not checkpoint1_path.exists():
                    checkpoint1_path = Path(f"outputs/{model1_session_id}/checkpoint-final")

                checkpoint2_path = Path(f"outputs/{model2_session_id}/checkpoints/final")
                if not checkpoint2_path.exists():
                    checkpoint2_path = Path(f"outputs/{model2_session_id}/checkpoint-final")

                success, error = model_tester.load_trained_model(str(checkpoint1_path), model1_session_id)
                if not success:
                    yield f"data: {json.dumps({'type': 'error', 'error': error})}\n\n"
                    return

                success, error = model_tester.load_trained_model(str(checkpoint2_path), model2_session_id)
                if not success:
                    yield f"data: {json.dumps({'type': 'error', 'error': error})}\n\n"
                    return

                config = TestConfig(
                    temperature=config_data.get('temperature', 0.7),
                    max_new_tokens=config_data.get('max_new_tokens', 512),
                    top_p=config_data.get('top_p', 0.95),
                    top_k=config_data.get('top_k', 50),
                    repetition_penalty=config_data.get('repetition_penalty', 1.0),
                    do_sample=config_data.get('do_sample', True)
                )

                # Use a queue to collect streaming tokens
                from queue import Queue
                import threading
                import time

                model1_queue = Queue()
                model2_queue = Queue()

                def model1_callback(token, is_end):
                    model1_queue.put(('token', token))
                    if is_end:
                        model1_queue.put(('end', None))

                def model2_callback(token, is_end):
                    model2_queue.put(('token', token))
                    if is_end:
                        model2_queue.put(('end', None))

                # Start generation in threads
                model1_result = {'success': False}
                model2_result = {'success': False}

                def generate_model1():
                    nonlocal model1_result
                    model1_result = model_tester.generate_response(
                        prompt=prompt,
                        model_type="trained",
                        model_key=model1_session_id,
                        config=config,
                        use_chat_template=use_chat_template,
                        streaming_callback=model1_callback
                    )

                def generate_model2():
                    nonlocal model2_result
                    model2_result = model_tester.generate_response(
                        prompt=prompt,
                        model_type="trained",
                        model_key=model2_session_id,
                        config=config,
                        use_chat_template=use_chat_template,
                        streaming_callback=model2_callback
                    )

                model1_thread = threading.Thread(target=generate_model1)
                model2_thread = threading.Thread(target=generate_model2)

                model1_thread.start()
                model2_thread.start()

                # Stream tokens as they arrive
                model1_done = False
                model2_done = False

                while not (model1_done and model2_done):
                    # Check model 1 queue
                    if not model1_done and not model1_queue.empty():
                        msg_type, token = model1_queue.get()
                        if msg_type == 'token':
                            yield f"data: {json.dumps({'type': 'model1', 'token': token})}\n\n"
                        else:
                            model1_done = True
                            yield f"data: {json.dumps({'type': 'model1_complete'})}\n\n"

                    # Check model 2 queue
                    if not model2_done and not model2_queue.empty():
                        msg_type, token = model2_queue.get()
                        if msg_type == 'token':
                            yield f"data: {json.dumps({'type': 'model2', 'token': token})}\n\n"
                        else:
                            model2_done = True
                            yield f"data: {json.dumps({'type': 'model2_complete'})}\n\n"

                    # Small delay to avoid busy-waiting
                    time.sleep(0.01)

                # Wait for threads to complete
                model1_thread.join()
                model2_thread.join()

                # Send final metadata
                yield f"data: {json.dumps({'type': 'complete', 'model1': model1_result, 'model2': model2_result})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Stream model comparison failed: {str(e)}")
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


@app.route('/api/test/upload-file', methods=['POST'])
def upload_test_file():
    """Upload and analyze a test file for batch testing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        if ext not in ['csv', 'json', 'jsonl']:
            return jsonify({'error': 'Invalid file format. Supported: CSV, JSON, JSONL'}), 400

        # Save file temporarily
        import tempfile
        import pandas as pd

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Parse file based on extension
            if ext == 'csv':
                df = pd.read_csv(temp_path)
                columns = df.columns.tolist()
                sample_count = len(df)
                samples = df.head(5).to_dict('records')
            elif ext == 'json':
                with open(temp_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    columns = df.columns.tolist()
                    sample_count = len(data)
                    samples = data[:5]
                else:
                    return jsonify({'error': 'JSON file must contain an array of objects'}), 400
            elif ext == 'jsonl':
                with open(temp_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                df = pd.DataFrame(data)
                columns = df.columns.tolist()
                sample_count = len(data)
                samples = data[:5]

            # Store file data in session for later use
            session['batch_test_file'] = {
                'path': temp_path,
                'format': ext,
                'columns': columns,
                'sample_count': sample_count
            }

            return jsonify({
                'success': True,
                'columns': columns,
                'sample_count': sample_count,
                'samples': samples,
                'filename': filename
            })

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except Exception as e:
        logger.error(f"Failed to process test file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test/batch-compare', methods=['POST'])
def batch_compare_models():
    """Run batch comparison between trained and base/comparison models."""
    try:
        data = request.json

        # Get file data from session
        if 'batch_test_file' not in session:
            return jsonify({'error': 'No test file uploaded'}), 400

        file_info = session['batch_test_file']
        temp_path = file_info['path']

        if not os.path.exists(temp_path):
            return jsonify({'error': 'Test file no longer available. Please upload again.'}), 400

        # Extract parameters
        session_id = data.get('session_id')
        base_model = data.get('base_model')
        instruction_column = data.get('instruction_column')
        response_column = data.get('response_column')  # Optional
        sample_size = data.get('sample_size')  # None means all
        use_chat_template = data.get('use_chat_template', True)
        compare_mode = data.get('compare_mode', 'base')  # 'base' or 'model'
        comparison_session_id = data.get('comparison_session_id')  # For model vs model

        # Validate required parameters
        if not session_id or not instruction_column:
            return jsonify({'error': 'Missing required parameters'}), 400

        if compare_mode == 'base' and not base_model:
            return jsonify({'error': 'Base model required for comparison'}), 400
        elif compare_mode == 'model' and not comparison_session_id:
            return jsonify({'error': 'Comparison model required'}), 400

        # Load test data
        import pandas as pd

        if file_info['format'] == 'csv':
            df = pd.read_csv(temp_path)
        elif file_info['format'] == 'json':
            with open(temp_path, 'r', encoding='utf-8') as f:
                df = pd.DataFrame(json.load(f))
        elif file_info['format'] == 'jsonl':
            with open(temp_path, 'r', encoding='utf-8') as f:
                df = pd.DataFrame([json.loads(line) for line in f])

        # Apply sample size limit if specified
        if sample_size:
            df = df.head(sample_size)

        # Validate columns exist
        if instruction_column not in df.columns:
            return jsonify({'error': f'Instruction column "{instruction_column}" not found'}), 400

        if response_column and response_column not in df.columns:
            return jsonify({'error': f'Response column "{response_column}" not found'}), 400

        # Get checkpoint path from registry
        session_info = session_registry.get_session(session_id)
        if not session_info:
            return jsonify({'error': f'Session {session_id} not found'}), 404

        checkpoint_path = session_info.checkpoint_path
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return jsonify({'error': f'Checkpoint not found for session {session_id}'}), 404

        # Load trained model
        success, error = model_tester.load_trained_model(checkpoint_path, session_id)
        if not success:
            return jsonify({'error': f'Failed to load trained model: {error}'}), 500

        # Load comparison model
        if compare_mode == 'base':
            success, error = model_tester.load_base_model(base_model)
            if not success:
                return jsonify({'error': f'Failed to load base model: {error}'}), 500
        else:
            # Load another trained model for comparison
            comp_session_info = session_registry.get_session(comparison_session_id)
            if not comp_session_info:
                return jsonify({'error': f'Comparison session {comparison_session_id} not found'}), 404

            comp_checkpoint = comp_session_info.checkpoint_path
            if not comp_checkpoint or not Path(comp_checkpoint).exists():
                return jsonify({'error': f'Checkpoint not found for comparison session {comparison_session_id}'}), 404

            success, error = model_tester.load_trained_model(comp_checkpoint, comparison_session_id)
            if not success:
                return jsonify({'error': f'Failed to load comparison model: {error}'}), 500

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

        # Process batch
        results = []
        for idx, row in df.iterrows():
            instruction = str(row[instruction_column])
            expected = str(row[response_column]) if response_column else None

            # Generate from trained model
            trained_response = model_tester.generate_response(
                prompt=instruction,
                model_type="trained",
                model_key=session_id,
                config=config,
                use_chat_template=use_chat_template
            )

            # Generate from comparison model
            if compare_mode == 'base':
                comparison_response = model_tester.generate_response(
                    prompt=instruction,
                    model_type="base",
                    model_key=base_model,
                    config=config,
                    use_chat_template=use_chat_template
                )
            else:
                comparison_response = model_tester.generate_response(
                    prompt=instruction,
                    model_type="trained",
                    model_key=comparison_session_id,
                    config=config,
                    use_chat_template=use_chat_template
                )

            result_item = {
                'index': idx,
                'instruction': instruction,
                'trained_response': trained_response.get('response', '') if trained_response.get('success') else 'ERROR',
                'comparison_response': comparison_response.get('response', '') if comparison_response.get('success') else 'ERROR',
                'trained_time': trained_response.get('metadata', {}).get('generation_time', 0),
                'comparison_time': comparison_response.get('metadata', {}).get('generation_time', 0)
            }

            # Add expected response and calculate match if available
            if expected:
                result_item['expected'] = expected
                result_item['trained_match'] = result_item['trained_response'].strip().lower() == expected.strip().lower()
                result_item['comparison_match'] = result_item['comparison_response'].strip().lower() == expected.strip().lower()

            results.append(result_item)

        # Calculate summary statistics
        total_samples = len(results)
        avg_trained_time = sum(r['trained_time'] for r in results) / total_samples if total_samples > 0 else 0
        avg_comparison_time = sum(r['comparison_time'] for r in results) / total_samples if total_samples > 0 else 0
        avg_trained_length = sum(len(r['trained_response']) for r in results) / total_samples if total_samples > 0 else 0
        avg_comparison_length = sum(len(r['comparison_response']) for r in results) / total_samples if total_samples > 0 else 0

        summary = {
            'total_samples': total_samples,
            'avg_trained_time': round(avg_trained_time, 3),
            'avg_comparison_time': round(avg_comparison_time, 3),
            'avg_trained_length': round(avg_trained_length, 1),
            'avg_comparison_length': round(avg_comparison_length, 1)
        }

        # Add accuracy metrics if expected responses provided
        if response_column:
            trained_matches = sum(1 for r in results if r.get('trained_match', False))
            comparison_matches = sum(1 for r in results if r.get('comparison_match', False))
            summary['trained_accuracy'] = round(trained_matches / total_samples * 100, 1) if total_samples > 0 else 0
            summary['comparison_accuracy'] = round(comparison_matches / total_samples * 100, 1) if total_samples > 0 else 0

        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        session.pop('batch_test_file', None)

        return jsonify({
            'success': True,
            'results': results,
            'summary': summary,
            'compare_mode': compare_mode
        })

    except Exception as e:
        logger.error(f"Batch comparison failed: {str(e)}")
        # Clean up on error
        if 'batch_test_file' in session:
            temp_path = session['batch_test_file'].get('path')
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            session.pop('batch_test_file', None)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Reward System API Routes
# ============================================================================

@app.route('/api/rewards/presets', methods=['GET'])
def get_reward_presets():
    """Get all available reward presets with metadata."""
    try:
        library = RewardPresetLibrary()
        return jsonify(library.to_dict())
    except Exception as e:
        logger.error(f"Failed to load reward presets: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rewards/templates', methods=['GET'])
def get_reward_templates():
    """Get all reward templates for quick start."""
    try:
        library = RewardTemplateLibrary()
        templates = {
            name: template.to_dict()
            for name, template in library.get_all_templates().items()
        }
        return jsonify({'templates': templates})
    except Exception as e:
        logger.error(f"Failed to load reward templates: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rewards/test', methods=['POST'])
def test_reward():
    """Test a reward configuration with sample inputs."""
    try:
        data = request.get_json()
        reward_config = data.get('reward_config')
        test_cases = data.get('test_cases', [])

        if not reward_config:
            return jsonify({'error': 'No reward configuration provided'}), 400

        if not test_cases:
            return jsonify({'error': 'No test cases provided'}), 400

        # Build the reward from configuration
        reward_builder = CustomRewardBuilder()

        if reward_config.get('type') == 'preset':
            # Use preset
            library = RewardPresetLibrary()
            preset_name = reward_config.get('preset_name')
            preset = library.get_preset(preset_name)

            if not preset:
                return jsonify({'error': f'Unknown preset: {preset_name}'}), 400

            reward_builder = preset.create_builder()
        else:
            # Build custom reward
            components = reward_config.get('components', [])

            for comp in components:
                comp_type = comp.get('type')
                weight = comp.get('weight', 1.0)

                if comp_type == 'binary':
                    pattern = comp.get('pattern')
                    reward_builder.add_binary_reward(
                        f"binary_{len(components)}",
                        regex_pattern=pattern,
                        weight=weight
                    )
                elif comp_type == 'numerical':
                    tolerance = comp.get('tolerance', 1e-6)
                    reward_builder.add_numerical_reward(
                        f"numerical_{len(components)}",
                        tolerance=tolerance,
                        weight=weight
                    )
                elif comp_type == 'length':
                    min_len = comp.get('min_length')
                    max_len = comp.get('max_length')
                    optimal_len = comp.get('optimal_length')
                    reward_builder.add_length_reward(
                        f"length_{len(components)}",
                        min_length=min_len,
                        max_length=max_len,
                        optimal_length=optimal_len,
                        weight=weight
                    )
                elif comp_type == 'format':
                    pattern = comp.get('pattern', r".*")
                    reward_builder.add_format_reward(
                        f"format_{len(components)}",
                        pattern=pattern,
                        weight=weight
                    )

        # Test the reward
        tester = RewardTester(reward_builder)

        # Run batch test if multiple cases
        if len(test_cases) > 1:
            results = tester.test_batch(test_cases)
        else:
            # Single test
            case = test_cases[0]
            result = tester.test_single(
                case.get('instruction', ''),
                case.get('generated', ''),
                case.get('reference')
            )
            results = {
                'results': [result],
                'statistics': {
                    'mean': result['total_reward'],
                    'std': 0,
                    'min': result['total_reward'],
                    'max': result['total_reward'],
                    'median': result['total_reward']
                }
            }

        return jsonify(results)

    except Exception as e:
        logger.error(f"Failed to test reward: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rewards/compare', methods=['POST'])
def compare_rewards():
    """Compare multiple responses using a reward function."""
    try:
        data = request.get_json()
        reward_config = data.get('reward_config')
        instruction = data.get('instruction')
        responses = data.get('responses', [])
        reference = data.get('reference')

        if not reward_config or not instruction or not responses:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Build the reward
        reward_builder = CustomRewardBuilder()

        if reward_config.get('type') == 'preset':
            library = RewardPresetLibrary()
            preset_name = reward_config.get('preset_name')
            preset = library.get_preset(preset_name)

            if not preset:
                return jsonify({'error': f'Unknown preset: {preset_name}'}), 400

            reward_builder = preset.create_builder()
        else:
            # Build custom reward (similar to test_reward)
            components = reward_config.get('components', [])
            for comp in components:
                comp_type = comp.get('type')
                weight = comp.get('weight', 1.0)

                if comp_type == 'binary':
                    pattern = comp.get('pattern')
                    reward_builder.add_binary_reward(
                        f"binary_{len(components)}",
                        regex_pattern=pattern,
                        weight=weight
                    )
                elif comp_type == 'numerical':
                    tolerance = comp.get('tolerance', 1e-6)
                    reward_builder.add_numerical_reward(
                        f"numerical_{len(components)}",
                        tolerance=tolerance,
                        weight=weight
                    )
                elif comp_type == 'length':
                    min_len = comp.get('min_length')
                    max_len = comp.get('max_length')
                    optimal_len = comp.get('optimal_length')
                    reward_builder.add_length_reward(
                        f"length_{len(components)}",
                        min_length=min_len,
                        max_length=max_len,
                        optimal_length=optimal_len,
                        weight=weight
                    )
                elif comp_type == 'format':
                    pattern = comp.get('pattern', r".*")
                    reward_builder.add_format_reward(
                        f"format_{len(components)}",
                        pattern=pattern,
                        weight=weight
                    )

        # Compare responses
        tester = RewardTester(reward_builder)
        results = tester.compare_responses(instruction, responses, reference)

        return jsonify({'comparisons': results})

    except Exception as e:
        logger.error(f"Failed to compare responses: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rewards/visualize', methods=['POST'])
def visualize_reward():
    """Get visualization data for a reward test result."""
    try:
        data = request.get_json()
        test_result = data.get('test_result')

        if not test_result:
            return jsonify({'error': 'No test result provided'}), 400

        # Create visualization
        tester = RewardTester(CustomRewardBuilder())  # Dummy builder for viz
        visualization = tester.visualize_components(test_result)

        return jsonify({
            'visualization': visualization,
            'chart_data': {
                'labels': list(test_result.get('components', {}).keys()),
                'values': list(test_result.get('components', {}).values())
            }
        })

    except Exception as e:
        logger.error(f"Failed to visualize reward: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rewards/recommend', methods=['POST'])
def recommend_reward():
    """Recommend a reward configuration based on task description."""
    try:
        data = request.get_json()
        task_description = data.get('task_description', '')
        example_input = data.get('example_input', '')
        example_output = data.get('example_output', '')

        # Simple keyword-based recommendation
        library = RewardPresetLibrary()

        # Search for matching presets
        keywords = task_description.lower().split()
        recommendations = []

        for preset_name, preset in library.presets.items():
            score = 0

            # Check for keyword matches
            for keyword in keywords:
                if keyword in preset.name.lower():
                    score += 2
                if keyword in preset.description.lower():
                    score += 1
                if any(keyword in tag.lower() for tag in preset.tags):
                    score += 1

            if score > 0:
                recommendations.append({
                    'preset': preset.to_dict(),
                    'score': score,
                    'reason': f"Matches {score} keywords from your description"
                })

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        # Also recommend templates
        template_library = RewardTemplateLibrary()
        template_recommendations = []

        for template_name, template in template_library.templates.items():
            if any(keyword in template.description.lower() for keyword in keywords):
                template_recommendations.append(template.to_dict())

        return jsonify({
            'preset_recommendations': recommendations[:5],  # Top 5
            'template_recommendations': template_recommendations[:3]  # Top 3
        })

    except Exception as e:
        logger.error(f"Failed to recommend reward: {e}")
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


@socketio.on('join_training_session')
def handle_join_session(data):
    """Handle client joining a training session room."""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        logger.info(f"Client {request.sid} joined session {session_id}")
        emit('joined_session', {'session_id': session_id})


@socketio.on('leave_training_session')
def handle_leave_session(data):
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
