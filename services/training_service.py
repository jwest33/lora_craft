"""
Training service for managing training sessions and execution.

This module contains the core training logic including:
- TrainingSession class for session state management
- run_training function for executing training in background threads
- Helper functions for session ID generation
"""

import os
import sys
import io
import re
import gc
import uuid
import queue
from datetime import datetime
from typing import Dict, Any, Optional
import torch

from core import (
    SystemConfig,
    GRPOModelTrainer,
    GRPOTrainingConfig,
    DatasetHandler,
    DatasetConfig,
    PromptTemplate,
    TemplateConfig,
    CustomRewardBuilder,
    SessionRegistry,
    SessionInfo,
)
from core.custom_rewards import RewardPresetLibrary
from utils.logging_config import get_logger

logger = get_logger(__name__)


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
        self.step_counter = 0  # Incremental step counter for metrics without step field

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


def create_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def run_training(session_id: str, config: Dict[str, Any],
                 training_sessions: Dict, session_queues: Dict,
                 system_config: SystemConfig, session_registry: SessionRegistry,
                 socketio, upload_folder: str):
    """
    Run training in background thread.

    Args:
        session_id: Unique session identifier
        config: Training configuration dictionary
        training_sessions: Dict mapping session IDs to TrainingSession objects
        session_queues: Dict mapping session IDs to their message queues
        system_config: SystemConfig instance
        session_registry: SessionRegistry instance for tracking completed sessions
        socketio: SocketIO instance for emitting events
        upload_folder: Path to uploaded files directory
    """

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

                                # Start with all metrics from the training log
                                processed_metrics = dict(metrics_dict)

                                # Add/override computed fields for backward compatibility
                                processed_metrics['step'] = step_value
                                processed_metrics['mean_reward'] = metrics_dict.get('reward', metrics_dict.get('rewards/reward_wrapper/mean', 0))

                                # Ensure learning_rate exists if not in metrics
                                if 'learning_rate' not in processed_metrics:
                                    processed_metrics['learning_rate'] = config.get('learning_rate', 2e-4)

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
        os.environ['TRANSFORMERS_NO_PROGRESS'] = '0'

        # Send initial status
        q.put(('log', f"Initializing training for session {session_id}"))

        # Unpack nested config structure from frontend
        # Frontend sends: {model: {...}, dataset: {...}, training: {...}, lora: {...}, template: {...}}
        # Backend expects flat structure for GRPOTrainingConfig
        model_config = config.get('model', {})
        dataset_config = config.get('dataset', {})
        training_config = config.get('training', {})
        lora_config = config.get('lora', {})
        grpo_config_data = config.get('grpo', {})
        template_config = config.get('template', {})
        pre_training_config = config.get('pre_training', {})
        output_config = config.get('output', {})

        # Debug: Log what we're receiving
        logger.info(f"=== DEBUG CONFIG ===")
        logger.info(f"training_config.max_new_tokens: {training_config.get('max_new_tokens')}")
        logger.info(f"top-level config.max_new_tokens: {config.get('max_new_tokens')}")
        logger.info(f"Full training_config keys: {list(training_config.keys())}")
        logger.info(f"=== END DEBUG ===")

        # Extract model name from nested structure, fall back to top-level for backward compatibility
        model_name = model_config.get('modelName') or config.get('model_name', 'unsloth/Qwen3-0.6B')

        # Log the selected model
        logger.info(f"Starting training with model: {model_name}")
        q.put(('log', f"Loading model: {model_name}"))

        # Create GRPO configuration with algorithm support
        grpo_config = GRPOTrainingConfig(
            # Model configuration
            model_name=model_name,

            # LoRA configuration (from lora config or old model config for backward compatibility)
            lora_r=lora_config.get('rank') or model_config.get('loraRank') or config.get('lora_rank', 16),
            lora_alpha=lora_config.get('alpha') or model_config.get('loraAlpha') or config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('dropout') or model_config.get('loraDropout') or config.get('lora_dropout', 0.0),
            lora_target_modules=lora_config.get('target_modules') or model_config.get('targetModules') or config.get('lora_target_modules', None),
            lora_bias=config.get('lora_bias', 'none'),

            # Training configuration (from nested training config or top-level for backward compatibility)
            num_train_epochs=training_config.get('num_epochs') or config.get('num_epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size') or config.get('batch_size', 1),
            gradient_accumulation_steps=training_config.get('gradient_accumulation') or config.get('gradient_accumulation_steps', 1),
            learning_rate=training_config.get('learning_rate') or config.get('learning_rate', 2e-4),
            warmup_steps=config.get('warmup_steps', 10),
            weight_decay=training_config.get('weight_decay') or config.get('weight_decay', 0.001),
            max_grad_norm=training_config.get('max_grad_norm') or config.get('max_grad_norm', 0.3),
            lr_scheduler_type=training_config.get('lr_schedule') or config.get('lr_scheduler_type', 'constant'),
            optim=config.get('optim', 'paged_adamw_32bit'),
            logging_steps=config.get('logging_steps', 1),  # Default to 1 for real-time frontend updates
            save_steps=output_config.get('save_steps') or config.get('save_steps', 100),
            eval_steps=output_config.get('eval_steps') or config.get('eval_steps', 100),
            seed=training_config.get('seed') or config.get('seed', 42),

            # Generation configuration
            max_sequence_length=config.get('max_sequence_length', 2048),
            max_new_tokens=training_config.get('max_new_tokens') if training_config.get('max_new_tokens') is not None else config.get('max_new_tokens', 512),
            temperature=grpo_config_data.get('temperature') or config.get('temperature', 0.7),
            top_p=grpo_config_data.get('top_p') or config.get('top_p', 0.95),
            top_k=grpo_config_data.get('top_k') or config.get('top_k', 50),
            repetition_penalty=grpo_config_data.get('repetition_penalty') or config.get('repetition_penalty', 1.0),
            # GRPO requires at least 2 generations to calculate advantages
            # Priority: grpo.num_generations > top-level num_generations > max(2, batch_size)
            # This ensures we always meet the minimum requirement
            num_generations_per_prompt=(
                grpo_config_data.get('num_generations') or
                config.get('num_generations') or
                max(2, training_config.get('batch_size') or config.get('batch_size', 2))
            ),
            num_generations=(
                grpo_config_data.get('num_generations') or
                config.get('num_generations') or
                max(2, training_config.get('batch_size') or config.get('batch_size', 2))
            ),  # For TRL compatibility

            # GRPO/Algorithm specific (from nested grpo config or top-level for backward compatibility)
            loss_type=config.get('loss_type', 'grpo'),
            importance_sampling_level=config.get('importance_sampling_level', 'token'),
            kl_penalty=grpo_config_data.get('kl_weight') or config.get('kl_penalty', 0.05),  # Increased default to prevent KL divergence
            clip_range=grpo_config_data.get('clip_range') or config.get('clip_range', 0.2),
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

            # Pre-training configuration (for format learning phase)
            # Handle both nested (from new frontend) and flat (from old code/API) config formats
            pre_training_epochs=(config.get('pre_training', {}).get('epochs') or
                                config.get('pre_training_epochs', 2)),  # Default: 2 epochs (matches official notebook)
            pre_training_max_samples=(config.get('pre_training', {}).get('max_samples') or
                                     config.get('pre_training_max_samples')),  # Separate from main max_samples
            pre_training_filter_by_length=(config.get('pre_training', {}).get('filter_by_length') or
                                          config.get('pre_training_filter_by_length', False)),
            pre_training_max_length_ratio=config.get('pre_training_max_length_ratio', 0.5),
            pre_training_learning_rate=(config.get('pre_training', {}).get('learning_rate') or
                                       config.get('pre_training_learning_rate', 5e-5)),

            # Adaptive pre-training configuration
            adaptive_pre_training=(config.get('pre_training', {}).get('adaptive') or
                                  config.get('adaptive_pre_training', True)),
            pre_training_min_success_rate=(config.get('pre_training', {}).get('min_success_rate') or
                                          config.get('pre_training_min_success_rate', 0.8)),
            pre_training_max_additional_epochs=(config.get('pre_training', {}).get('max_additional_epochs') or
                                               config.get('pre_training_max_additional_epochs', 3)),
            pre_training_validation_samples=(config.get('pre_training', {}).get('validation_samples') or
                                            config.get('pre_training_validation_samples', 5)),

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
            # Check if this is a reward sample (has 'type' == 'reward_sample')
            if metrics.get('type') == 'reward_sample':
                logger.debug(f"Reward sample callback for session {session_id}: step={metrics.get('step', 'N/A')}, reward={metrics.get('total_reward', 'N/A')}")
                q.put(('reward_sample', metrics))
            else:
                # Regular metrics
                logger.debug(f"Metrics callback invoked for session {session_id}: step={metrics.get('step', 'N/A')}, loss={metrics.get('loss', 'N/A')}, reward={metrics.get('mean_reward', 'N/A')}")
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

        # IMPORTANT: Dataset Sample Size Semantics
        # - max_samples in DatasetConfig: Samples PER EPOCH for main GRPO training
        # - pre_training_max_samples: SEPARATE limit for pre-training phase only
        # Example: max_samples=1000, pre_training_max_samples=100, num_epochs=3
        #   -> Pre-training: 100 samples for pre_training_epochs (default 2)
        #   -> Main training: 1000 samples per epoch × 3 epochs = 3000 training steps

        # Determine source type based on dataset_path (from nested dataset config or top-level)
        dataset_path = dataset_config.get('path') or config.get('dataset_path', 'tatsu-lab/alpaca')

        # Normalize dataset path - preserve forward slashes for HuggingFace datasets
        # This is critical on Windows where paths may have backslashes
        if dataset_path and '/' in dataset_path:
            # Check if this looks like a HuggingFace dataset (not a local path)
            if not any(dataset_path.startswith(prefix) for prefix in ('.', 'C:', '/', 'uploads')) and '\\uploads\\' not in dataset_path:
                # Replace backslashes with forward slashes for HuggingFace dataset names
                original_path = dataset_path
                dataset_path = dataset_path.replace('\\', '/')
                if original_path != dataset_path:
                    logger.info(f"Normalized HuggingFace dataset path: {original_path} -> {dataset_path}")
                    q.put(('log', f"Normalized dataset path to: {dataset_path}"))

        # Get source type from either the nested dataset config or top-level
        # Nested format (from saved configs): config.dataset.source
        # Flat format (from live frontend): config.dataset_source
        source_type = (dataset_config.get('source') or
                      config.get('dataset_source', 'huggingface'))

        # Safety check: Override source_type based on dataset_path characteristics
        if dataset_path:
            # Check if this is an uploaded file
            if (dataset_path.startswith(upload_folder) or
                '\\uploads\\' in dataset_path or '/uploads/' in dataset_path):
                source_type = 'local'
                q.put(('log', f"Using uploaded dataset file: {os.path.basename(dataset_path)}"))
            # Check if this looks like a HuggingFace dataset (contains '/' but no file extension)
            elif '/' in dataset_path and not any(dataset_path.endswith(ext) for ext in ['.json', '.jsonl', '.csv', '.parquet', '.txt']):
                # This looks like a HuggingFace dataset (e.g., 'tatsu-lab/alpaca')
                if source_type != 'huggingface':
                    logger.info(f"Overriding source_type from '{source_type}' to 'huggingface' based on path pattern: {dataset_path}")
                    q.put(('log', f"Detected HuggingFace dataset pattern, using source: huggingface"))
                source_type = 'huggingface'

        logger.info(f"Dataset source type: {source_type}, path: {dataset_path}")
        q.put(('log', f"Dataset source: {source_type}, path: {dataset_path}"))

        # Get sample size from nested dataset config or top-level
        dataset_sample_size = dataset_config.get('sample_size') or config.get('max_samples')
        # Convert 0 to None (0 means "all" in the UI)
        if dataset_sample_size == 0:
            dataset_sample_size = None

        # Get field mappings from nested dataset config or top-level
        instruction_field = (dataset_config.get('instruction_field') or
                           config.get('instruction_field', 'instruction'))
        response_field = (dataset_config.get('response_field') or
                        config.get('response_field', 'output'))

        dataset_config_obj = DatasetConfig(
            source_type=source_type,
            source_path=dataset_path,
            subset=config.get('dataset_config', None),  # Config for multi-config datasets
            split=config.get('dataset_split', 'train'),  # Use 'train' as default, let max_samples handle limiting
            instruction_field=instruction_field,
            response_field=response_field,
            max_samples=dataset_sample_size  # Don't default to 100, use None if not provided
        )

        dataset_handler = DatasetHandler(dataset_config_obj)
        dataset = dataset_handler.load()
        actual_samples = len(dataset)
        if dataset_sample_size:
            q.put(('log', f"Loaded dataset with {actual_samples} samples (max_samples: {dataset_sample_size})"))
        else:
            q.put(('log', f"Loaded dataset with {actual_samples} samples (no limit configured)"))

        # Setup prompt template with chat template (from nested template config or top-level)
        # CRITICAL: Use proper fallback chain to ensure custom system_prompt is used
        custom_system_prompt = template_config.get('system_prompt')
        if not custom_system_prompt:
            custom_system_prompt = config.get('system_prompt',
                'You are given a problem.\n'
                'Think about the problem and provide your working out.\n'
                'Place it between <start_working_out> and <end_working_out>.\n'
                'Then, provide your solution between <SOLUTION></SOLUTION>')

        template_config_obj = TemplateConfig(
            name="training",
            description="Training template",
            instruction_template=config.get('instruction_template', "{instruction}"),
            response_template=config.get('response_template', "{response}"),
            reasoning_start_marker=template_config.get('reasoning_start') or config.get('reasoning_start', '<start_working_out>'),
            reasoning_end_marker=template_config.get('reasoning_end') or config.get('reasoning_end', '<end_working_out>'),
            solution_start_marker=template_config.get('solution_start') or config.get('solution_start', '<SOLUTION>'),
            solution_end_marker=template_config.get('solution_end') or config.get('solution_end', '</SOLUTION>'),
            system_prompt=custom_system_prompt,
            chat_template=template_config.get('chat_template') or config.get('chat_template'),  # Add chat template from config
            # For custom templates, don't prepend reasoning marker since it's in the template itself
            # Only prepend for built-in templates that don't have it in {% if add_generation_prompt %}
            model_type=template_config.get('chat_template_type') or config.get('chat_template_type', 'grpo'),
            prepend_reasoning_start=((template_config.get('chat_template_type') or config.get('chat_template_type', 'grpo')) != 'custom')
        )
        template = PromptTemplate(template_config_obj)

        # Debug: Log reward configuration
        logger.info(f"=== REWARD CONFIG DEBUG ===")
        logger.info(f"reward_config from config: {config.get('reward_config')}")
        logger.info(f"reward from config: {config.get('reward')}")
        logger.info(f"All config keys: {list(config.keys())}")
        q.put(('log', "Setting up reward function..."))

        # Setup reward function based on configuration
        # Support both 'reward_config' (new) and 'reward' (legacy from config.js) for backward compatibility
        reward_builder = CustomRewardBuilder()
        reward_config = config.get('reward_config') or config.get('reward') or {'type': 'preset', 'preset_name': 'math'}

        logger.info(f"Resolved reward_config: {reward_config}")
        q.put(('log', f"Reward config type: {reward_config.get('type')}"))

        if reward_config.get('type') == 'preset':
            # Use RewardPresetLibrary for consistency with other endpoints
            library = RewardPresetLibrary()
            preset_name = reward_config.get('preset_name', 'math')
            logger.info(f"Looking up preset: {preset_name}")
            q.put(('log', f"Loading reward preset: {preset_name}"))
            preset = library.get_preset(preset_name)

            if preset:
                reward_builder = preset.create_builder()
                q.put(('log', f"Using reward preset: {preset_name}"))
            else:
                # Fallback to math preset if preset not found
                q.put(('log', f"Warning: Preset '{preset_name}' not found, falling back to math preset"))
                reward_builder.add_numerical_reward("numerical_accuracy", tolerance=1e-6, weight=0.7)
                reward_builder.add_format_reward("answer_format",
                    pattern=r"\\boxed\{[^}]+\}|Final Answer:.*|Answer:.*", weight=0.2)
                reward_builder.add_length_reward("length", min_length=10, max_length=500,
                    optimal_length=100, weight=0.1)
        else:
            # Custom reward configuration
            q.put(('log', f"Loading custom reward with {len(reward_config.get('components', []))} components"))
            components = reward_config.get('components', [])
            for idx, comp in enumerate(components):
                comp_type = comp.get('type')
                comp_name = comp.get('name', f"{comp_type}_{idx}")
                weight = comp.get('weight', 1.0)
                parameters = comp.get('parameters', {})

                logger.info(f"Adding custom component: {comp_name} (type={comp_type}, weight={weight}, params={list(parameters.keys())})")
                q.put(('log', f"  - {comp_name} ({comp_type}): weight={weight:.2f}"))

                # Smart detection based on component name and parameters
                # Priority: name-based > parameter-based > type-based fallback

                # 1. Detect signal_accuracy by name or unique parameters
                if (comp_name == 'signal_accuracy' or
                    'valid_signals' in parameters or
                    'direction_match_score' in parameters):
                    valid_signals = parameters.get('valid_signals', [])
                    direction_match_score = parameters.get('direction_match_score', 0.70)
                    logger.info(f"  → Using add_signal_accuracy with {len(valid_signals)} signals, direction_score={direction_match_score}")
                    reward_builder.add_signal_accuracy(
                        comp_name,
                        valid_signals=valid_signals,
                        direction_match_score=direction_match_score,
                        weight=weight
                    )

                # 2. Detect template_validation by name or unique parameters
                elif (comp_name in ['template_structure', 'template_validation'] or
                      'section_tags' in parameters or
                      'required_sections' in parameters):
                    section_tags = parameters.get('section_tags', [])
                    required_sections = parameters.get('required_sections', section_tags)
                    order_matters = parameters.get('order_matters', False)
                    logger.info(f"  → Using add_template_validation with tags={section_tags}, required={required_sections}")
                    reward_builder.add_template_validation(
                        comp_name,
                        section_tags=section_tags,
                        required_sections=required_sections,
                        order_matters=order_matters,
                        weight=weight
                    )

                # 3. Standard binary reward
                elif comp_type == 'binary':
                    pattern = parameters.get('pattern') or comp.get('pattern')
                    logger.info(f"  → Using add_binary_reward with pattern")
                    reward_builder.add_binary_reward(comp_name, regex_pattern=pattern, weight=weight)

                # 4. Numerical reward
                elif comp_type == 'numerical':
                    tolerance = parameters.get('tolerance', 1e-6)
                    logger.info(f"  → Using add_numerical_reward with tolerance={tolerance}")
                    reward_builder.add_numerical_reward(comp_name, tolerance=tolerance, weight=weight)

                # 5. Continuous/length reward
                elif comp_type in ['continuous', 'length']:
                    min_len = parameters.get('min_length')
                    max_len = parameters.get('max_length')
                    optimal_len = parameters.get('optimal_length')
                    if min_len is not None or max_len is not None:
                        logger.info(f"  → Using add_length_reward: min={min_len}, max={max_len}, optimal={optimal_len}")
                        reward_builder.add_length_reward(
                            comp_name,
                            min_length=min_len,
                            max_length=max_len,
                            optimal_length=optimal_len,
                            weight=weight
                        )
                    else:
                        logger.warning(f"  → Continuous component '{comp_name}' has no length parameters - skipping")

                # 6. Format reward
                elif comp_type == 'format':
                    pattern = parameters.get('pattern') or comp.get('pattern', r".*")
                    logger.info(f"  → Using add_format_reward")
                    reward_builder.add_format_reward(comp_name, pattern=pattern, weight=weight)

                else:
                    logger.warning(f"  → Unknown component type '{comp_type}' for '{comp_name}' - skipping")

        # Log algorithm selection based on importance_sampling_level
        importance_level = config.get('importance_sampling_level', 'token')
        algorithm_name = 'GSPO' if importance_level == 'sequence' else 'GRPO'
        q.put(('log', f"Using {algorithm_name} algorithm for training (importance_sampling_level: {importance_level})"))

        # Pre-training phase for format learning (if enabled)
        # Handle both nested and flat config formats
        enable_pre_training = (config.get('pre_training', {}).get('enabled') if 'pre_training' in config
                             else config.get('enable_pre_training', True))
        if enable_pre_training:
            q.put(('log', "Starting pre-training phase for format learning..."))

            # Get pre-training configuration (handle nested format)
            validate_format = (config.get('pre_training', {}).get('validate_format') if 'pre_training' in config
                             else config.get('validate_format', True))

            # NOTE: We pass the FULL dataset to pre_fine_tune, which will apply
            # its own filtering/sampling based on pre_training_max_samples and
            # pre_training_filter_by_length configuration. This matches the official
            # GRPO notebook where pre-training uses a filtered subset of the data.
            # The main GRPO training will use the full dataset (limited by max_samples).

            # Configure tokenizer with chat template
            if hasattr(trainer, 'tokenizer') and trainer.tokenizer:
                template.setup_for_unsloth(trainer.tokenizer)
                q.put(('log', f"Applied {config.get('chat_template_type', 'grpo')} chat template to tokenizer"))

            # Run adaptive or standard pre-fine-tuning
            # Simple mode is now the default (matching official Unsloth notebook)
            # Adaptive mode with validation can be enabled via config if needed
            adaptive_enabled = config.get('pre_training', {}).get('adaptive', False) if 'pre_training' in config else False

            if adaptive_enabled:
                q.put(('log', "Using adaptive pre-training with automatic format validation"))
                pre_metrics = trainer.adaptive_pre_fine_tune(dataset, template)

                # Report adaptive training results
                if isinstance(pre_metrics, dict) and 'final_success_rate' in pre_metrics:
                    q.put(('log', f"Pre-training completed:"))
                    q.put(('log', f"  Total epochs: {pre_metrics.get('total_epochs', '?')}"))
                    q.put(('log', f"  Additional epochs: {pre_metrics.get('additional_epochs', 0)}"))
                    q.put(('log', f"  Final success rate: {pre_metrics.get('final_success_rate', 0)*100:.1f}%"))
                    if pre_metrics.get('success'):
                        q.put(('log', "  ✅ Model successfully learned the format!"))
                    else:
                        q.put(('log', "  ⚠️  Warning: Format learning below target threshold"))
            else:
                q.put(('log', "Using standard pre-training (adaptive mode disabled)"))
                pre_metrics = trainer.pre_fine_tune(dataset, template)
                q.put(('log', f"Pre-training completed (see logs above for sample count and epochs)"))

            # Legacy validation code (only runs if adaptive is disabled AND validate_format is true)
            if not adaptive_enabled and validate_format:
                q.put(('log', "=" * 60))
                q.put(('log', "POST-PRETRAINING FORMAT VALIDATION TEST"))
                q.put(('log', "=" * 60))

                # Get a sample from the dataset to test with
                # CRITICAL: Use the configured instruction field name from dataset config
                test_sample = None
                test_instruction = "What is 2 + 2?"  # Fallback
                if len(dataset) > 0:
                    test_sample = dataset[0]
                    # Use the configured instruction field (e.g., 'input' for technical analysis)
                    # dataset_config is a dict at this point, not a DatasetConfig object
                    instruction_field = dataset_config.get('instruction_field', 'instruction')
                    test_instruction = test_sample.get(instruction_field,
                                                      test_sample.get('instruction', 'What is 2 + 2?'))

                q.put(('log', f"Test instruction: {test_instruction[:100]}..."))

                # Generate a sample response to check format
                try:
                    # CRITICAL: Use the chat template to format the prompt correctly
                    # This ensures the model sees the same format it was trained on
                    messages = [{'role': 'user', 'content': test_instruction}]
                    formatted_prompt = template.apply(messages, mode='inference')

                    q.put(('log', f"Formatted prompt (first 300 chars): {formatted_prompt[:300]}..."))

                    # Tokenize the properly formatted prompt
                    inputs = trainer.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(trainer.model.device)

                    # Generate response
                    with torch.no_grad():
                        outputs = trainer.model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=trainer.tokenizer.pad_token_id or trainer.tokenizer.eos_token_id
                        )

                    # Decode response
                    generated_text = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Extract the response part (everything after the prompt)
                    # Since we used the formatted prompt, we can safely extract what comes after it
                    if formatted_prompt in generated_text:
                        response = generated_text[len(formatted_prompt):].strip()
                    else:
                        # Fallback: try to find where the assistant's response starts
                        # Look for the reasoning start marker as the beginning of the response
                        reasoning_marker = template_config_obj.reasoning_start_marker
                        if reasoning_marker in generated_text:
                            response = generated_text[generated_text.index(reasoning_marker):].strip()
                        else:
                            response = generated_text

                    q.put(('log', f"\nGenerated response:\n{response[:500]}...\n"))

                    # Check format compliance using custom markers from template config
                    # Get markers from the template config we created earlier
                    reasoning_start = template_config_obj.reasoning_start_marker.lower()
                    reasoning_end = template_config_obj.reasoning_end_marker.lower()
                    solution_start = template_config_obj.solution_start_marker.lower()
                    solution_end = template_config_obj.solution_end_marker.lower()

                    response_lower = response.lower()
                    has_reasoning_markers = reasoning_start in response_lower and reasoning_end in response_lower
                    has_solution_markers = solution_start in response_lower and solution_end in response_lower

                    # Format is compliant if it has both reasoning and solution markers
                    format_ok = has_reasoning_markers and has_solution_markers

                    # Report results
                    q.put(('log', "Format Check Results:"))
                    q.put(('log', f"  Expected Reasoning Markers: {template_config_obj.reasoning_start_marker} ... {template_config_obj.reasoning_end_marker}"))
                    q.put(('log', f"  Expected Solution Markers: {template_config_obj.solution_start_marker} ... {template_config_obj.solution_end_marker}"))
                    q.put(('log', f"  Has reasoning markers: {has_reasoning_markers}"))
                    q.put(('log', f"  Has solution markers: {has_solution_markers}"))
                    q.put(('log', f"  Format Compliant: {'✓ YES' if format_ok else '✗ NO'}"))

                    if not format_ok:
                        q.put(('log', "  ⚠️  WARNING: Model may need more pre-training epochs!"))
                        if not has_reasoning_markers:
                            q.put(('log', f"    Missing: {template_config_obj.reasoning_start_marker} ... {template_config_obj.reasoning_end_marker}"))
                        if not has_solution_markers:
                            q.put(('log', f"    Missing: {template_config_obj.solution_start_marker} ... {template_config_obj.solution_end_marker}"))
                    else:
                        q.put(('log', "  ✅ Model has learned the expected format!"))

                    q.put(('log', "=" * 60))

                except Exception as e:
                    q.put(('log', f"Format validation error: {str(e)}"))
                    q.put(('log', "=" * 60))

        # Memory cleanup before GRPO (matching official Unsloth notebook)
        # This frees up GPU memory from pre-training dataset
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            q.put(('log', "Memory cleanup complete"))
        except Exception as e:
            q.put(('log', f"Memory cleanup skipped: {e}"))

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

        # Flatten key config values for easier access
        config_copy = config.copy()
        config_copy['model_name'] = model_name

        # Ensure system prompt is properly saved - check multiple sources
        system_prompt = (
            template_config.get('system_prompt') or
            config.get('system_prompt') or
            config.get('template', {}).get('system_prompt')
        )
        config_copy['system_prompt'] = system_prompt

        # Also preserve the full template config for complete chat template information
        if template_config:
            config_copy['template'] = template_config

        session_info = SessionInfo(
            session_id=session_id,
            model_name=model_name,
            status='completed',
            checkpoint_path=f"outputs/{session_id}/checkpoints/final",
            created_at=session_obj.created_at.isoformat() if session_obj.created_at else None,
            completed_at=datetime.now().isoformat(),
            best_reward=final_reward,
            epochs_trained=metrics.get('epochs_trained', config.get('num_epochs', 0)),
            training_config=config_copy,
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
