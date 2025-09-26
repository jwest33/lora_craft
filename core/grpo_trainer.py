"""GRPO (Group Relative Policy Optimization) trainer module."""
from unsloth import FastModel
import os
import torch
import gc
import json
import time
import platform
import warnings
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from accelerate import Accelerator
import safetensors.torch

from .dataset_handler import DatasetHandler, DatasetConfig
from .prompt_templates import PromptTemplate
from .custom_rewards import CustomRewardBuilder
from .system_config import SystemConfig, TrainingConfig
from .model_exporter import ModelExporter
from utils.logging_config import get_logger


logger = get_logger(__name__)


class CallbackLogHandler(logging.Handler):
    """Custom log handler that sends logs to a callback function."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        """Emit a log record to the callback."""
        if self.callback:
            log_entry = {
                'level': record.levelname,
                'message': self.format(record),
                'timestamp': time.time()
            }
            self.callback(log_entry)


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""
    # Model configuration
    model_name: str
    use_4bit: bool = False  # Disabled by default to avoid xformers issues
    use_8bit: bool = False
    load_in_4bit: bool = False  # Disabled by default to avoid xformers issues
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = None
    lora_bias: str = "none"

    # Training configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    warmup_ratio: float = 0.1
    logging_steps: int = 1  # Log every step for real-time metrics
    save_steps: int = 100
    eval_steps: int = 100
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    seed: int = 42

    # GRPO/GSPO specific
    loss_type: str = "grpo"  # Always use "grpo" for TRL compatibility
    importance_sampling_level: str = "token"  # Options: "token" (GRPO), "sequence" (GSPO)
    max_sequence_length: int = 2048
    max_new_tokens: int = 256  # Maximum tokens to generate (reduced for performance)
    num_generations_per_prompt: int = 2  # Reduced from 4 for faster training
    num_generations: int = 2  # Same as num_generations_per_prompt for TRL compatibility
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    kl_penalty: float = 0.01
    clip_range: float = 0.2
    value_coefficient: float = 1.0

    # GSPO specific parameters
    epsilon: float = 3e-4
    epsilon_high: float = 4e-4

    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"

    # Flags
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class GRPOModelTrainer:
    """GRPO trainer for fine-tuning language models."""

    def __init__(self,
                 config: GRPOTrainingConfig,
                 system_config: Optional[SystemConfig] = None):
        """Initialize GRPO trainer.

        Args:
            config: GRPO configuration
            system_config: System configuration for optimization
        """
        self.config = config
        self.system_config = system_config or SystemConfig()

        # Check platform for vLLM compatibility
        self.is_windows = platform.system() == 'Windows'
        if self.is_windows:
            logger.warning("Running on Windows - vLLM not supported, using transformers for generation")

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.accelerator = None
        self.optimizer = None
        self.lr_scheduler = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_reward = float('-inf')
        self.training_history = []

        # Callbacks
        self.progress_callback = None
        self.metrics_callback = None
        self.log_callback = None  # For sending logs to frontend

        # Setup directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.output_dir, self.config.cache_dir, self.config.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def set_log_callback(self, callback):
        """Set the log callback and configure logging to use it.

        Args:
            callback: Function to call with log messages
        """
        self.log_callback = callback

        # Add custom handler to logger
        if callback:
            handler = CallbackLogHandler(callback)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

            # Add handler to this module's logger
            logger.addHandler(handler)

            # Also add to dataset_handler logger
            dataset_logger = logging.getLogger('grpo_gui.core.dataset_handler')
            dataset_logger.addHandler(handler)

    def setup_model(self,
                   model_name: Optional[str] = None,
                   use_unsloth: bool = True) -> Tuple[Any, Any]:
        """Setup model and tokenizer.

        Args:
            model_name: Model name (overrides config)
            use_unsloth: Whether to use Unsloth for loading

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or self.config.model_name
        logger.info(f"Loading model: {model_name}")

        if use_unsloth:
            try:
                # Load with Unsloth
                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=self.config.max_sequence_length,
                    dtype=torch.float16 if self.config.fp16 else None,
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.use_8bit,
                    full_finetuning=False,
                )

                # Set padding side for generation
                self.tokenizer.padding_side = 'left'

                # Note: Chat template will be set later via template.setup_for_unsloth()
                logger.info("Tokenizer loaded, chat template will be configured during pre-training")

                # Get LoRA model with new API
                self.model = FastModel.get_peft_model(
                    self.model,
                    finetune_vision_layers=False,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias=self.config.lora_bias,
                    random_state=self.config.seed,
                )

                logger.info("Model loaded with Unsloth")

            except ImportError:
                logger.warning("Unsloth not available, falling back to standard loading")
                use_unsloth = False

        if not use_unsloth:
            # Standard loading with transformers
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.use_4bit,
                load_in_8bit=self.config.use_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config if self.config.use_4bit or self.config.use_8bit else None,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.config.cache_dir,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir,
                padding_side='left',  # Required for decoder-only models during generation
            )

            # Setup tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Prepare model for training
            self.model = prepare_model_for_kbit_training(self.model)

            # Setup LoRA
            self.setup_lora()

        return self.model, self.tokenizer

    def setup_lora(self):
        """Setup LoRA configuration."""
        if not self.model:
            raise ValueError("Model must be loaded before setting up LoRA")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules or ["q_proj", "v_proj"],
            bias=self.config.lora_bias,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def pre_fine_tune(self,
                     dataset: Dataset,
                     template: PromptTemplate,
                     epochs: int = 1) -> Dict[str, Any]:
        """Pre-fine-tuning phase for format learning.

        Args:
            dataset: Training dataset
            template: Prompt template
            epochs: Number of pre-training epochs

        Returns:
            Training metrics
        """
        logger.info("Starting pre-fine-tuning phase")

        # Ensure chat template is set on tokenizer
        if hasattr(template, 'setup_for_unsloth') and self.tokenizer:
            template.setup_for_unsloth(self.tokenizer)
            logger.info(f"Applied chat template: {template.config.model_type}")

        # Apply template to dataset (just format, don't tokenize yet)
        def apply_template(example):
            formatted = template.apply(example, mode='training')
            return {'text': formatted}

        formatted_dataset = dataset.map(
            apply_template,
            remove_columns=dataset.column_names  # Remove original columns, keep only text
        )

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # DataCollator will handle padding
                max_length=self.config.max_sequence_length,
                return_special_tokens_mask=False
            )

        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']  # Remove text, keep only tokenized data
        )

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Training arguments for pre-fine-tuning
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/pre_finetune",
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            learning_rate=self.config.learning_rate * 0.5,  # Lower LR for pre-training
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            seed=self.config.seed,
        )

        # Create trainer
        from transformers import Trainer

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        # Train
        train_result = trainer.train()

        # Save pre-trained model
        pre_trained_path = Path(self.config.checkpoint_dir) / "pre_trained"
        trainer.save_model(pre_trained_path)

        logger.info(f"Pre-fine-tuning completed. Model saved to {pre_trained_path}")

        return train_result.metrics

    def _format_dataset_for_trl(self, dataset: Dataset, template: PromptTemplate) -> Dataset:
        """Format dataset for TRL's expected format.

        Args:
            dataset: Original dataset
            template: Prompt template

        Returns:
            Formatted dataset
        """
        formatted_items = []

        for item in dataset:
            # Format the prompt using the template
            prompt = template.apply(item, mode='inference')

            # TRL expects a specific format with prompt field
            formatted_item = {
                "prompt": [
                    {"role": "user", "content": item.get('instruction', '')}
                ],
            }

            # Add any additional fields that might be needed for rewards
            if 'output' in item:
                formatted_item['answer'] = item['output']
            if 'input' in item:
                formatted_item['context'] = item['input']

            formatted_items.append(formatted_item)

        return Dataset.from_list(formatted_items)

    def _create_trl_reward_funcs(self, reward_builder: CustomRewardBuilder) -> List[Callable]:
        """Convert our reward builder to TRL's expected format.

        Args:
            reward_builder: Our custom reward builder

        Returns:
            List of reward functions for TRL
        """
        def reward_wrapper(prompts, completions, **kwargs):
            """Wrapper to adapt our reward function to TRL's format."""
            scores = []

            for prompt, completion in zip(prompts, completions):
                # Extract text from completion format
                if isinstance(completion, list) and len(completion) > 0:
                    response_text = completion[0].get('content', '')
                else:
                    response_text = str(completion)

                # Compute reward using our custom builder
                # Create a sample dict for compatibility
                sample = {
                    'instruction': prompt[0]['content'] if isinstance(prompt, list) else prompt,
                    'generated': response_text
                }

                reward, reward_components = reward_builder.compute_total_reward(
                    instruction=sample['instruction'],
                    generated=response_text,
                    reference=None
                )

                # Use the total reward directly
                scores.append(reward)

            return scores

        return [reward_wrapper]

    def grpo_train(self,
                  dataset: Dataset,
                  template: PromptTemplate,
                  reward_builder: CustomRewardBuilder,
                  validation_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Execute GRPO training using TRL's GRPOTrainer.

        Args:
            dataset: Training dataset
            template: Prompt template
            reward_builder: Reward function builder
            validation_dataset: Optional validation dataset

        Returns:
            Training metrics
        """

        logger.info("Starting GRPO training with TRL...")

        # Disable Torch compilation for TRL to avoid Dynamo errors with Unsloth
        # This is necessary due to incompatibility between Unsloth's optimizations and TRL's gradient computation
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        os.environ['TORCH_COMPILE_DISABLE'] = '1'

        # Also disable Unsloth's auto-compilation for TRL modules
        os.environ['UNSLOTH_DISABLE_COMPILE'] = '1'

        logger.info("Disabled Torch compilation for GRPO training to ensure compatibility")

        # Clear Unsloth compilation cache if it exists to avoid conflicts
        cache_dir = Path("./unsloth_compiled_cache")
        if cache_dir.exists():
            try:
                import shutil
                # Rename instead of delete to preserve for debugging
                backup_dir = Path(f"./unsloth_compiled_cache_backup_{int(time.time())}")
                shutil.move(str(cache_dir), str(backup_dir))
                logger.info(f"Moved compilation cache to {backup_dir}")
            except Exception as e:
                logger.warning(f"Could not move compilation cache: {e}")

        # Ensure chat template is applied to tokenizer if not already done
        if hasattr(template, 'setup_for_unsloth') and self.tokenizer and not hasattr(self.tokenizer, '_template_applied'):
            template.setup_for_unsloth(self.tokenizer)
            self.tokenizer._template_applied = True  # Mark as applied
            logger.info(f"Applied chat template for GRPO training: {template.config.model_type}")

        # Format dataset for TRL's GRPOTrainer
        formatted_dataset = self._format_dataset_for_trl(dataset, template)

        # Calculate batch sizes for GRPO
        # The constraints are:
        # 1. generation_batch_size must be divisible by global_batch_size
        # 2. global_batch_size must be divisible by num_generations
        num_gens = self.config.num_generations
        original_batch_size = self.config.per_device_train_batch_size
        grad_accum = self.config.gradient_accumulation_steps

        # Find a batch_size that works with num_generations
        # We need batch_size to be compatible with num_generations
        if original_batch_size % num_gens != 0:
            # Find the closest valid batch_size
            if num_gens <= original_batch_size:
                # Round down to nearest multiple of num_generations
                batch_size = (original_batch_size // num_gens) * num_gens
                if batch_size == 0:
                    batch_size = num_gens
            else:
                # num_generations is larger than batch_size, use num_generations
                batch_size = num_gens
            logger.warning(f"Adjusted batch_size from {original_batch_size} to {batch_size} for compatibility with num_generations={num_gens}")
        else:
            batch_size = original_batch_size

        global_batch_size = batch_size * grad_accum

        # generation_batch_size must be a multiple of global_batch_size
        # Use the same as global_batch_size for simplicity
        generation_batch_size = global_batch_size

        logger.info(f"GRPO batch configuration: batch_size={batch_size}, global_batch_size={global_batch_size}, generation_batch_size={generation_batch_size}, num_generations={num_gens}")

        # Calculate the actual number of training steps based on dataset and epochs
        num_train_samples = len(formatted_dataset)
        steps_per_epoch = max(1, num_train_samples // global_batch_size)  # At least 1 step per epoch
        max_steps = steps_per_epoch * self.config.num_train_epochs

        logger.info(f"Training steps calculation: {num_train_samples} samples / {global_batch_size} batch size = {steps_per_epoch} steps/epoch")
        logger.info(f"Total training steps: {steps_per_epoch} steps/epoch * {self.config.num_train_epochs} epochs = {max_steps} steps")

        # Create TRL GRPO configuration with algorithm support
        grpo_config = TRLGRPOConfig(
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=batch_size,  # Use adjusted batch_size
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            generation_batch_size=generation_batch_size,  # Must be divisible by num_generations
            max_steps=max_steps,  # Calculated based on dataset size and epochs
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            num_generations=self.config.num_generations,  # Number of generations per prompt
            max_prompt_length=128,
            max_completion_length=self.config.max_new_tokens,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
            report_to="none",
            output_dir=self.config.output_dir,
            # GRPO/GSPO algorithm selection
            loss_type=self.config.loss_type,
            importance_sampling_level=self.config.importance_sampling_level,
            # GSPO specific parameters
            epsilon=self.config.epsilon,
            epsilon_high=self.config.epsilon_high,
            # Generation parameters
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )

        # Create reward functions list for TRL
        reward_funcs = self._create_trl_reward_funcs(reward_builder)

        # Initialize TRL's GRPOTrainer with valid parameters only
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=formatted_dataset,
        )

        # Custom logging callback
        original_log = trainer.log
        parent = self  # Reference to GRPOModelTrainer for callbacks

        def custom_log(logs, start_time=None):
            if logs and isinstance(logs, dict):
                # Only process logs that contain actual training metrics (not intermediate logs)
                has_reward_data = 'reward' in logs or 'rewards/reward_wrapper/mean' in logs

                if has_reward_data:
                    # Extract actual values from the logs
                    actual_loss = float(logs.get('loss', 0.0))
                    actual_reward = float(logs.get('reward', logs.get('rewards/reward_wrapper/mean', 0.0)))
                    grad_norm = float(logs.get('grad_norm', 0.0))

                    # Convert to our format
                    metrics = {
                        'epoch': logs.get('epoch', parent.current_epoch),
                        'step': trainer.state.global_step if hasattr(trainer, 'state') else 0,
                        'loss': actual_loss,
                        'mean_reward': actual_reward,
                        'learning_rate': logs.get('learning_rate', parent.config.learning_rate),
                        'grad_norm': grad_norm,
                        'reward_std': logs.get('reward_std', logs.get('rewards/reward_wrapper/std', 0.0)),
                    }

                    # Add all reward components
                    for key, value in logs.items():
                        if key.startswith('rewards/') or key.startswith('completions/'):
                            metrics[key] = value

                    # Update tracking
                    if metrics['mean_reward'] > parent.best_reward:
                        parent.best_reward = metrics['mean_reward']
                        parent.save_checkpoint(f"best_step_{metrics['step']}")

                    parent.training_history.append(metrics)

                    # Callback to frontend
                    if parent.metrics_callback:
                        parent.metrics_callback(metrics)

                    # Log to console with actual values - use higher precision for small values
                    if abs(actual_loss) < 0.01:
                        loss_str = f"{actual_loss:.6f}"
                    else:
                        loss_str = f"{actual_loss:.4f}"

                    logger.info(f"Step {metrics['step']}: Loss={loss_str}, Reward={actual_reward:.6f}, Grad Norm={grad_norm:.4f}")

                    # Debug: Also print the raw dictionary to verify values
                    print(logs)

            # Call original log
            if start_time is not None:
                original_log(logs, start_time)
            else:
                original_log(logs)

        trainer.log = custom_log

        # Update epoch tracking
        parent.current_epoch = 0

        # Train with TRL
        logger.info("Starting TRL GRPO training...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*right-padding.*")
                warnings.filterwarnings("ignore", message=".*decoder-only.*")
                warnings.filterwarnings("ignore", message=".*Dynamo.*")
                warnings.filterwarnings("ignore", message=".*compilation.*")
                trainer.train()

            # Save final checkpoint immediately after training completes
            logger.info("Training complete, saving final checkpoint...")
            self.save_checkpoint("final")

        except Exception as e:
            # Check if it's a compilation-related error
            if "Dynamo" in str(e) or "compile" in str(e).lower():
                logger.error(f"Compilation error during training: {e}")
                logger.info("Attempting to continue with eager mode...")

                # Force eager mode and retry
                import torch
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.cache_size_limit = 1

                # Retry training with suppressed errors
                try:
                    trainer.train()
                    logger.info("Training completed in eager mode")
                    self.save_checkpoint("final")
                except Exception as retry_error:
                    logger.error(f"Training failed even in eager mode: {retry_error}")
                    raise retry_error
            else:
                # Not a compilation error, re-raise
                logger.error(f"Training error: {e}")
                raise e

        # Set model back to eval mode
        self.model.eval()

        return self._compile_training_metrics()

    def _compile_training_metrics(self) -> Dict[str, Any]:
        """Compile training metrics."""
        return {
            'final_reward': self.best_reward,
            'training_history': self.training_history,
            'global_steps': self.global_step,
            'epochs_trained': self.current_epoch + 1,
        }

    def save_checkpoint(self, name: str):
        """Save training checkpoint.

        Args:
            name: Checkpoint name
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_reward': self.best_reward,
            'training_history': self.training_history,
            'config': self.config.to_dict(),
        }

        with open(checkpoint_path / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        path = Path(checkpoint_path)

        # Load model
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Load training state
        with open(path / 'training_state.json', 'r') as f:
            state = json.load(f)

        self.global_step = state['global_step']
        self.current_epoch = state['current_epoch']
        self.best_reward = state['best_reward']
        self.training_history = state['training_history']

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def export_model(
        self,
        export_format: str = "huggingface",
        export_name: Optional[str] = None,
        quantization: Optional[str] = None,
        merge_lora: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Export the trained model to specified format.

        Args:
            export_format: Format to export to (safetensors, huggingface, gguf, merged)
            export_name: Optional custom name for export
            quantization: Quantization level for GGUF format
            merge_lora: Whether to merge LoRA weights with base model
            progress_callback: Callback for progress updates

        Returns:
            Tuple of (success, export_path, metadata)
        """
        if not hasattr(self, 'model') or self.model is None:
            return False, "", {"error": "No model loaded"}

        # Save current checkpoint if not already saved
        checkpoint_path = Path(self.config.checkpoint_dir) / "final"
        if not checkpoint_path.exists():
            self.save_checkpoint("final")

        # Use ModelExporter to handle the export
        exporter = ModelExporter(export_dir="./exports")

        # Generate session ID from checkpoint path
        # checkpoint_dir is in format ./outputs/{session_id}/checkpoints
        session_id = Path(self.config.checkpoint_dir).parent.name

        return exporter.export_model(
            model_path=str(checkpoint_path),
            session_id=session_id,
            export_format=export_format,
            export_name=export_name,
            quantization=quantization,
            merge_lora=merge_lora,
            progress_callback=progress_callback
        )

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.

        Returns:
            List of checkpoint information
        """
        checkpoints = []
        checkpoint_dir = Path(self.config.checkpoint_dir)

        if checkpoint_dir.exists():
            for cp_dir in checkpoint_dir.iterdir():
                if cp_dir.is_dir() and (cp_dir / "training_state.json").exists():
                    with open(cp_dir / "training_state.json", 'r') as f:
                        state = json.load(f)
                    checkpoints.append({
                        "name": cp_dir.name,
                        "path": str(cp_dir),
                        "global_step": state.get("global_step"),
                        "epoch": state.get("current_epoch"),
                        "best_reward": state.get("best_reward")
                    })

        return sorted(checkpoints, key=lambda x: x.get("global_step", 0), reverse=True)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleaned up model resources")
