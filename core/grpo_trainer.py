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
    """Configuration for GRPO training.

    GRPO Training Process:
        GRPO (Group Relative Policy Optimization) training consists of two phases:

        1. Pre-training Phase (Optional, but recommended):
           - Purpose: Teach the model the specific format requirements (reasoning markers,
             solution markers) before reinforcement learning
           - Duration: Typically 1-2 epochs on a small subset of data
           - Method: Standard supervised fine-tuning (SFT) with complete prompt+response examples
           - In the official GRPO notebook: Uses ~59 filtered short samples for format learning

        2. GRPO Training Phase (Main training):
           - Purpose: Reinforcement learning with reward-based optimization
           - Duration: Multiple epochs on the full dataset
           - Method: Model generates responses and receives rewards based on quality
           - In the official GRPO notebook: Uses full dataset (e.g., 17K samples)

    Key Configuration Notes:
        - pre_training_max_samples: Separate from main training max_samples. This allows
          pre-training to use a small filtered subset while main training uses the full dataset.
        - max_samples (in dataset config): Represents samples PER EPOCH for main training
        - num_train_epochs: Number of epochs for main GRPO training (not pre-training)
        - pre_training_epochs: Separate epoch count just for format learning phase
    """
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
    logging_steps: int = 1  # Log every step for real-time frontend metrics
    save_steps: int = 100
    eval_steps: int = 100
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    seed: int = 42

    # Pre-training configuration
    pre_training_epochs: int = 2  # Default: 2 epochs (matches official GRPO notebook)
    pre_training_max_samples: Optional[int] = None  # Separate sample limit for pre-training phase
    pre_training_filter_by_length: bool = False  # Filter pre-training samples by length
    pre_training_max_length_ratio: float = 0.5  # Max length as ratio of max_sequence_length

    # GRPO/GSPO specific
    loss_type: str = "grpo"  # Always use "grpo" for TRL compatibility
    importance_sampling_level: str = "token"  # Options: "token" (GRPO), "sequence" (GSPO)
    max_sequence_length: int = 2048
    max_new_tokens: int = 512  # Maximum tokens to generate (reduced for performance)
    num_generations_per_prompt: int = 2  # Reduced from 4 for faster training
    num_generations: int = 2  # Same as num_generations_per_prompt for TRL compatibility
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    kl_penalty: float = 0.05  # Increased to prevent KL divergence
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
        self.training_phase = 'idle'  # 'idle', 'pre-training', 'training'
        self.grpo_start_step = 0  # Track step when GRPO training starts (after pre-training)

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
                # Determine dtype for model loading
                if self.config.fp16:
                    model_dtype = torch.float16
                elif self.config.bf16:
                    model_dtype = torch.bfloat16
                else:
                    model_dtype = None

                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=self.config.max_sequence_length,
                    dtype=model_dtype,
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
                     epochs: Optional[int] = None) -> Dict[str, Any]:
        """Pre-fine-tuning phase for format learning.

        This phase teaches the model to follow the specific formatting requirements
        (reasoning markers, solution markers) before GRPO reinforcement learning.

        In the official GRPO notebook, pre-training uses a small filtered subset (~59 samples)
        of short examples to quickly learn the format, while main GRPO training uses the full dataset.

        Args:
            dataset: Training dataset (full dataset - will be filtered if configured)
            template: Prompt template
            epochs: Number of pre-training epochs (defaults to config.pre_training_epochs)

        Returns:
            Training metrics
        """
        logger.info("Starting pre-fine-tuning phase")
        self.training_phase = 'pre-training'

        # Use epochs from config if not specified
        if epochs is None:
            epochs = self.config.pre_training_epochs

        # Apply pre-training specific filtering/sampling
        pre_train_dataset = dataset
        original_size = len(dataset)

        # Filter by length if configured (matches official notebook approach)
        if self.config.pre_training_filter_by_length:
            max_length = int(self.config.max_sequence_length * self.config.pre_training_max_length_ratio)
            logger.info(f"Filtering pre-training dataset by length (max: {max_length} tokens)")

            # Tokenize to get lengths
            def get_length(example):
                if 'instruction' in example:
                    text = str(example.get('instruction', '')) + str(example.get('response', example.get('output', '')))
                    tokens = self.tokenizer(text, truncation=False)
                    return {'length': len(tokens['input_ids'])}
                return {'length': 0}

            dataset_with_lengths = pre_train_dataset.map(get_length)
            pre_train_dataset = dataset_with_lengths.filter(lambda x: x['length'] <= max_length)
            logger.info(f"Filtered dataset from {original_size} to {len(pre_train_dataset)} samples by length")

        # Apply sample limit if configured (separate from main training max_samples)
        if self.config.pre_training_max_samples:
            if len(pre_train_dataset) > self.config.pre_training_max_samples:
                pre_train_dataset = pre_train_dataset.select(range(self.config.pre_training_max_samples))
                logger.info(f"Limited pre-training dataset from {len(dataset)} to {self.config.pre_training_max_samples} samples")

        logger.info(f"Pre-training with {len(pre_train_dataset)} samples for {epochs} epochs (original dataset: {original_size} samples)")

        # Store original tokenizer configuration
        original_padding_side = self.tokenizer.padding_side if hasattr(self.tokenizer, 'padding_side') else None

        # Ensure chat template is set on tokenizer
        if hasattr(template, 'setup_for_unsloth') and self.tokenizer:
            template.setup_for_unsloth(self.tokenizer)
            logger.info(f"Applied chat template: {template.config.model_type}")

        # Set padding side to 'right' for training (required by DataCollatorForLanguageModeling)
        self.tokenizer.padding_side = 'right'

        # Apply template to dataset for pre-training
        # CRITICAL: Use the same prompt format as GRPO will use, but include the response
        def apply_template(example):
            # Format the prompt exactly as GRPO will see it
            if 'instruction' in example:
                # Get the prompt format (what GRPO will use)
                prompt_messages = [
                    {'role': 'user', 'content': example.get('instruction', '')}
                ]
                # Use inference mode to get the prompt format without response
                prompt = template.apply(prompt_messages, mode='inference')

                # Get the response with reasoning markers
                response = example.get('response', example.get('output', ''))

                # For pre-training, we want the model to learn the format
                # So we combine prompt + response with proper reasoning and solution markers
                if template.config.reasoning_start_marker and template.config.reasoning_end_marker:
                    # Check if response already has the markers
                    has_reasoning_markers = (template.config.reasoning_start_marker in response and
                                           template.config.reasoning_end_marker in response)
                    has_solution_markers = (template.config.solution_start_marker in response and
                                          template.config.solution_end_marker in response)

                    # If no markers at all, add the configured template markers
                    if not has_reasoning_markers and not has_solution_markers:
                        # Use the template's configured markers (not hardcoded defaults)
                        reasoning_start = template.config.reasoning_start_marker
                        reasoning_end = template.config.reasoning_end_marker
                        solution_start = template.config.solution_start_marker
                        solution_end = template.config.solution_end_marker

                        # Wrap response with configured markers
                        response = (f"{reasoning_start}\n"
                                  f"Let me work through this step by step.\n"
                                  f"{reasoning_end}\n"
                                  f"{solution_start}\n"
                                  f"{response}\n"
                                  f"{solution_end}")

                # Combine prompt and response for training
                formatted = prompt + response

                # Add EOS token if needed
                if hasattr(template, 'config') and template.config.model_type == 'grpo':
                    if not formatted.endswith('</s>'):
                        formatted += '</s>'
            else:
                # Fallback for non-standard format
                formatted = template.apply(example, mode='training')

            # Log sample format for debugging (only first example)
            if not hasattr(apply_template, 'logged'):
                logger.info(f"Pre-training format sample (first 1000 chars): {formatted[:1000]}")
                apply_template.logged = True

            return {'text': formatted}

        formatted_dataset = pre_train_dataset.map(
            apply_template,
            remove_columns=pre_train_dataset.column_names  # Remove original columns, keep only text
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

        # Setup data collator with matching dtype
        # Detect the model's actual dtype (critical for Unsloth models which may not match config flags)
        model_dtype = None

        # Try to get dtype from model directly
        if hasattr(self.model, 'dtype'):
            model_dtype = self.model.dtype
            logger.info(f"Detected model dtype from model.dtype: {model_dtype}")
        else:
            # Fall back to checking a parameter
            try:
                # Get dtype from first parameter
                first_param = next(self.model.parameters())
                model_dtype = first_param.dtype
                logger.info(f"Detected model dtype from parameters: {model_dtype}")
            except StopIteration:
                # No parameters found, fall back to config flags
                if self.config.fp16:
                    model_dtype = torch.float16
                elif self.config.bf16:
                    model_dtype = torch.bfloat16
                logger.info(f"Using dtype from config flags: {model_dtype}")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Override the collator's return_tensors behavior to use correct dtype
        original_call = data_collator.__call__

        def dtype_aware_call(features):
            batch = original_call(features)
            # Convert tensors to model's dtype if specified
            if model_dtype is not None:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor) and batch[key].dtype in (torch.float32, torch.float64):
                        batch[key] = batch[key].to(dtype=model_dtype)
                        logger.debug(f"Converted {key} from {batch[key].dtype} to {model_dtype}")
            return batch

        data_collator.__call__ = dtype_aware_call
        logger.info(f"Data collator configured with dtype conversion to {model_dtype}")

        # Determine precision flags for TrainingArguments based on actual model dtype
        # This is critical because the model may be loaded in a different dtype than config suggests
        use_fp16 = self.config.fp16
        use_bf16 = self.config.bf16

        if model_dtype == torch.bfloat16:
            use_bf16 = True
            use_fp16 = False
            logger.info("Setting bf16=True in TrainingArguments to match model dtype")
        elif model_dtype == torch.float16:
            use_fp16 = True
            use_bf16 = False
            logger.info("Setting fp16=True in TrainingArguments to match model dtype")

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
            fp16=use_fp16,
            bf16=use_bf16,
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

        # Clear the environment variable to avoid issues with generation later
        if 'UNSLOTH_RETURN_LOGITS' in os.environ:
            del os.environ['UNSLOTH_RETURN_LOGITS']
            logger.info("Cleared UNSLOTH_RETURN_LOGITS environment variable")

        # Save pre-trained model
        pre_trained_path = Path(self.config.checkpoint_dir) / "pre_trained"
        trainer.save_model(pre_trained_path)

        # Restore original tokenizer configuration for generation
        if original_padding_side is not None:
            self.tokenizer.padding_side = original_padding_side
            logger.info(f"Restored tokenizer padding_side to '{original_padding_side}' for generation")
        else:
            # Default to left for generation if not set
            self.tokenizer.padding_side = 'left'
            logger.info("Set tokenizer padding_side to 'left' for generation")

        # Ensure model is in eval mode after training
        self.model.eval()

        # Clear any cached compiled functions from Unsloth
        if hasattr(self.model, '_clear_cache'):
            self.model._clear_cache()
            logger.info("Cleared model compilation cache")

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
        vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 151936
        skipped_count = 0
        first_logged = False

        for item in dataset:
            # Format the prompt using the template
            prompt = template.apply(item, mode='inference')

            # Log first prompt for comparison with pre-training
            if not first_logged:
                logger.info(f"GRPO prompt format sample (first 1000 chars): {prompt[:1000]}")
                first_logged = True

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

            # Tokenize to check for invalid tokens (defensive check)
            if self.tokenizer:
                try:
                    test_tokens = self.tokenizer.encode(
                        formatted_item["prompt"][0]["content"],
                        add_special_tokens=True
                    )
                    # Check if any token exceeds vocab size
                    if any(token >= vocab_size for token in test_tokens):
                        logger.warning(f"Skipping sample with out-of-range tokens (max={max(test_tokens)})")
                        skipped_count += 1
                        continue
                except Exception as e:
                    logger.warning(f"Error tokenizing sample: {e}")
                    skipped_count += 1
                    continue

            formatted_items.append(formatted_item)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} samples due to tokenization issues")

        return Dataset.from_list(formatted_items)

    def _create_trl_reward_funcs(self, reward_builder: CustomRewardBuilder) -> List[Callable]:
        """Convert our reward builder to TRL's expected format.

        Args:
            reward_builder: Our custom reward builder

        Returns:
            List of reward functions for TRL
        """
        # Capture model dtype and device for tensor conversion
        model_dtype = self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
        model_device = self.model.device if hasattr(self.model, 'device') else 'cpu'

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

            # Convert scores to tensor with model's dtype to avoid dtype mismatch
            return torch.tensor(scores, dtype=model_dtype, device=model_device)

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
        self.training_phase = 'training'  # Set training phase
        # Capture the current step as the GRPO start point (will be > 0 if pre-training occurred)
        self.grpo_start_step = self.global_step
        logger.info(f"GRPO training starting at step {self.grpo_start_step} (epoch counter will reset to 0)")

        # Disable Torch compilation for TRL to avoid Dynamo errors with Unsloth
        # This is necessary due to incompatibility between Unsloth's optimizations and TRL's gradient computation
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        os.environ['TORCH_COMPILE_DISABLE'] = '1'

        # Also disable Unsloth's auto-compilation for TRL modules
        os.environ['UNSLOTH_DISABLE_COMPILE'] = '1'

        logger.info("Disabled Torch compilation for GRPO training to ensure compatibility")

        # Clear Unsloth compilation cache if it exists to avoid conflicts
        cache_dir = Path("./unsloth_compiled_cache")
        #if cache_dir.exists():
        #    try:
        #        import shutil
                # Rename instead of delete to preserve for debugging
        #        backup_dir = Path(f"./unsloth_compiled_cache_backup_{int(time.time())}")
                # shutil.move(str(cache_dir), str(backup_dir))
                #logger.info(f"Moved compilation cache to {backup_dir}")
        #    except Exception as e:
        #        logger.warning(f"Could not move compilation cache: {e}")

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
        # The dataset handler has already applied max_samples limiting if configured
        # So we just use the actual dataset size as samples per epoch
        samples_per_epoch = len(formatted_dataset)
        logger.info(f"Using {samples_per_epoch} samples per epoch")

        # IMPORTANT: TRL's GRPO internally divides batch_size by num_generations
        # to determine the number of unique prompts processed per step.
        # Each unique prompt generates num_generations completions.
        effective_prompts_per_step = global_batch_size // num_gens
        steps_per_epoch = max(1, samples_per_epoch // effective_prompts_per_step)
        max_steps = steps_per_epoch * self.config.num_train_epochs

        logger.info(f"Training steps calculation: {samples_per_epoch} samples/epoch / {effective_prompts_per_step} prompts per step (batch_size {global_batch_size} ÷ {num_gens} generations) = {steps_per_epoch} steps/epoch")
        logger.info(f"Total training steps: {steps_per_epoch} steps/epoch * {self.config.num_train_epochs} epochs = {max_steps} steps")
        logger.info(f"Note: Each step processes {effective_prompts_per_step} unique prompts, generating {num_gens} completions per prompt ({effective_prompts_per_step * num_gens} total completions per step)")

        # Create TRL GRPO configuration with algorithm support
        logger.info(f"Configuring GRPO training with logging_steps={self.config.logging_steps} for frequent updates")
        grpo_config = TRLGRPOConfig(
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=batch_size,  # Use adjusted batch_size
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            generation_batch_size=generation_batch_size,  # Must be divisible by num_generations
            num_train_epochs=self.config.num_train_epochs,  # Use epochs instead of max_steps
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
            # KL divergence regularization (critical for stable training)
            beta=self.config.kl_penalty,  # Controls KL divergence weight
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

        # Add debugging info for vocab sizes
        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        if hasattr(self.model, 'config'):
            logger.info(f"Model config vocab size: {getattr(self.model.config, 'vocab_size', 'Not found')}")
        if hasattr(self.model, 'get_input_embeddings'):
            embeddings = self.model.get_input_embeddings()
            if embeddings is not None:
                logger.info(f"Model embedding size: {embeddings.weight.shape}")

        # Check and fix vocab size mismatch
        if hasattr(self.model, 'config') and hasattr(self.tokenizer, 'vocab_size'):
            model_vocab_size = getattr(self.model.config, 'vocab_size', None)
            tokenizer_vocab_size = self.tokenizer.vocab_size

            if model_vocab_size and model_vocab_size != tokenizer_vocab_size:
                logger.warning(f"Vocab size mismatch: Model={model_vocab_size}, Tokenizer={tokenizer_vocab_size}")

                # NEVER resize DOWN as it causes index out of bounds errors
                if tokenizer_vocab_size > model_vocab_size:
                    # Only resize UP when tokenizer has MORE tokens
                    if hasattr(self.model, 'resize_token_embeddings'):
                        logger.info(f"Resizing model embeddings UP from {model_vocab_size} to {tokenizer_vocab_size}")
                        self.model.resize_token_embeddings(tokenizer_vocab_size)
                        logger.info("Model embeddings resized successfully")
                    else:
                        logger.error("Cannot resize token embeddings - model may not work correctly!")
                else:
                    # Model has more tokens than tokenizer - keep model size
                    logger.info(f"Keeping model vocab size at {model_vocab_size} (larger than tokenizer's {tokenizer_vocab_size})")
                    # Update tokenizer to match model's vocab size to prevent issues
                    if hasattr(self.tokenizer, 'vocab_size'):
                        logger.info(f"Updating tokenizer reference vocab_size to match model: {model_vocab_size}")
                        # Don't actually resize tokenizer, just update the reference
                        # The model will handle any tokens beyond tokenizer's actual vocab

        # Ensure generation config is properly set
        if hasattr(self.model, 'generation_config'):
            logger.info(f"Generation config pad_token_id: {self.model.generation_config.pad_token_id}")
            logger.info(f"Generation config eos_token_id: {self.model.generation_config.eos_token_id}")

            # Update generation config to match tokenizer
            if self.tokenizer.pad_token_id is not None:
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            if self.tokenizer.eos_token_id is not None:
                self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

            # Set max_length to avoid dimension issues
            self.model.generation_config.max_length = self.config.max_sequence_length
            self.model.generation_config.max_new_tokens = self.config.max_new_tokens
            logger.info(f"Updated generation config max_length: {self.model.generation_config.max_length}")

        # Reset model state to avoid dimension mismatches
        # This is critical when reusing a model from a previous training session
        try:
            import torch as torch_module  # Re-import to ensure it's in scope

            # Clear any cached states in the model
            if hasattr(self.model, 'eval'):
                self.model.eval()
            if hasattr(self.model, 'train'):
                self.model.train()

            # Reset gradient checkpointing if available
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            # Clear any cached attention states
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()

            # Ensure model is on the correct device
            if torch_module.cuda.is_available():
                self.model = self.model.cuda()
                torch_module.cuda.empty_cache()

            logger.info("Reset model state for GRPO training")
        except Exception as e:
            logger.warning(f"Could not fully reset model state: {e}")

        # Validate token IDs in dataset before training
        max_token_id = 0
        invalid_samples = []
        vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 151936

        logger.info(f"Validating token IDs in dataset (vocab_size={vocab_size})...")

        for idx, sample in enumerate(formatted_dataset):
            if 'input_ids' in sample:
                token_ids = sample['input_ids']
                if isinstance(token_ids, list):
                    max_id = max(token_ids) if token_ids else 0
                    if max_id >= vocab_size:
                        invalid_samples.append((idx, max_id))
                    max_token_id = max(max_token_id, max_id)

        if invalid_samples:
            logger.error(f"Found {len(invalid_samples)} samples with token IDs >= {vocab_size}")
            for idx, max_id in invalid_samples[:5]:  # Show first 5
                logger.error(f"  Sample {idx}: max token ID = {max_id}")

            # Filter out invalid samples
            logger.info("Filtering out invalid samples...")
            valid_indices = [i for i in range(len(formatted_dataset))
                           if not any(idx == i for idx, _ in invalid_samples)]
            formatted_dataset = formatted_dataset.select(valid_indices)
            logger.info(f"Filtered dataset size: {len(formatted_dataset)} samples")
        else:
            logger.info(f"All token IDs are valid (max={max_token_id} < {vocab_size})")

        # DO NOT override model's vocab_size with tokenizer's if model is larger
        # This causes index out of bounds errors
        if hasattr(self.model, 'config'):
            logger.info(f"Final model config vocab_size: {self.model.config.vocab_size}")

        # Initialize TRL's GRPOTrainer with valid parameters only
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=formatted_dataset,
        )

        # Re-apply generation config after GRPOTrainer init (TRL may reset it)
        if hasattr(trainer.model, 'generation_config'):
            logger.info("Re-applying generation config after GRPOTrainer initialization")
            trainer.model.generation_config.top_k = self.config.top_k
            trainer.model.generation_config.top_p = self.config.top_p
            trainer.model.generation_config.temperature = self.config.temperature
            trainer.model.generation_config.max_length = self.config.max_sequence_length
            trainer.model.generation_config.max_new_tokens = self.config.max_new_tokens
            trainer.model.generation_config.repetition_penalty = self.config.repetition_penalty
            logger.info(f"Generation config set: top_k={self.config.top_k}, top_p={self.config.top_p}, "
                       f"temperature={self.config.temperature}, max_length={self.config.max_sequence_length}")

        # Custom logging callback
        original_log = trainer.log
        parent = self  # Reference to GRPOModelTrainer for callbacks
        last_step_reported = 0  # Track last reported step

        def custom_log(logs, start_time=None):
            nonlocal last_step_reported

            if logs and isinstance(logs, dict):
                # Check if we have step information
                current_step = None
                if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step'):
                    current_step = trainer.state.global_step

                # Debug: Log available keys when step changes to understand TRL's log structure
                if current_step and current_step > last_step_reported:
                    last_step_reported = current_step
                    logger.debug(f"Step {current_step} - Available log keys: {list(logs.keys())}")

                # Send metrics only when we have complete data from TRL
                # TRL calls log() multiple times per step - first with basic metrics, then with all metrics
                # We only send when we detect the complete metrics (presence of 'kl' or 'epoch' indicates complete data)
                has_complete_metrics = 'kl' in logs or 'epoch' in logs or 'completions/mean_length' in logs

                if current_step is not None and has_complete_metrics and (parent.config.logging_steps == 1 or current_step % parent.config.logging_steps == 0):
                    logger.debug(f"Metrics logging triggered at step {current_step} with complete metrics (logging_steps={parent.config.logging_steps})")
                    # Extract actual values from the logs with comprehensive fallbacks
                    # Try multiple possible key names for loss
                    actual_loss = float(logs.get('loss', logs.get('train_loss', logs.get('train/loss', 0.0))))

                    # Try multiple possible key names for reward (TRL uses different keys)
                    actual_reward = float(
                        logs.get('rewards/reward_wrapper/mean',  # TRL's primary key
                        logs.get('reward',                        # TRL's secondary key
                        logs.get('rewards/mean',                  # Alternative
                        logs.get('mean_reward', 0.0))))           # Our own key
                    )

                    # Extract reward std with fallbacks
                    actual_reward_std = float(
                        logs.get('rewards/reward_wrapper/std',   # TRL's primary key
                        logs.get('reward_std',                    # TRL's secondary key
                        logs.get('rewards/std', 0.0)))            # Alternative
                    )

                    grad_norm = float(logs.get('grad_norm', 0.0))

                    # Debug: Log the actual values we extracted
                    logger.debug(f"Step {current_step} - Extracted metrics: loss={actual_loss}, reward={actual_reward}, reward_std={actual_reward_std}, grad_norm={grad_norm}")

                    # Use the current_step we already captured above (don't recalculate)
                    # If we don't have it yet, get it now
                    if current_step is None:
                        current_step = trainer.state.global_step if hasattr(trainer, 'state') else 0

                    # Calculate epoch based on step and total steps
                    if parent.training_phase == 'pre-training':
                        # For pre-training, report special value
                        calculated_epoch = -1
                    else:
                        # For main training, calculate actual epoch (0-based)
                        # Subtract the GRPO start step to reset epoch counter at the start of GRPO training
                        grpo_steps = current_step - parent.grpo_start_step
                        steps_per_epoch = max(1, max_steps / parent.config.num_train_epochs)
                        calculated_epoch = grpo_steps / steps_per_epoch

                    # Convert to our format - use current_step for continuous step numbering across phases
                    metrics = {
                        'epoch': calculated_epoch,
                        'step': current_step,  # Use global current_step for continuous numbering
                        'loss': actual_loss,
                        'mean_reward': actual_reward,
                        'learning_rate': logs.get('learning_rate', parent.config.learning_rate),
                        'grad_norm': grad_norm,
                        'reward_std': actual_reward_std,  # Use extracted value with fallbacks
                        'training_phase': parent.training_phase,  # Add phase indicator
                    }

                    # Add all GRPO-specific metrics from TRL
                    # Completion statistics
                    for key in ['completions/mean_length', 'completions/min_length', 'completions/max_length',
                                'completions/mean_terminated_length', 'completions/min_terminated_length',
                                'completions/max_terminated_length', 'completions/clipped_ratio']:
                        if key in logs:
                            metrics[key] = float(logs[key])

                    # Reward metrics
                    for key in ['reward', 'reward_std', 'frac_reward_zero_std']:
                        if key in logs:
                            metrics[key] = float(logs[key])

                    # KL divergence and entropy
                    if 'kl' in logs:
                        metrics['kl'] = float(logs['kl'])
                    if 'entropy' in logs:
                        metrics['entropy'] = float(logs['entropy'])

                    # Clip ratio metrics
                    for key in ['clip_ratio/region_mean', 'clip_ratio/low_mean', 'clip_ratio/low_min',
                                'clip_ratio/high_mean', 'clip_ratio/high_max']:
                        if key in logs:
                            metrics[key] = float(logs[key])

                    # Add all reward function outputs (reward/*)
                    for key, value in logs.items():
                        if key.startswith('reward/') and key not in metrics:
                            try:
                                metrics[key] = float(value)
                            except (ValueError, TypeError):
                                pass  # Skip non-numeric values

                    # Update tracking
                    if metrics['mean_reward'] > parent.best_reward:
                        parent.best_reward = metrics['mean_reward']
                        parent.save_checkpoint(f"best_step_{metrics['step']}")

                    parent.training_history.append(metrics)

                    # Callback to frontend
                    if parent.metrics_callback:
                        logger.info(f"Sending metrics update to frontend for step {current_step}: loss={actual_loss:.4f}, reward={actual_reward:.4f}, reward_std={actual_reward_std:.4f}")
                        logger.debug(f"Full metrics dict keys being sent: {list(metrics.keys())}")
                        parent.metrics_callback(metrics)
                    else:
                        logger.warning(f"No metrics callback registered - metrics not sent for step {current_step}")

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

            # Capture final training state from trainer before saving checkpoint
            if hasattr(trainer, 'state'):
                self.global_step = trainer.state.global_step
                self.current_epoch = int(trainer.state.epoch) if hasattr(trainer.state, 'epoch') else self.current_epoch

            # Save final checkpoint immediately after training completes
            logger.info("Training complete, saving final checkpoint...")
            self.save_checkpoint("final")

        except Exception as e:
            error_str = str(e)

            # Check for dimension mismatch errors
            if "batch2 tensor" in error_str and "Expected size" in error_str:
                logger.error(f"Dimension mismatch error detected: {e}")
                logger.info("Attempting to fix model state and retry...")

                try:
                    import torch as torch_module  # Re-import to ensure it's in scope

                    # Reset model completely
                    self.model.eval()
                    torch_module.cuda.empty_cache() if torch_module.cuda.is_available() else None

                    # Force model reinitialization for attention layers
                    for module in self.model.modules():
                        if hasattr(module, 'reset_parameters'):
                            try:
                                module.reset_parameters()
                            except:
                                pass

                    # Set model back to training mode
                    self.model.train()

                    # Recreate trainer with fresh state
                    trainer = GRPOTrainer(
                        model=self.model,
                        processing_class=self.tokenizer,
                        reward_funcs=reward_funcs,
                        args=grpo_config,
                        train_dataset=formatted_dataset,
                    )

                    # Re-apply generation config after recreation (TRL may reset it)
                    if hasattr(trainer.model, 'generation_config'):
                        logger.info("Re-applying generation config after trainer recreation")
                        trainer.model.generation_config.top_k = self.config.top_k
                        trainer.model.generation_config.top_p = self.config.top_p
                        trainer.model.generation_config.temperature = self.config.temperature
                        trainer.model.generation_config.max_length = self.config.max_sequence_length
                        trainer.model.generation_config.max_new_tokens = self.config.max_new_tokens
                        trainer.model.generation_config.repetition_penalty = self.config.repetition_penalty

                    # Reattach logging callback
                    trainer.log = custom_log

                    logger.info("Retrying training with reset model state...")
                    trainer.train()
                    logger.info("Training completed after model state reset")

                    # Capture final training state from trainer before saving checkpoint
                    if hasattr(trainer, 'state'):
                        self.global_step = trainer.state.global_step
                        self.current_epoch = int(trainer.state.epoch) if hasattr(trainer.state, 'epoch') else self.current_epoch

                    self.save_checkpoint("final")

                except Exception as retry_error:
                    logger.error(f"Training failed even after model reset: {retry_error}")
                    # Provide more helpful error message
                    if "batch2 tensor" in str(retry_error):
                        logger.error("This appears to be a persistent model architecture incompatibility issue.")
                        logger.error("Try using a different model or adjusting the training configuration.")
                    raise retry_error

            # Check if it's a compilation-related error
            elif "Dynamo" in error_str or "compile" in error_str.lower():
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

                    # Capture final training state from trainer before saving checkpoint
                    if hasattr(trainer, 'state'):
                        self.global_step = trainer.state.global_step
                        self.current_epoch = int(trainer.state.epoch) if hasattr(trainer.state, 'epoch') else self.current_epoch

                    self.save_checkpoint("final")
                except Exception as retry_error:
                    logger.error(f"Training failed even in eager mode: {retry_error}")
                    raise retry_error
            else:
                # Not a known error type, re-raise
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
        # Handle infinity values which are not valid JSON
        best_reward_value = self.best_reward
        if best_reward_value == float('-inf'):
            best_reward_value = None
        elif best_reward_value == float('inf'):
            best_reward_value = None
        elif best_reward_value != best_reward_value:  # Check for NaN
            best_reward_value = None

        state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_reward': best_reward_value,
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
        # Handle None values that replaced infinity
        best_reward = state['best_reward']
        self.best_reward = best_reward if best_reward is not None else float('-inf')
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
