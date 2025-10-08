"""GRPO (Group Relative Policy Optimization) trainer module."""
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

# Import device manager and logger utilities first
from .device_manager import get_device, is_cuda_available, use_unsloth, get_optimal_device_map
from utils.logging_config import get_logger

# Initialize logger first before using it
logger = get_logger(__name__)

# Conditional Unsloth import - MUST import before trl, transformers, peft for optimizations
UNSLOTH_AVAILABLE = False
FastModel = None
if use_unsloth():
    try:
        from unsloth import FastModel
        UNSLOTH_AVAILABLE = True
        logger.info("[OK] Unsloth optimizations enabled")
    except ImportError as e:
        logger.warning(f"[WARN] Unsloth not available despite CUDA: {e}")
        logger.warning("[WARN] Falling back to standard HuggingFace model loading")
else:
    logger.info("[INFO] Running in CPU mode - Unsloth optimizations disabled")

# Import ML libraries AFTER unsloth to ensure optimizations are applied
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
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
from accelerate import Accelerator
import safetensors.torch

# Import remaining core modules
from .dataset_handler import DatasetHandler, DatasetConfig
from .prompt_templates import PromptTemplate
from .custom_rewards import CustomRewardBuilder
from .system_config import SystemConfig, TrainingConfig
from .model_exporter import ModelExporter


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
    lora_r: int = 1  # Changed from 16 → 1
    lora_alpha: int = 32  # Changed from 16 → 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = None
    lora_bias: str = "none"

    # Training configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6  # Changed from 2e-4 → 1e-6
    warmup_steps: int = 10
    warmup_ratio: float = 0.0  # Changed from 0.1 → 0.0
    logging_steps: int = 1  # Log every step for real-time frontend metrics
    save_steps: int = 100
    eval_steps: int = 100
    max_grad_norm: float = 1.0  # Changed from 0.3 → 1.0
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"  # Changed from "constant" → "cosine"
    seed: int = 42

    # Pre-training configuration
    pre_training_epochs: int = 3  # Default: 3 epochs for better format learning
    pre_training_max_samples: Optional[int] = 500  # We default to 500 for format learning
    pre_training_filter_by_length: bool = False  # Filter pre-training samples by length
    pre_training_max_length_ratio: float = 0.5  # Max length as ratio of max_sequence_length
    pre_training_learning_rate: float = 5e-5  # Higher LR for supervised pre-training vs RL (configurable)

    # Adaptive pre-training configuration
    adaptive_pre_training: bool = True  # Enable adaptive pre-training with validation
    pre_training_min_success_rate: float = 0.8  # Minimum format compliance rate (80%)
    pre_training_max_additional_epochs: int = 3  # Maximum additional epochs if validation fails
    pre_training_validation_samples: int = 5  # Number of samples to validate format learning

    # GRPO/GSPO specific
    loss_type: str = "grpo"  # Always use "grpo" for TRL compatibility
    importance_sampling_level: str = "token"  # Options: "token" (GRPO), "sequence" (GSPO)
    max_sequence_length: int = 5120  # Changed from 2048 → 5120 (1024 prompt + 4096 completion)
    max_new_tokens: int = 4096  # Changed from 512 → 4096
    num_generations_per_prompt: int = 4  # Changed from 2 → 4
    num_generations: int = 4  # Changed from 2 → 4 (same as num_generations_per_prompt for TRL compatibility)
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    kl_penalty: float = 0.0  # Changed from 0.05 → 0.0
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

    def _combine_instruction_and_input(self, instruction: str, input_text: str) -> str:
        """Combine instruction and input fields for prompt.

        This follows the Alpaca/Stanford format where instruction and input
        are separate fields that need to be combined for the actual prompt.

        Args:
            instruction: The instruction/question
            input_text: Additional context/input data

        Returns:
            Combined prompt text
        """
        if not input_text or input_text.strip() == '':
            return instruction

        # If instruction ends with punctuation, add double newline for separation
        # Otherwise, add single space
        if instruction and instruction[-1] in '.!?':
            return f"{instruction}\n\n{input_text}"
        else:
            return f"{instruction} {input_text}"

    def _generate_reasoning_placeholder(self, example: Dict[str, Any], template) -> str:
        """Generate dynamic reasoning placeholder for short responses.

        Uses multiple strategies to create contextually appropriate reasoning:
        1. Template configuration (reasoning_placeholder_template)
        2. Instruction-based generation
        3. Input context extraction
        4. Domain detection heuristics
        5. Generic fallback

        Args:
            example: Dataset example with instruction/input/response fields
            template: PromptTemplate instance with configuration

        Returns:
            Generated reasoning placeholder text

        Examples:
            Template configuration (in JSON):
            {
                "reasoning_placeholder_template": "Based on the provided technical indicators and analysis."
            }

            Advanced Jinja2 template:
            {
                "reasoning_placeholder_template": "Analyzing: {{ instruction.split('.')[0] }}"
            }

            Available variables in template:
            - instruction: The instruction field from the example
            - input: The input field from the example
            - combined_prompt: The instruction and input combined
            - example: Full example dictionary with all fields
        """
        from jinja2 import Template as Jinja2Template, TemplateSyntaxError

        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        combined_prompt = self._combine_instruction_and_input(instruction, input_text)

        # Strategy 1: Use template configuration if available
        if template.config.reasoning_placeholder_template:
            try:
                jinja_template = Jinja2Template(template.config.reasoning_placeholder_template)
                return jinja_template.render(
                    instruction=instruction,
                    input=input_text,
                    combined_prompt=combined_prompt,
                    example=example
                ).strip()
            except (TemplateSyntaxError, Exception) as e:
                logger.warning(f"Failed to render reasoning_placeholder_template: {e}. Using fallback.")

        # Strategy 2: Extract from instruction (use first sentence/question)
        if instruction:
            # Try to get the first sentence
            first_sentence = instruction.split('.')[0].strip()
            if first_sentence and len(first_sentence) > 10:
                # If it's a question, rephrase as statement
                if first_sentence.endswith('?'):
                    return f"Analyzing the question: {first_sentence}"
                else:
                    return f"Based on the instruction: {first_sentence}."

        # Strategy 3: Detect domain and use appropriate reasoning
        combined_lower = combined_prompt.lower()

        # Technical/financial analysis
        if any(term in combined_lower for term in ['indicator', 'analysis', 'signal', 'market', 'stock', 'trade', 'price']):
            return "Based on the provided technical indicators and analysis."

        # Mathematics
        if any(term in combined_lower for term in ['calculate', 'solve', 'equation', 'proof', 'theorem', 'mathematics']):
            return "Working through the mathematical problem step by step."

        # Code/Programming
        if any(term in combined_lower for term in ['code', 'function', 'program', 'implement', 'algorithm', 'debug']):
            return "Analyzing the programming task and requirements."

        # General Q&A
        if any(term in combined_lower for term in ['question', 'answer', 'explain', 'what', 'why', 'how']):
            return "Considering the question and available information."

        # Strategy 4: Generic fallback
        return "Based on the given information and requirements."

    def setup_model(self,
                   model_name: Optional[str] = None,
                   use_unsloth_arg: bool = True) -> Tuple[Any, Any]:
        """Setup model and tokenizer.

        Args:
            model_name: Model name (overrides config)
            use_unsloth_arg: Whether to use Unsloth for loading (if available)

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or self.config.model_name
        logger.info(f"Loading model: {model_name}")

        # Check if Unsloth is available and requested
        should_use_unsloth = UNSLOTH_AVAILABLE and use_unsloth_arg

        if should_use_unsloth:
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

                logger.info("Model loaded with Unsloth optimizations")

            except Exception as e:
                logger.warning(f"Unsloth loading failed: {e}, falling back to standard loading")
                should_use_unsloth = False

        if not should_use_unsloth:
            # Standard loading with transformers
            logger.info("Loading model with standard HuggingFace transformers")

            # Quantization not supported on CPU
            use_quantization = (self.config.use_4bit or self.config.use_8bit) and is_cuda_available()

            if use_quantization:
                compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=self.config.use_4bit,
                    load_in_8bit=self.config.use_8bit,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.use_nested_quant,
                )
            else:
                bnb_config = None
                if not is_cuda_available():
                    logger.info("Running on CPU - quantization disabled")

            # Get optimal device map
            device_map = get_optimal_device_map(model_name)

            # Determine dtype for CPU mode
            if not is_cuda_available():
                # Use float32 for CPU for better compatibility
                torch_dtype = torch.float32
            elif self.config.fp16:
                torch_dtype = torch.float16
            elif self.config.bf16:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir,
                low_cpu_mem_usage=True,
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

            # Prepare model for training (only if using quantization)
            if use_quantization:
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                # Enable gradient checkpointing for CPU mode if configured
                if self.config.gradient_checkpointing and hasattr(self.model, 'enable_input_require_grads'):
                    self.model.enable_input_require_grads()

            # Setup LoRA
            self.setup_lora()

            logger.info(f"Model loaded successfully on {get_device()}")

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
            target_modules=self.config.lora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",      # Attention
                "gate_proj", "up_proj", "down_proj"          # MLP
            ],
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

            # Tokenize to get lengths (batched for memory efficiency)
            def get_length(examples):
                # Process batch of examples
                lengths = []
                for i in range(len(examples.get('instruction', examples.get('text', [])))):
                    if 'instruction' in examples:
                        text = str(examples['instruction'][i]) + str(examples.get('response', examples.get('output', ['']))[i])
                    else:
                        text = str(examples.get('text', [''])[i])
                    tokens = self.tokenizer(text, truncation=False)
                    lengths.append(len(tokens['input_ids']))
                return {'length': lengths}

            dataset_with_lengths = pre_train_dataset.map(
                get_length,
                batched=True,
                batch_size=1000,
                load_from_cache_file=False,  # Don't cache intermediate results
                desc="Computing sequence lengths"
            )
            pre_train_dataset = dataset_with_lengths.filter(lambda x: x['length'] <= max_length)
            logger.info(f"Filtered dataset from {original_size} to {len(pre_train_dataset)} samples by length")

            # Free memory from intermediate dataset
            del dataset_with_lengths
            gc.collect()
            logger.info("Freed memory from length filtering")

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

            # Log full system prompt and chat template for verification
            logger.info("=" * 80)
            logger.info("PRE-FINE-TUNING - SYSTEM PROMPT AND CHAT TEMPLATE")
            logger.info("=" * 80)

            # Log system prompt
            system_prompt = template.config.system_prompt or ''
            logger.info(f"System Prompt ({len(system_prompt)} chars):")
            logger.info("-" * 80)
            logger.info(system_prompt)
            logger.info("-" * 80)

            # Log full chat template
            chat_template = self.tokenizer.chat_template if hasattr(self.tokenizer, 'chat_template') else 'Not available'
            logger.info(f"Chat Template ({len(chat_template) if isinstance(chat_template, str) else 0} chars):")
            logger.info("-" * 80)
            logger.info(chat_template)
            logger.info("-" * 80)
            logger.info("=" * 80)

        # Set padding side to 'right' for training (required by DataCollatorForLanguageModeling)
        self.tokenizer.padding_side = 'right'

        # Apply template to dataset for pre-training
        # CRITICAL: Use the same prompt format as GRPO will use, but include the response
        def apply_template(example):
            # Format the prompt exactly as GRPO will see it
            if 'instruction' in example:
                # Combine instruction and input fields (Alpaca format)
                combined_prompt = self._combine_instruction_and_input(
                    example.get('instruction', ''),
                    example.get('input', '')
                )

                # Get the prompt format (what GRPO will use)
                prompt_messages = [
                    {'role': 'user', 'content': combined_prompt}
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
                    # CRITICAL: Preserve the actual response content instead of using placeholder text
                    # IMPORTANT: Don't add reasoning_start since the prompt already ends with it (from add_generation_prompt)
                    if not has_reasoning_markers and not has_solution_markers:
                        # Use the template's configured markers (not hardcoded defaults)
                        reasoning_end = template.config.reasoning_end_marker
                        solution_start = template.config.solution_start_marker
                        solution_end = template.config.solution_end_marker

                        # For structured responses (like technical analysis with separate reasoning and signal):
                        # - If response is short (likely just a label/signal), use it as the solution
                        # - Add minimal reasoning placeholder to teach structure
                        # For longer responses, split or duplicate appropriately
                        response_stripped = response.strip()

                        # If response looks like a single-word/short label (e.g., "STRONG_BUY")
                        # treat it as the solution and generate minimal reasoning
                        if len(response_stripped.split()) <= 3 and len(response_stripped) < 50:
                            # Short response: likely a label/classification
                            # Put it in the solution section, add dynamic reasoning placeholder
                            # NOTE: No reasoning_start because prompt already has it
                            reasoning_placeholder = self._generate_reasoning_placeholder(example, template)
                            response = (f"{reasoning_placeholder}\n"
                                      f"{reasoning_end}\n"
                                      f"{solution_start}\n"
                                      f"{response_stripped}\n"
                                      f"{solution_end}")
                        else:
                            # Longer response: use the full response content in both sections
                            # This teaches the model to generate properly formatted outputs
                            # NOTE: No reasoning_start because prompt already has it
                            response = (f"{response_stripped}\n"
                                      f"{reasoning_end}\n"
                                      f"{solution_start}\n"
                                      f"{response_stripped}\n"
                                      f"{solution_end}")

                # Combine prompt and response for training
                formatted = prompt + response

                # Add EOS token if needed (critical for preventing infinite generation)
                # Add for all templates, not just model_type=='grpo'
                if self.tokenizer and self.tokenizer.eos_token:
                    eos = self.tokenizer.eos_token
                    if not formatted.endswith(eos):
                        formatted += eos
                elif not formatted.endswith('</s>'):
                    # Fallback to </s> if tokenizer doesn't have eos_token
                    formatted += '</s>'
            else:
                # Fallback for non-standard format
                formatted = template.apply(example, mode='training')

            # Log sample format for debugging (only first example)
            if not hasattr(apply_template, 'logged'):
                logger.info("=" * 80)
                logger.info("PRE-TRAINING FORMAT SAMPLE (First Training Example)")
                logger.info("=" * 80)
                # Show where prompt ends and response begins
                if prompt in formatted:
                    prompt_end_idx = len(prompt)
                    logger.info(f"PROMPT (last 200 chars): ...{formatted[max(0, prompt_end_idx-200):prompt_end_idx]}")
                    logger.info(f"RESPONSE: {formatted[prompt_end_idx:min(len(formatted), prompt_end_idx+500)]}")
                    if len(formatted) > prompt_end_idx + 500:
                        logger.info(f"... ({len(formatted) - prompt_end_idx - 500} more chars)")
                else:
                    logger.info(f"Full formatted sample (first 1000 chars): {formatted[:1000]}")
                logger.info("=" * 80)
                apply_template.logged = True

            return {'text': formatted}

        formatted_dataset = pre_train_dataset.map(
            apply_template,
            remove_columns=pre_train_dataset.column_names,  # Remove original columns, keep only text
            load_from_cache_file=False,  # Don't cache to reduce memory usage
            desc="Applying chat template"
        )

        # Clean up pre_train_dataset to free memory
        del pre_train_dataset
        gc.collect()

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
            remove_columns=['text'],  # Remove text, keep only tokenized data
            load_from_cache_file=False  # Don't cache to reduce memory usage
        )

        # Clean up formatted_dataset to free memory
        del formatted_dataset
        gc.collect()

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
            learning_rate=self.config.pre_training_learning_rate,  # Higher LR for supervised pre-training (not RL like main GRPO training)
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

        # CRITICAL: Update self.model with the trained model from trainer
        # The trainer may wrap the model, so we get the underlying model
        # This ensures validation and subsequent GRPO training use the trained weights
        if hasattr(trainer, 'model'):
            self.model = trainer.model
            logger.info("Updated self.model with trained weights from trainer")

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

    def validate_format_learning(
        self,
        dataset: Dataset,
        template: PromptTemplate,
        num_samples: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Validate that the model has learned the expected format.

        Args:
            dataset: Dataset to sample from for validation
            template: Prompt template with expected markers
            num_samples: Number of samples to test (defaults to config)

        Returns:
            Tuple of (success_rate, detailed_results)
                success_rate: Float 0.0-1.0 indicating % of samples with correct format
                detailed_results: Dict with breakdown of validation results
        """
        if num_samples is None:
            num_samples = self.config.pre_training_validation_samples

        logger.info("=" * 80)
        logger.info(f"VALIDATING FORMAT LEARNING ({num_samples} samples)")
        logger.info("=" * 80)

        # Ensure model is in eval mode
        self.model.eval()

        # Sample random examples from dataset with fixed seed for reproducibility
        import random
        random.seed(self.config.seed)
        sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        results = {
            'total_samples': num_samples,
            'passed': 0,
            'failed': 0,
            'missing_reasoning': 0,
            'missing_solution': 0,
            'wrong_order': 0,
            'samples': []
        }

        for idx in sample_indices:
            test_sample = dataset[idx]

            # Get instruction
            instruction = test_sample.get('instruction', test_sample.get('input', ''))
            if not instruction:
                logger.warning(f"Sample {idx} has no instruction, skipping")
                continue

            # Combine instruction and input if needed
            combined_instruction = self._combine_instruction_and_input(
                test_sample.get('instruction', ''),
                test_sample.get('input', '')
            )

            try:
                # Format prompt using template
                messages = [{'role': 'user', 'content': combined_instruction}]
                formatted_prompt = template.apply(messages, mode='inference')

                # Tokenize - use config max_sequence_length for consistency
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_sequence_length
                ).to(self.model.device)

                # Generate response - use deterministic sampling for consistent validation
                # Use full config max_new_tokens (don't cap - model needs room for both sections)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,  # Deterministic for consistent validation
                        temperature=None,  # Explicitly override model's generation_config
                        top_p=None,  # Explicitly override model's generation_config
                        top_k=None,  # Explicitly override model's generation_config
                        repetition_penalty=self.config.repetition_penalty,  # Prevent infinite loops
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )

                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract response (everything after prompt)
                if formatted_prompt in generated_text:
                    response = generated_text[len(formatted_prompt):].strip()
                else:
                    # Fallback: look for reasoning marker
                    if template.config.reasoning_start_marker in generated_text:
                        response = generated_text[generated_text.index(template.config.reasoning_start_marker):].strip()
                    else:
                        response = generated_text
                        logger.debug(f"Sample {idx}: Could not find prompt in generated text, using full output as response")

                # Validate format
                is_valid, validation_info = template.validate_grpo_format(response)

                # Log detailed information for failed samples (use INFO for visibility)
                if not is_valid:
                    logger.info(f"Sample {idx} VALIDATION FAILURE DETAILS:")
                    logger.info(f"  Generated text length: {len(generated_text)} chars")
                    logger.info(f"  Response length: {len(response)} chars")
                    logger.info(f"  Has reasoning: {validation_info['has_reasoning']}")
                    logger.info(f"  Has solution: {validation_info['has_solution']}")
                    logger.info(f"  Errors: {validation_info['errors']}")
                    logger.info(f"  Looking for: reasoning_start='{template.config.reasoning_start_marker}', reasoning_end='{template.config.reasoning_end_marker}'")
                    logger.info(f"               solution_start='{template.config.solution_start_marker}', solution_end='{template.config.solution_end_marker}'")
                    logger.info(f"  Response preview (first 300 chars): {response[:300]}")
                    if len(response) > 300:
                        logger.info(f"  Response end (last 300 chars): ...{response[-300:]}")
                    else:
                        logger.info(f"  Full response: {response}")

                sample_result = {
                    'instruction': combined_instruction[:100] + '...' if len(combined_instruction) > 100 else combined_instruction,
                    'response': response[:200] + '...' if len(response) > 200 else response,
                    'valid': is_valid,
                    'has_reasoning': validation_info['has_reasoning'],
                    'has_solution': validation_info['has_solution'],
                    'errors': validation_info['errors']
                }

                results['samples'].append(sample_result)

                if is_valid:
                    results['passed'] += 1
                    logger.info(f"Sample {idx}: PASS")
                    # Log first successful sample for comparison
                    if results['passed'] == 1:
                        logger.debug(f"Sample {idx} SUCCESS (first pass):")
                        logger.debug(f"  Response length: {len(response)} chars")
                        logger.debug(f"  Response: {response}")

                else:
                    results['failed'] += 1
                    logger.warning(f"✗ Sample {idx}: FAIL - {', '.join(validation_info['errors'])}")

                    # For first failure, log even more detail to help diagnose
                    if results['failed'] == 1:
                        logger.info(f"FIRST FAILURE - Full generated text for debugging:")
                        logger.info(f"  Formatted prompt (last 200 chars): ...{formatted_prompt[-200:]}")
                        logger.info(f"  Generated text: {generated_text}")

                    if not validation_info['has_reasoning']:
                        results['missing_reasoning'] += 1
                    if not validation_info['has_solution']:
                        results['missing_solution'] += 1

            except Exception as e:
                logger.error(f"Validation error for sample {idx}: {e}")
                results['failed'] += 1
                results['samples'].append({
                    'instruction': combined_instruction[:100],
                    'response': f"ERROR: {str(e)}",
                    'valid': False,
                    'has_reasoning': False,
                    'has_solution': False,
                    'errors': [str(e)]
                })

            # Clear GPU cache after each validation sample to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate success rate
        total_tested = results['passed'] + results['failed']
        success_rate = results['passed'] / total_tested if total_tested > 0 else 0.0

        # Log summary
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Samples tested: {total_tested}")
        logger.info(f"Passed: {results['passed']} ({success_rate*100:.1f}%)")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Missing reasoning markers: {results['missing_reasoning']}")
        logger.info(f"Missing solution markers: {results['missing_solution']}")
        logger.info(f"Success rate: {success_rate*100:.1f}%")
        logger.info("=" * 80)

        return success_rate, results

    def adaptive_pre_fine_tune(
        self,
        dataset: Dataset,
        template: PromptTemplate
    ) -> Dict[str, Any]:
        """Adaptive pre-fine-tuning with automatic format validation and correction.

        This method runs pre-fine-tuning and validates format learning. If the model
        hasn't learned the format well enough, it automatically adds more epochs
        until the success rate threshold is met or max additional epochs is reached.

        Args:
            dataset: Training dataset
            template: Prompt template

        Returns:
            Training metrics including validation results
        """
        if not self.config.adaptive_pre_training:
            # Adaptive mode disabled, just run normal pre-training
            logger.info("Adaptive pre-training disabled, running standard pre-training")
            return self.pre_fine_tune(dataset, template)

        logger.info("=" * 80)
        logger.info("STARTING ADAPTIVE PRE-FINE-TUNING")
        logger.info("=" * 80)
        logger.info(f"Initial epochs: {self.config.pre_training_epochs}")
        logger.info(f"Target success rate: {self.config.pre_training_min_success_rate*100:.0f}%")
        logger.info(f"Max additional epochs: {self.config.pre_training_max_additional_epochs}")
        logger.info(f"Validation samples: {self.config.pre_training_validation_samples}")
        logger.info("=" * 80)

        # Run initial pre-training
        logger.info(f"\nPhase 1: Initial pre-training ({self.config.pre_training_epochs} epochs)")
        initial_metrics = self.pre_fine_tune(dataset, template, epochs=self.config.pre_training_epochs)

        # Validate format learning
        logger.info("\nPhase 2: Validating format learning...")
        success_rate, validation_results = self.validate_format_learning(dataset, template)

        all_validation_results = [
            {
                'epoch': self.config.pre_training_epochs,
                'success_rate': success_rate,
                'results': validation_results
            }
        ]

        additional_epochs = 0
        total_epochs = self.config.pre_training_epochs

        # Adaptive loop: add more epochs if needed
        while (success_rate < self.config.pre_training_min_success_rate and
               additional_epochs < self.config.pre_training_max_additional_epochs):

            additional_epochs += 1
            total_epochs += 1

            logger.info("=" * 80)
            logger.info(f"VALIDATION FAILED ({success_rate*100:.1f}% < {self.config.pre_training_min_success_rate*100:.0f}%)")
            logger.info(f"Adding 1 more epoch (total: {total_epochs}, additional: {additional_epochs}/{self.config.pre_training_max_additional_epochs})")
            logger.info("=" * 80)

            # Log diagnostic information
            if validation_results['missing_reasoning'] > 0:
                logger.info(f"  Issue: {validation_results['missing_reasoning']} samples missing reasoning markers")
            if validation_results['missing_solution'] > 0:
                logger.info(f"  Issue: {validation_results['missing_solution']} samples missing solution markers")

            # Run one more epoch of pre-training
            logger.info(f"\nRunning additional epoch {additional_epochs}...")
            self.pre_fine_tune(dataset, template, epochs=1)

            # Re-validate
            logger.info("\nRe-validating format learning...")
            success_rate, validation_results = self.validate_format_learning(dataset, template)

            all_validation_results.append({
                'epoch': total_epochs,
                'success_rate': success_rate,
                'results': validation_results
            })

        # Final summary
        logger.info("=" * 80)
        logger.info("ADAPTIVE PRE-TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {total_epochs}")
        logger.info(f"Additional epochs needed: {additional_epochs}")
        logger.info(f"Final success rate: {success_rate*100:.1f}%")

        if success_rate >= self.config.pre_training_min_success_rate:
            logger.info("✅ SUCCESS: Model has learned the format!")
        else:
            logger.warning(f"⚠️  WARNING: Success rate {success_rate*100:.1f}% below target {self.config.pre_training_min_success_rate*100:.0f}%")
            logger.warning("   Consider:")
            logger.warning("   - Increasing pre_training_epochs")
            logger.warning("   - Increasing pre_training_max_additional_epochs")
            logger.warning("   - Using shorter/simpler examples")
            logger.warning("   - Checking dataset quality")

        logger.info("=" * 80)

        # Compile comprehensive metrics
        return {
            'initial_metrics': initial_metrics,
            'total_epochs': total_epochs,
            'additional_epochs': additional_epochs,
            'final_success_rate': success_rate,
            'validation_history': all_validation_results,
            'target_success_rate': self.config.pre_training_min_success_rate,
            'success': success_rate >= self.config.pre_training_min_success_rate
        }

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

            # Combine instruction and input fields (Alpaca format)
            combined_prompt = self._combine_instruction_and_input(
                item.get('instruction', ''),
                item.get('input', '')
            )

            # TRL expects a specific format with prompt field
            formatted_item = {
                "prompt": [
                    {"role": "user", "content": combined_prompt}
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

    def _create_trl_reward_funcs(self, reward_builder: CustomRewardBuilder, trainer=None) -> List[Callable]:
        """Convert our reward builder to TRL's expected format.

        Args:
            reward_builder: Our custom reward builder
            trainer: TRL GRPOTrainer instance (optional, for accessing state)

        Returns:
            List of reward functions for TRL
        """
        # Capture model dtype and device for tensor conversion
        model_dtype = self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
        model_device = self.model.device if hasattr(self.model, 'device') else 'cpu'

        # Counter for sampling (shared across calls via closure)
        sample_counter = {'count': 0}
        # Store trainer reference for accessing step
        trainer_ref = {'instance': trainer}

        def reward_wrapper(prompts, completions, **kwargs):
            """Wrapper to adapt our reward function to TRL's format."""
            scores = []

            # Sample every 10th batch (adjustable)
            should_sample = (sample_counter['count'] % 10 == 0)
            logger.debug(f"Reward wrapper called: batch_count={sample_counter['count']}, should_sample={should_sample}, num_prompts={len(prompts)}")
            sample_counter['count'] += 1

            for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
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
                    reference=None,
                    tokenizer=self.tokenizer
                )

                # Use the total reward directly
                scores.append(reward)

                # Sample first 2 examples from sampled batches
                if should_sample and idx < 2 and self.metrics_callback:
                    # Try to get current step from trainer state if available
                    # Otherwise use batch count as approximate step (incremented before this check)
                    current_step = sample_counter['count']
                    if trainer_ref['instance'] and hasattr(trainer_ref['instance'], 'state'):
                        current_step = trainer_ref['instance'].state.global_step
                    elif self.global_step > 0:
                        current_step = self.global_step

                    reward_sample = {
                        'type': 'reward_sample',
                        'instruction': sample['instruction'],
                        'generated': response_text,
                        'total_reward': float(reward),
                        'components': {k: float(v) for k, v in reward_components.items()},
                        'step': current_step,
                        'timestamp': time.time()
                    }
                    logger.debug(f"Emitting reward sample: step={current_step}, batch_count={sample_counter['count']}, idx={idx}, reward={reward:.4f}")
                    self.metrics_callback(reward_sample)

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

        # Immediately notify frontend of phase change to reset visualizations
        if self.metrics_callback:
            phase_change_notification = {
                'training_phase': 'training',
                'step': self.global_step,
                'epoch': 0
            }
            self.metrics_callback(phase_change_notification)
            logger.info("Sent phase change notification to frontend to reset visualizations")

        # Check if pre-training checkpoint exists (for reference)
        pre_trained_path = Path(self.config.checkpoint_dir) / "pre_trained"
        if pre_trained_path.exists():
            logger.info("=" * 80)
            logger.info(f"Pre-trained checkpoint found at {pre_trained_path}")
            logger.info("Model already has trained adapter weights from pre-training phase")
            logger.info("(No need to reload - following official Unsloth SFT→GRPO approach)")

            # Verify adapters are active
            if hasattr(self.model, 'peft_config'):
                logger.info(f"Active PEFT adapters: {list(self.model.peft_config.keys())}")

            logger.info("=" * 80)
        else:
            logger.warning(f"Pre-trained checkpoint not found at {pre_trained_path}")
            logger.warning("Starting GRPO training without pre-training phase!")
            logger.warning("The model may not have learned the output format yet.")

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

            # Log full system prompt and chat template for verification
            logger.info("=" * 80)
            logger.info("GRPO TRAINING - SYSTEM PROMPT AND CHAT TEMPLATE")
            logger.info("=" * 80)

            # Log system prompt
            system_prompt = template.config.system_prompt or ''
            logger.info(f"System Prompt ({len(system_prompt)} chars):")
            logger.info("-" * 80)
            logger.info(system_prompt)
            logger.info("-" * 80)

            # Log full chat template
            chat_template = self.tokenizer.chat_template if hasattr(self.tokenizer, 'chat_template') else 'Not available'
            logger.info(f"Chat Template ({len(chat_template) if isinstance(chat_template, str) else 0} chars):")
            logger.info("-" * 80)
            logger.info(chat_template)
            logger.info("-" * 80)
            logger.info("=" * 80)

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

        # Calculate proper max_prompt_length from actual dataset (following official Unsloth notebook)
        # This prevents truncation of long prompts which destroys context
        logger.info("=" * 80)
        logger.info("Calculating optimal max_prompt_length from dataset prompts...")
        logger.info("=" * 80)

        prompt_lengths = []
        for item in formatted_dataset:
            # Apply chat template with add_generation_prompt to match actual GRPO usage
            prompt_text = self.tokenizer.apply_chat_template(
                item['prompt'],
                add_generation_prompt=True,
                tokenize=False
            )
            tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            prompt_lengths.append(len(tokens))

        # Calculate 90th percentile to avoid truncating most prompts
        # while filtering out extreme outliers (like official notebook does)
        import numpy as np
        percentile_90 = int(np.quantile(prompt_lengths, 0.9))
        max_prompt_length = percentile_90 + 10  # Add small buffer

        # Ensure we don't exceed total sequence length
        max_completion_length = self.config.max_sequence_length - max_prompt_length
        if max_completion_length < 256:
            # If completion space is too small, reduce prompt length
            logger.warning(f"Completion length would be too small ({max_completion_length}), adjusting...")
            max_prompt_length = self.config.max_sequence_length - 512
            max_completion_length = 512

        logger.info(f"Prompt length statistics:")
        logger.info(f"  Min: {min(prompt_lengths)} tokens")
        logger.info(f"  Mean: {int(np.mean(prompt_lengths))} tokens")
        logger.info(f"  Median: {int(np.median(prompt_lengths))} tokens")
        logger.info(f"  90th percentile: {percentile_90} tokens")
        logger.info(f"  Max: {max(prompt_lengths)} tokens")
        logger.info(f"  Configured max_prompt_length: {max_prompt_length} tokens")
        logger.info(f"  Configured max_completion_length: {max_completion_length} tokens")

        # Warn about prompts that will be truncated
        truncated_count = sum(1 for length in prompt_lengths if length > max_prompt_length)
        if truncated_count > 0:
            logger.warning(f"  {truncated_count}/{len(prompt_lengths)} prompts exceed max_prompt_length and will be truncated")
            logger.warning(f"  Consider filtering dataset or increasing max_sequence_length")

        logger.info("=" * 80)

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
            max_prompt_length=max_prompt_length,  # Calculated from actual prompt lengths
            max_completion_length=max_completion_length,  # Remainder of sequence length
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
            device = get_device()
            self.model = self.model.to(device)
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

            logger.info(f"Reset model state for GRPO training on {device}")
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

                # Send metrics only when we have actual training data from TRL
                # TRL calls log() multiple times per step - we want the call with actual metrics
                # Check for presence of key metrics that indicate this is a real training log
                # IMPORTANT: Must have reward metrics to avoid emitting 0.0 fallback values
                has_reward_metrics = (
                    'rewards/reward_wrapper/mean' in logs or
                    'reward' in logs
                )

                has_other_training_metrics = (
                    'loss' in logs or
                    'kl' in logs or
                    'epoch' in logs or
                    'completions/mean_length' in logs
                )

                # Only emit if we have reward metrics AND other training metrics
                # This prevents emitting partial logs with 0.0 reward fallbacks
                has_training_metrics = has_reward_metrics and has_other_training_metrics

                # Debug: Log why we're accepting or rejecting this log call
                if current_step and not has_training_metrics:
                    logger.debug(f"Step {current_step} - Skipping log call without complete metrics. Has reward: {has_reward_metrics}, Has other: {has_other_training_metrics}, Keys: {list(logs.keys())[:10]}")

                if current_step is not None and has_training_metrics and (parent.config.logging_steps == 1 or current_step % parent.config.logging_steps == 0):
                    logger.debug(f"Metrics logging triggered at step {current_step} with complete metrics (logging_steps={parent.config.logging_steps})")
                    # Extract actual values from the logs with comprehensive fallbacks
                    # Try multiple possible key names for loss
                    actual_loss = float(logs.get('loss', logs.get('train_loss', logs.get('train/loss', 0.0))))

                    # Try multiple possible key names for reward (TRL uses different keys)
                    # Use proper None checks to handle 0.0 as a valid reward value
                    actual_reward = float(
                        logs.get('rewards/reward_wrapper/mean')
                        if logs.get('rewards/reward_wrapper/mean') is not None  # TRL's primary key
                        else logs.get('reward')
                        if logs.get('reward') is not None                        # TRL's secondary key
                        else logs.get('rewards/mean')
                        if logs.get('rewards/mean') is not None                  # Alternative
                        else logs.get('mean_reward', 0.0)                        # Our own key with fallback
                    )

                    # Extract reward std with fallbacks
                    # Use proper None checks to handle 0.0 as a valid std value
                    actual_reward_std = float(
                        logs.get('rewards/reward_wrapper/std')
                        if logs.get('rewards/reward_wrapper/std') is not None   # TRL's primary key
                        else logs.get('reward_std')
                        if logs.get('reward_std') is not None                    # TRL's secondary key
                        else logs.get('rewards/std', 0.0)                        # Alternative with fallback
                    )

                    grad_norm = float(logs.get('grad_norm', 0.0))

                    # Debug: Log the actual values we extracted
                    logger.debug(f"Step {current_step} - Extracted metrics: loss={actual_loss}, reward={actual_reward}, reward_std={actual_reward_std}, grad_norm={grad_norm}")

                    # Use the current_step we already captured above (don't recalculate)
                    # If we don't have it yet, get it now
                    if current_step is None:
                        current_step = trainer.state.global_step if hasattr(trainer, 'state') else 0

                    # Update global_step in real-time for reward sampling
                    # This ensures reward samples emitted during training have correct step numbers
                    parent.global_step = current_step

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
                        'total_steps': max_steps,  # Total steps for progress calculation
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
                    current_reward = metrics['mean_reward']
                    if current_reward > parent.best_reward:
                        logger.info(f"New best reward: {current_reward:.6f} (previous: {parent.best_reward:.6f})")
                        parent.best_reward = current_reward
                        parent.save_checkpoint(f"best_step_{metrics['step']}")
                    else:
                        logger.debug(f"Current reward {current_reward:.6f} not better than best {parent.best_reward:.6f}")

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
        # If best_reward is still at initial value, calculate from training history
        final_reward = self.best_reward

        # Check if we have an empty or missing training history
        if not self.training_history:
            logger.error("Training history is empty! Metrics callback may not have fired during training.")
            logger.error("This suggests TRL did not provide expected log keys during training.")
            # Set final_reward to None if we have no data
            if final_reward == float('-inf'):
                final_reward = None
                logger.warning("No training metrics captured - best_reward will be None")
        elif final_reward == float('-inf'):
            logger.warning("best_reward not updated during training, calculating from training_history")
            # Find the maximum mean_reward from all logged metrics
            rewards = [m.get('mean_reward', float('-inf')) for m in self.training_history if 'mean_reward' in m]
            if rewards:
                final_reward = max(rewards)
                logger.info(f"Calculated best_reward from history: {final_reward:.6f}")
                # Update self.best_reward for checkpoint saving
                self.best_reward = final_reward
            else:
                logger.warning("No mean_reward found in training_history")
                final_reward = None

        return {
            'final_reward': final_reward,
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
            logger.warning(f"Checkpoint '{name}' has best_reward at initial value (-inf), may indicate tracking issue")
            best_reward_value = None
        elif best_reward_value == float('inf'):
            logger.warning(f"Checkpoint '{name}' has best_reward at infinity, replacing with None")
            best_reward_value = None
        elif best_reward_value != best_reward_value:  # Check for NaN
            logger.warning(f"Checkpoint '{name}' has best_reward as NaN, replacing with None")
            best_reward_value = None
        else:
            logger.info(f"Checkpoint '{name}' saving with best_reward: {best_reward_value:.6f}")

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
