"""GRPO (Group Relative Policy Optimization) trainer module."""

import os
import torch
import gc
import json
import time
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
import logging

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
from utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model configuration
    model_name: str
    use_4bit: bool = True
    use_8bit: bool = False
    load_in_4bit: bool = True
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
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    seed: int = 42

    # GRPO/GSPO specific
    loss_type: str = "grpo"  # Options: "grpo", "gspo", "dr_grpo"
    importance_sampling_level: str = "token"  # Options: "token", "sequence" (for GSPO)
    max_sequence_length: int = 2048
    num_generations_per_prompt: int = 4
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


class GRPOTrainer:
    """GRPO trainer for fine-tuning language models."""

    def __init__(self,
                 config: GRPOConfig,
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

        # Setup directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.output_dir, self.config.cache_dir, self.config.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

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
                from unsloth import FastLanguageModel

                # Load with Unsloth
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=self.config.max_sequence_length,
                    dtype=torch.float16 if self.config.fp16 else None,
                    load_in_4bit=self.config.load_in_4bit,
                )

                # Get LoRA model
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=self.config.lora_r,
                    target_modules=self.config.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias=self.config.lora_bias,
                    use_gradient_checkpointing=self.config.gradient_checkpointing,
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

        # Apply template to dataset
        def apply_template(example):
            formatted = template.apply(example, mode='training')
            return {'text': formatted}

        formatted_dataset = dataset.map(apply_template)

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
            train_dataset=formatted_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train
        train_result = trainer.train()

        # Save pre-trained model
        pre_trained_path = Path(self.config.checkpoint_dir) / "pre_trained"
        trainer.save_model(pre_trained_path)

        logger.info(f"Pre-fine-tuning completed. Model saved to {pre_trained_path}")

        return train_result.metrics

    def grpo_train(self,
                  dataset: Dataset,
                  template: PromptTemplate,
                  reward_builder: CustomRewardBuilder,
                  validation_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Main GRPO training loop.

        Args:
            dataset: Training dataset
            template: Prompt template
            reward_builder: Reward function builder
            validation_dataset: Optional validation dataset

        Returns:
            Training metrics
        """
        logger.info("Starting GRPO training")

        # Setup accelerator
        # Determine mixed precision setting
        if self.config.fp16:
            mixed_precision = "fp16"
        elif self.config.bf16:
            mixed_precision = "bf16"
        else:
            mixed_precision = None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )

        # Prepare model and optimizer
        self.model, self.optimizer, dataset = self.accelerator.prepare(
            self.model,
            self._setup_optimizer(),
            dataset
        )

        # Training loop
        for epoch in range(self.config.num_train_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(dataset, template, reward_builder)

            # Validation
            if validation_dataset:
                val_metrics = self._validate(validation_dataset, template, reward_builder)
                epoch_metrics['validation'] = val_metrics

            # Save checkpoint
            if epoch_metrics.get('mean_reward', 0) > self.best_reward:
                self.best_reward = epoch_metrics['mean_reward']
                self.save_checkpoint(f"best_epoch_{epoch}")

            self.training_history.append(epoch_metrics)

            # Callback
            if self.metrics_callback:
                self.metrics_callback(epoch_metrics)

        return self._compile_training_metrics()

    def _train_epoch(self,
                    dataset: Dataset,
                    template: PromptTemplate,
                    reward_builder: CustomRewardBuilder) -> Dict[str, Any]:
        """Train for one epoch.

        Args:
            dataset: Training dataset
            template: Prompt template
            reward_builder: Reward function

        Returns:
            Epoch metrics
        """
        self.model.train()
        epoch_losses = []
        epoch_rewards = []
        epoch_kl_penalties = []

        progress_bar = tqdm(dataset, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Generate samples
            generations = self._generate_samples(batch, template)

            # Compute rewards
            rewards, reward_components = self._compute_rewards(
                batch, generations, reward_builder
            )

            # Compute loss
            loss, kl_penalty = self._compute_grpo_loss(
                batch, generations, rewards
            )

            # Backward pass
            self.accelerator.backward(loss)

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    metrics = {
                        'loss': loss.item(),
                        'reward': np.mean(rewards),
                        'kl_penalty': kl_penalty.item() if kl_penalty else 0,
                    }
                    logger.info(f"Step {self.global_step}: {metrics}")

                    if self.metrics_callback:
                        self.metrics_callback(metrics)

            epoch_losses.append(loss.item())
            epoch_rewards.extend(rewards)
            if kl_penalty:
                epoch_kl_penalties.append(kl_penalty.item())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': np.mean(epoch_losses[-100:]),
                'reward': np.mean(epoch_rewards[-100:])
            })

            # Progress callback
            if self.progress_callback:
                self.progress_callback(batch_idx / len(dataset))

        return {
            'epoch': self.current_epoch,
            'mean_loss': np.mean(epoch_losses),
            'mean_reward': np.mean(epoch_rewards),
            'mean_kl_penalty': np.mean(epoch_kl_penalties) if epoch_kl_penalties else 0,
            'std_reward': np.std(epoch_rewards),
        }

    def _generate_samples(self,
                         batch: Dict[str, Any],
                         template: PromptTemplate) -> List[str]:
        """Generate samples for a batch.

        Args:
            batch: Input batch
            template: Prompt template

        Returns:
            List of generated texts
        """
        # Apply template for inference
        prompts = []
        for i in range(len(batch['instruction'])):
            sample = {k: v[i] for k, v in batch.items()}
            prompt = template.apply(sample, mode='inference')
            prompts.append(prompt)

        # Use vLLM if available and not on Windows
        if not self.is_windows and self._try_vllm_generation(prompts):
            return self._vllm_generate(prompts)
        else:
            # Fallback to transformers generation
            return self._transformers_generate(prompts)

    def _try_vllm_generation(self, prompts: List[str]) -> bool:
        """Check if vLLM is available for generation.

        Args:
            prompts: List of prompts

        Returns:
            True if vLLM is available, False otherwise
        """
        if self.is_windows:
            return False

        try:
            import vllm
            return True
        except ImportError:
            return False

    def _vllm_generate(self, prompts: List[str]) -> List[str]:
        """Generate using vLLM (Linux/Mac only).

        Args:
            prompts: List of prompts

        Returns:
            List of generated texts
        """
        try:
            from vllm import LLM, SamplingParams

            # Initialize vLLM model if not already done
            if not hasattr(self, 'vllm_model'):
                self.vllm_model = LLM(
                    model=self.config.model_name,
                    dtype="float16" if self.config.fp16 else "auto",
                    gpu_memory_utilization=0.9,
                )

            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_tokens=self.config.max_sequence_length // 2,
                n=self.config.num_generations_per_prompt,
            )

            outputs = self.vllm_model.generate(prompts, sampling_params)

            generations = []
            for output in outputs:
                for generated in output.outputs:
                    generations.append(generated.text)

            return generations

        except Exception as e:
            logger.warning(f"vLLM generation failed: {e}, falling back to transformers")
            return self._transformers_generate(prompts)

    def _transformers_generate(self, prompts: List[str]) -> List[str]:
        """Generate using standard transformers.

        Args:
            prompts: List of prompts

        Returns:
            List of generated texts
        """
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_sequence_length // 2,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                num_return_sequences=self.config.num_generations_per_prompt,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        generations = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract only the generated part
            for prompt in prompts:
                if prompt in text:
                    text = text.split(prompt)[-1].strip()
                    break
            generations.append(text)

        return generations

    def _compute_rewards(self,
                        batch: Dict[str, Any],
                        generations: List[str],
                        reward_builder: CustomRewardBuilder) -> Tuple[List[float], List[Dict]]:
        """Compute rewards for generated samples.

        Args:
            batch: Input batch
            generations: Generated texts
            reward_builder: Reward function

        Returns:
            Tuple of (rewards, reward_components)
        """
        rewards = []
        components = []

        batch_size = len(batch['instruction'])
        gens_per_prompt = len(generations) // batch_size

        for i in range(batch_size):
            instruction = batch['instruction'][i]
            reference = batch.get('response', [None] * batch_size)[i]

            # Get generations for this prompt
            prompt_generations = generations[i * gens_per_prompt:(i + 1) * gens_per_prompt]

            # Compute rewards for each generation
            for gen in prompt_generations:
                reward, comp = reward_builder.compute_total_reward(
                    instruction, gen, reference
                )
                rewards.append(reward)
                components.append(comp)

        return rewards, components

    def _compute_grpo_loss(self,
                          batch: Dict[str, Any],
                          generations: List[str],
                          rewards: List[float]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute GRPO loss.

        Args:
            batch: Input batch
            generations: Generated texts
            rewards: Computed rewards

        Returns:
            Tuple of (loss, kl_penalty)
        """
        # This is a simplified GRPO loss
        # In practice, you would need to implement the full GRPO algorithm

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, device=self.model.device)

        # Normalize rewards
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Placeholder for actual loss computation
        # This would involve computing log probabilities and policy gradients
        loss = -rewards_normalized.mean()

        # KL penalty (simplified)
        kl_penalty = None
        if self.config.kl_penalty > 0:
            kl_penalty = torch.tensor(0.0, device=self.model.device)  # Placeholder

        return loss, kl_penalty

    def _validate(self,
                 dataset: Dataset,
                 template: PromptTemplate,
                 reward_builder: CustomRewardBuilder) -> Dict[str, Any]:
        """Validate model on validation set.

        Args:
            dataset: Validation dataset
            template: Prompt template
            reward_builder: Reward function

        Returns:
            Validation metrics
        """
        self.model.eval()
        val_rewards = []

        with torch.no_grad():
            for batch in tqdm(dataset, desc="Validation"):
                generations = self._generate_samples(batch, template)
                rewards, _ = self._compute_rewards(batch, generations, reward_builder)
                val_rewards.extend(rewards)

        return {
            'mean_reward': np.mean(val_rewards),
            'std_reward': np.std(val_rewards),
        }

    def _setup_optimizer(self):
        """Setup optimizer."""
        from torch.optim import AdamW

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        return optimizer

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

        logger.info(f"Checkpoint loaded from {path}")

    def export_model(self,
                    output_path: str,
                    format: str = 'safetensors',
                    quantization: Optional[str] = None):
        """Export trained model.

        Args:
            output_path: Output path
            format: Export format ('safetensors', 'gguf', 'huggingface')
            quantization: Quantization type ('4bit', '8bit')
        """
        logger.info(f"Exporting model to {format} format")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == 'safetensors' or format == 'huggingface':
            # Merge LoRA weights
            merged_model = self.model.merge_and_unload()

            # Save
            merged_model.save_pretrained(
                output_path,
                safe_serialization=(format == 'safetensors')
            )
            self.tokenizer.save_pretrained(output_path)

        elif format == 'gguf':
            # Export to GGUF format (requires llama.cpp)
            logger.warning("GGUF export requires external tools (llama.cpp)")
            # Implementation would require llama.cpp conversion tools

        else:
            raise ValueError(f"Unknown export format: {format}")

        logger.info(f"Model exported to {output_path}")

    def cleanup(self):
        """Cleanup resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Resources cleaned up")


if __name__ == "__main__":
    # Test GRPO trainer
    config = GRPOConfig(
        model_name="qwen2.5-0.5b",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )

    trainer = GRPOTrainer(config)
    print(f"GRPO Trainer initialized with config: {config.model_name}")
