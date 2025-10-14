# LoRA Craft Configuration Schema

This document describes the standard configuration format used across LoRA Craft's frontend UI and CLI.

## Overview

All training configurations use a **nested JSON structure** with metadata and training parameters separated. This allows for better organization, versioning, and sharing of configurations.

## File Format

- **Format**: JSON only (`.json`)
- **Location**: `configs/` directory (or `configs/examples/` for examples)
- **Character Encoding**: UTF-8

## Configuration Structure

```json
{
  "name": "config_name",
  "description": "Description of what this config does",
  "config": {
    // Actual training configuration parameters
  },
  "timestamp": 1234567890000
}
```

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for this configuration |
| `description` | string | No | Human-readable description of the configuration purpose |
| `config` | object | Yes | The actual training configuration object (see below) |
| `timestamp` | number | No | Unix timestamp in milliseconds when config was created/modified |

### Configuration Object (`config`)

The `config` object contains all training parameters organized into logical sections:

#### 1. Setup Mode

```json
"setupMode": "setup-recommended"
```

Options: `"setup-recommended"`, `"setup-custom"`

#### 2. Model Configuration

```json
"model": {
  "modelName": "unsloth/Qwen2.5-1.5B",
  "customModelPath": "",
  "quantization": "q8_0",
  "loraRank": 16,
  "loraAlpha": 32,
  "loraDropout": 0,
  "targetModulesArray": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "family": "qwen"
}
```

**Fields:**
- `modelName` (string): HuggingFace model identifier or local path
- `customModelPath` (string): Custom local model path (if not using HF model)
- `quantization` (string): Quantization format (e.g., `"q8_0"`, `"q4_0"`, `"none"`)
- `loraRank` (number): LoRA rank (typically 8, 16, 32, or 64)
- `loraAlpha` (number): LoRA alpha (typically 2x rank)
- `loraDropout` (number): Dropout rate for LoRA layers (0.0 - 0.1)
- `targetModulesArray` (array): List of module names to apply LoRA to
- `family` (string): Model family identifier (e.g., `"qwen"`, `"llama"`, `"mistral"`)

#### 3. Dataset Configuration

```json
"dataset": {
  "source": "huggingface",
  "path": "openai/gsm8k",
  "split": "train",
  "sample_size": 0,
  "train_split": 80,
  "max_samples": null,
  "instruction_field": "question",
  "response_field": "answer"
}
```

**Fields:**
- `source` (string): Dataset source (`"huggingface"`, `"upload"`, `"local"`)
- `path` (string): Dataset path or HuggingFace identifier
- `split` (string): Dataset split to use (`"train"`, `"test"`, `"validation"`)
- `sample_size` (number): Number of samples to use (0 = all)
- `train_split` (number): Percentage for train/validation split (0-100)
- `max_samples` (number|null): Maximum samples per epoch (null = unlimited)
- `instruction_field` (string): Column name for instruction/input text
- `response_field` (string): Column name for response/output text

#### 4. Template Configuration

```json
"template": {
  "reasoning_start": "<analysis>",
  "reasoning_end": "</analysis>",
  "solution_start": "<signal>",
  "solution_end": "</signal>",
  "system_prompt": "Your system prompt here...",
  "chat_template": "Jinja2 template string...",
  "chat_template_type": "custom"
}
```

**Fields:**
- `reasoning_start` (string): Marker for start of reasoning/thinking section
- `reasoning_end` (string): Marker for end of reasoning section
- `solution_start` (string): Marker for start of final answer/solution
- `solution_end` (string): Marker for end of solution
- `system_prompt` (string): System prompt instruction for the model
- `chat_template` (string): Jinja2 chat template
- `chat_template_type` (string): Template type (`"custom"`, `"alpaca"`, `"chatml"`, etc.)

#### 5. Algorithm Configuration

```json
"algorithm": {
  "selected": "grpo",
  "epsilon": 0.0003,
  "epsilon_high": 0.0004
}
```

**Fields:**
- `selected` (string): Training algorithm (`"grpo"`, `"sft"`, `"dpo"`)
- `epsilon` (number): GRPO epsilon parameter (exploration factor)
- `epsilon_high` (number): Upper bound for epsilon

#### 6. Training Configuration

```json
"training": {
  "num_epochs": 3,
  "batch_size": 12,
  "gradient_accumulation": 1,
  "learning_rate": 0.00001,
  "lr_schedule": "cosine",
  "lr_scheduler_type": "constant",
  "warmup_ratio": 0.1,
  "warmup_steps": 10,
  "weight_decay": 0.001,
  "max_grad_norm": 0.3,
  "max_sequence_length": 300,
  "max_new_tokens": 600,
  "seed": 42,
  "optimizer": "paged_adamw_32bit"
}
```

**Fields:**
- `num_epochs` (number): Number of training epochs
- `batch_size` (number): Batch size per device
- `gradient_accumulation` (number): Gradient accumulation steps
- `learning_rate` (number): Initial learning rate
- `lr_schedule` (string): Learning rate schedule (`"linear"`, `"cosine"`, `"constant"`)
- `lr_scheduler_type` (string): Scheduler type
- `warmup_ratio` (number): Ratio of total steps for warmup
- `warmup_steps` (number): Number of warmup steps
- `weight_decay` (number): Weight decay for regularization
- `max_grad_norm` (number): Maximum gradient norm for clipping
- `max_sequence_length` (number): Maximum input sequence length
- `max_new_tokens` (number): Maximum tokens to generate
- `seed` (number): Random seed for reproducibility
- `optimizer` (string): Optimizer type (`"adamw"`, `"paged_adamw_32bit"`, etc.)

#### 7. GRPO Configuration

```json
"grpo": {
  "num_generations": 2,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "repetition_penalty": 1.01,
  "kl_weight": 0.1,
  "kl_penalty": 0.05,
  "clip_range": 0.2,
  "value_coefficient": 1
}
```

**Fields:**
- `num_generations` (number): Number of generations per prompt for GRPO
- `temperature` (number): Sampling temperature (0.0 - 2.0)
- `top_p` (number): Nucleus sampling threshold (0.0 - 1.0)
- `top_k` (number): Top-K sampling parameter
- `repetition_penalty` (number): Penalty for token repetition
- `kl_weight` (number): KL divergence weight
- `kl_penalty` (number): KL penalty coefficient
- `clip_range` (number): PPO-style clipping range
- `value_coefficient` (number): Value loss coefficient

#### 8. Optimization Configuration

```json
"optimizations": {
  "use_flash_attention": false,
  "gradient_checkpointing": false,
  "mixed_precision": true,
  "use_bf16": true
}
```

**Fields:**
- `use_flash_attention` (boolean): Enable Flash Attention 2
- `gradient_checkpointing` (boolean): Enable gradient checkpointing (saves VRAM)
- `mixed_precision` (boolean): Enable mixed precision training
- `use_bf16` (boolean): Use BFloat16 instead of FP16

#### 9. Output Configuration

```json
"output": {
  "name": "",
  "save_steps": 100,
  "eval_steps": 100,
  "logging_steps": 10
}
```

**Fields:**
- `name` (string): Output directory name (empty = auto-generated)
- `save_steps` (number): Save checkpoint every N steps
- `eval_steps` (number): Evaluate every N steps
- `logging_steps` (number): Log metrics every N steps

#### 10. Pre-training Configuration

```json
"pre_training": {
  "enabled": true,
  "epochs": 2,
  "max_samples": 300,
  "filter_by_length": false,
  "validate_format": true
}
```

**Fields:**
- `enabled` (boolean): Enable supervised pre-training phase
- `epochs` (number): Number of pre-training epochs
- `max_samples` (number): Maximum samples for pre-training
- `filter_by_length` (boolean): Filter out very long/short samples
- `validate_format` (boolean): Validate data format before training

#### 11. Fine-tuning Internals

```json
"fine_tuning_internals": {
  "dataset_processing_batch_size": 1000,
  "short_response_max_words": 3,
  "short_response_max_chars": 50,
  "reward_sample_interval": 10,
  "reward_samples_per_batch": 2,
  "log_preview_chars": 300,
  "prompt_length_percentile": 0.9,
  "prompt_length_buffer": 10,
  "min_completion_length": 256,
  "fallback_completion_length": 512,
  "min_loss_threshold": 0.01
}
```

Advanced internal parameters for fine-tuning behavior.

#### 12. Reward Configuration

```json
"reward": {
  "type": "preset",
  "preset_name": "Math Reasoning"
}
```

**Fields:**
- `type` (string): Reward type (`"preset"` or `"custom"`)
- `preset_name` (string): Name of reward preset (for type `"preset"`)

## Example Configurations

Example configurations can be found in `configs/examples/`:

- `gsm8k_math_example.json` - Math reasoning on GSM8K dataset
- `technical_analysis.2025.10.10.10.1.json` - Technical analysis for trading signals

## Using Configurations

### Frontend UI

Configurations are automatically saved to and loaded from the `configs/` directory through the web interface.

### CLI

Load a configuration file with the CLI:

```bash
loracraft train start configs/examples/gsm8k_math_example.json
```

The CLI automatically handles both nested (new) and flat (legacy) config formats.

## Validation

All configurations are validated before training using the backend's validation system. Required fields include:

- Model configuration (model name or path)
- Dataset configuration (source and path)
- Basic training parameters (epochs, batch size, learning rate)

## Notes

- **Legacy Format**: The CLI previously used a flat JSON/YAML format. This is still supported for backward compatibility, but the nested format is now standard.
- **YAML Support**: YAML is only used for CLI user settings (`~/.loracraft/config.yaml`), not for training configurations.
- **Timestamps**: Timestamps are stored in Unix milliseconds for consistency across systems.
- **Config Names**: Config names should be unique and use alphanumeric characters, underscores, and dots.
