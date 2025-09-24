# GRPO Fine-Tuner

GUI application for GRPO (Group Relative Policy Optimization) fine-tuning of language models using Unsloth framework with comprehensive dataset customization, system optimization, and multi-format export capabilities.

## Features

### Core Capabilities
- **GRPO Training**: State-of-the-art fine-tuning using Group Relative Policy Optimization via TRL
- **Unsloth Integration**: Optimized training with Unsloth framework for 2x faster performance
- **Multi-Model Support**: Compatible with Qwen3, LLaMA 3.2, and Phi-4 model families
- **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning with extensive configuration
- **Custom Reward System**: Flexible rewards with regex matching, numerical validation, and custom Python functions
- **Dataset Flexibility**: Support for HuggingFace datasets, local files (JSON, CSV, Parquet), and API endpoints
- **Prompt Templates**: Customizable templates with reasoning markers and special tokens

### GUI Features
- **Modern Themed Interface**: Light, Dark, and Synthwave themes with custom styling
- **6 Specialized Tabs**: Dataset, Model & Training, GRPO Settings, System, Monitoring, and Export
- **Real-time Monitoring**: Live training metrics, loss curves, and reward tracking
- **System Optimization**: Automatic hardware detection with GPU/CPU/RAM monitoring
- **Configuration Management**: Save and load training configurations in JSON format
- **Validation System**: Built-in configuration validation before training

## Supported Models

### Qwen3 Family
- `unsloth/Qwen3-0.6B` - 600M parameters (~1.2GB VRAM)
- `unsloth/Qwen3-1.7B` - 1.7B parameters (~3.4GB VRAM)
- `unsloth/Qwen3-4B` - 4B parameters (~8GB VRAM)
- `unsloth/Qwen3-8B` - 8B parameters (~16GB VRAM)

### LLaMA 3.2 Family
- `unsloth/Llama-3.2-1B-Instruct` - 1B parameters (~2GB VRAM)
- `unsloth/Llama-3.2-3B-Instruct` - 3B parameters (~6GB VRAM)

### Phi-4 Family
- `unsloth/phi-4-reasoning` - 15B parameters (~30GB VRAM)

## Installation


### Install Dependencies

#### Step 1: Install PyTorch with CUDA support
```bash
# For CUDA 12.8
pip install torch--index-url https://download.pytorch.org/whl/cu128
```

#### Step 2: Install remaining dependencies
```bash
pip install -r requirements.txt
```

### Platform-Specific Notes

#### Windows Users
- vLLM is not supported on Windows; transformers will be used for generation
- Full Unsloth support with optimized kernels
- NVIDIA GPU monitoring requires nvidia-ml-py

#### Linux/Mac Users
- Optional: Install vLLM for faster generation:
  ```bash
  pip install vllm>=0.2.0
  ```
- Full Unsloth optimization support

## Quick Start

### GUI Mode (Default)
```bash
python grpo_finetuner.py
```

### Headless Mode (No GUI)
```bash
python grpo_finetuner.py --headless --config configs/example.json
```

### Create Example Configuration
```bash
python grpo_finetuner.py --create-example-config
```

### Debug Mode
```bash
python grpo_finetuner.py --debug
```

## GUI Tabs Overview

### 1. Dataset Tab
- **Data Sources**: HuggingFace Hub, local files, API endpoints, direct input
- **Format Support**: JSON, JSONL, CSV, Parquet
- **Field Mapping**: Configure instruction/response fields
- **Data Preview**: View and validate dataset samples
- **Template Testing**: Test prompt templates with your data

### 2. Model & Training Tab
- **Model Selection**: Choose from supported Unsloth models
- **LoRA Configuration**:
  - Rank, Alpha, Dropout settings
  - Target modules (attention + FFN layers)
  - Automatic optimization recommendations
- **Training Parameters**:
  - Learning rate with scheduling
  - Batch size and gradient accumulation
  - Epochs and warmup steps
  - Mixed precision and gradient checkpointing

### 3. GRPO Settings Tab
- **Generation Parameters**: Temperature, Top-p, Top-k sampling
- **GRPO Specific**:
  - KL penalty for policy regularization
  - Clip range for stable training
  - Number of generations per prompt
- **Reward Configuration**: Setup custom reward functions

### 4. System Tab
- **Hardware Detection**: Automatic GPU/CPU capabilities assessment
- **Memory Management**: VRAM and RAM optimization
- **Performance Settings**: Flash attention, CPU offloading options
- **System Information**: Real-time resource monitoring

### 5. Monitoring Tab
- **Training Progress**: Real-time loss and reward curves
- **Metrics Display**: Training speed, memory usage, ETA
- **Log Viewer**: Integrated logging with filtering
- **Checkpoint Management**: Save and resume training

### 6. Export Tab
- **Export Formats**:
  - SafeTensors (16-bit and 4-bit quantization)
  - GGUF (Q4_K_M, Q5_K_M, Q8_0 quantization)
  - HuggingFace format
- **HuggingFace Hub**: Direct upload to model repository
- **Batch Export**: Multiple format export in one operation

## Configuration

### Example Configuration File

```json
{
  "model_name": "unsloth/Qwen3-1.7B",
  "dataset_source": "huggingface",
  "dataset_path": "tatsu-lab/alpaca",
  "dataset_split": "train",
  "instruction_field": "instruction",
  "response_field": "output",
  "template_name": "default",
  "reasoning_start": "[THINKING]",
  "reasoning_end": "[/THINKING]",
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.0,
  "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
  "learning_rate": 5e-6,
  "batch_size": 1,
  "gradient_accumulation": 1,
  "num_epochs": 2,
  "warmup_steps": 10,
  "temperature": 0.7,
  "top_p": 0.95,
  "num_generations": 4,
  "kl_penalty": 0.01,
  "clip_range": 0.2,
  "use_flash_attention": false,
  "gradient_checkpointing": false,
  "mixed_precision": true,
  "output_dir": "./outputs",
  "checkpoint_dir": "./checkpoints",
  "export_model": true,
  "export_path": "./exports/model",
  "export_format": "safetensors",
  "pre_finetune": true
}
```

## Advanced Features

### Custom Reward Functions

#### Binary Rewards
```python
def compute_reward(instruction, generated, reference):
    # Check for specific criteria
    if "correct_answer" in generated.lower():
        return 1.0
    return 0.0
```

#### Regex-based Rewards
```python
# Configure in GUI or config file
{
  "reward_type": "regex",
  "regex_pattern": r"\[ANSWER\]:\s*\d+",
  "weight": 1.0
}
```

#### Numerical Comparison
```python
# Extract and compare numerical values
{
  "reward_type": "numerical",
  "extract_number": true,
  "number_tolerance": 0.01,
  "relative_tolerance": false
}
```

### Prompt Templates

Define custom templates with special markers:

```
System: You are a helpful AI assistant.
User: {instruction}
[THINKING]
Let me think about this step by step...
[/THINKING]
Assistant: {response}
```

### Training Strategies

#### Pre-Fine-Tuning Phase
- Initial supervised fine-tuning before GRPO
- Helps establish baseline performance
- Configurable epochs and learning rate

#### GRPO Training
- Group relative optimization for better generalization
- Multiple generations per prompt for diversity
- KL-regularized policy optimization

### Memory (poor) Estimates by Model Size
- 0.6B models: ~2GB VRAM + 4GB RAM
- 1-2B models: ~4GB VRAM + 8GB RAM
- 4B models: ~8GB VRAM + 16GB RAM
- 8B models: ~16GB VRAM + 32GB RAM
- 15B models: ~30GB VRAM + 64GB RAM

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size to 1
   - Enable gradient checkpointing
   - Use 4-bit quantization
   - Reduce sequence length

2. **Slow Training**
   - Enable mixed precision (fp16/bf16)
   - Use flash attention (if supported)
   - Increase gradient accumulation
   - Check GPU utilization

3. **Import Errors**
   - Ensure PyTorch is installed with CUDA support
   - Verify Unsloth installation
   - Check Python version compatibility

4. **GUI Not Starting**
   - Verify Tkinter installation: `python -m tkinter`
   - Check display settings on remote systems
   - Try headless mode as alternative

### Debug Mode

Run with enhanced logging:
```bash
python grpo_finetuner.py --debug
```

Check logs in `./logs/` directory for detailed information.

## Command Line Options

```
usage: grpo_finetuner.py [-h] [--config CONFIG] [--headless]
                         [--create-example-config] [--debug] [--version]

GRPO Fine-Tuner - Train language models with GRPO

options:
  -h, --help               Show help message
  --config CONFIG          Configuration file path (JSON)
  --headless              Run without GUI
  --create-example-config  Create example configuration file
  --debug                 Enable debug mode with verbose logging
  --version               Show version information (v1.0.0)
```

## Project Structure

```
grpo_gui/
├── core/                  # Core training modules
│   ├── grpo_trainer.py   # GRPO implementation
│   ├── dataset_handler.py # Dataset processing
│   ├── custom_rewards.py  # Reward functions
│   ├── prompt_templates.py # Template system
│   └── system_config.py   # Hardware detection
├── gui/                   # GUI components
│   ├── app.py            # Main application
│   ├── tabs/             # Tab implementations
│   ├── themed_dialog.py  # Themed dialogs
│   ├── styled_widgets.py # Custom widgets
│   └── theme_manager.py  # Theme system
├── utils/                 # Utilities
│   ├── logging_config.py # Logging setup
│   └── validators.py     # Input validation
├── assets/               # Application assets
│   ├── icon.png         # App icon
│   └── Rationale-Regular.ttf # Custom font
├── configs/              # Configuration files
├── logs/                 # Training logs
├── outputs/              # Model outputs
├── checkpoints/          # Training checkpoints
├── exports/              # Exported models
└── cache/                # Model cache
```

## License

This project is licensed under the MIT License.
