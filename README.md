# GRPO Fine-Tuner

GUI for GRPO (Group Relative Policy Optimization) fine-tuning of Qwen and LLaMA models with full dataset/prompt customization and automatic system configuration detection.

## Features

### Core Capabilities
- **GRPO Training**: Advanced fine-tuning using Group Relative Policy Optimization
- **Multi-Model Support**: Compatible with Qwen2.5, LLaMA 3.x, Mistral, and more
- **LoRA Integration**: Efficient training with Low-Rank Adaptation
- **Custom Rewards**: Flexible reward system with regex matching, numerical comparison, and custom functions
- **Dataset Flexibility**: Support for HuggingFace datasets, local files (JSON, CSV, Parquet), and APIs
- **Prompt Templates**: Customizable templates with reasoning markers

### GUI Features
- **Intuitive Interface**: Tab-based organization for easy navigation
- **Real-time Monitoring**: Live training metrics and progress tracking
- **System Optimization**: Automatic hardware detection and configuration
- **Model Export**: Multiple export formats (SafeTensors, GGUF, HuggingFace)
- **Configuration Management**: Save and load training configurations

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Platform-Specific Notes

#### Windows Users
- vLLM is not supported on Windows; the application will automatically use transformers for generation
- Unsloth may have limited Windows support; install if compatible:
  ```bash
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  ```

#### Linux/Mac Users
- For faster generation, install vLLM:
  ```bash
  pip install vllm>=0.2.0
  ```
- Install Unsloth for optimized training:
  ```bash
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  ```

## Quick Start

### GUI Mode

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

## Usage Guide

### 1. Dataset Configuration

The application supports multiple data sources:

- **HuggingFace Hub**: Enter dataset name (e.g., "tatsu-lab/alpaca")
- **Local Files**: Browse for JSON, JSONL, CSV, or Parquet files
- **API Endpoints**: Provide URL for remote datasets
- **Direct Input**: Enter data directly for testing

### 2. Model Configuration

Select and configure your model:

- Choose base model (Qwen3, LLaMA 3.x, etc.)
- Configure LoRA parameters (rank, alpha, dropout)
- Set training hyperparameters (learning rate, batch size, epochs)

### 3. GRPO Settings

Configure GRPO-specific parameters:

- Generation temperature and top-p
- Number of generations per prompt
- KL penalty and clip range
- Custom reward functions

### 4. Training

Start training with real-time monitoring:

- Click "Start Training" to begin
- Monitor loss and reward metrics
- View training logs in real-time
- Pause or stop training as needed

### 5. Export

Export your trained model in various formats:

- SafeTensors (16-bit or 4-bit)
- GGUF (various quantization levels)
- HuggingFace format
- Direct HuggingFace Hub upload

## Configuration

### Example Configuration File

```json
{
  "model_name": "qwen2.5-0.5b",
  "dataset_source": "huggingface",
  "dataset_path": "tatsu-lab/alpaca",
  "lora_rank": 16,
  "learning_rate": 2e-4,
  "batch_size": 4,
  "num_epochs": 3,
  "temperature": 0.7,
  "use_flash_attention": false,
  "gradient_checkpointing": false,
  "mixed_precision": true
}
```

## Advanced Features

### Custom Reward Functions

Create custom reward functions in Python:

```python
def compute_reward(instruction, generated, reference):
    # Your custom logic here
    if "correct_answer" in generated:
        return 1.0
    return 0.0
```

### Prompt Templates

Define custom templates with special markers:

```
System: {system}
User: {instruction}
[THINKING]
# Model reasoning here
[/THINKING]
Assistant: {response}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Enable mixed precision and flash attention
3. **Import Errors**: Ensure all dependencies are installed
4. **GUI Not Starting**: Check Tkinter installation

### Debug Mode

Run with debug output:

```bash
python grpo_finetuner.py --debug
```

## Command Line Options

```
usage: grpo_finetuner.py [-h] [--config CONFIG] [--headless]
                         [--create-example-config] [--debug] [--version]

options:
  -h, --help            Show help message
  --config CONFIG       Configuration file path
  --headless           Run without GUI
  --create-example-config  Create example configuration
  --debug              Enable debug mode
  --version            Show version information
```

## License

This project is licensed under the MIT License.
