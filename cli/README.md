# LoRA Craft CLI

A comprehensive command-line interface for managing the LoRA Craft GRPO training pipeline without requiring the web UI.

## Features

- **Complete Training Workflow**: Start, stop, and monitor training sessions from the command line
- **Dataset Management**: Download, upload, preview, and manage datasets
- **Model Testing**: Interactive and batch testing of trained models
- **Model Export**: Export models to GGUF and other formats with quantization options
- **Live Monitoring**: Real-time training dashboard with metrics, logs, and progress
- **Configuration Management**: Save and load training configurations
- **Batch Operations**: Perform operations on multiple models simultaneously

## Installation

### Quick Install

From the LoRA Craft project root:

```bash
pip install -e .
```

This installs the `loracraft` command globally.

### Manual Install

1. Install dependencies:
```bash
pip install -r cli/requirements.txt
```

2. Add to your PATH or run directly:
```bash
python -m cli.main --help
```

## Quick Start

### 1. Start the LoRA Craft Server

First, ensure the LoRA Craft Flask server is running:

```bash
python server.py
```

The CLI will connect to `http://localhost:5000` by default.

### 2. Check System Status

```bash
loracraft system status
```

### 3. List Available Datasets

```bash
loracraft dataset list
```

### 4. Download a Dataset

```bash
loracraft dataset download openai/gsm8k --config main
```

### 5. Start Training

Create a training configuration file (YAML or JSON), then:

```bash
loracraft train start config.yaml --watch
```

The `--watch` flag enables live monitoring.

### 6. Monitor Training

```bash
loracraft train monitor <session_id>
```

### 7. Test Your Model

```bash
loracraft model test <session_id> --prompt "What is 2+2?"
```

### 8. Export Model

```bash
loracraft export create <session_id> --format gguf --quantization q4_k_m
```

## Command Reference

### System Commands

```bash
# Show system status
loracraft system status

# Show detailed system info
loracraft system info
```

### Dataset Commands

```bash
# List available datasets
loracraft dataset list
loracraft dataset list --category math
loracraft dataset list --cached-only

# Download a dataset
loracraft dataset download <dataset_name>
loracraft dataset download openai/gsm8k --config main

# Preview dataset samples
loracraft dataset preview <dataset_name> --samples 10

# Show dataset fields
loracraft dataset fields <dataset_name>

# Upload custom dataset
loracraft dataset upload /path/to/dataset.json

# List uploaded datasets
loracraft dataset list-uploaded

# Check dataset status
loracraft dataset status <dataset_name>

# Clear cache
loracraft dataset clear-cache
```

### Training Commands

```bash
# Start training from config
loracraft train start config.yaml
loracraft train start config.yaml --watch  # With live monitoring

# Stop training
loracraft train stop <session_id>

# Get training status
loracraft train status <session_id>

# View logs
loracraft train logs <session_id>
loracraft train logs <session_id> --follow  # Follow mode

# List training sessions
loracraft train list
loracraft train list --status-filter running

# View metrics
loracraft train metrics <session_id>

# Live monitoring dashboard
loracraft train monitor <session_id>

# View training history
loracraft train history <session_id>
```

### Model Commands

```bash
# List trained models
loracraft model list

# List base models
loracraft model list --base

# Get model info
loracraft model info <session_id>

# Test model with a prompt
loracraft model test <session_id> --prompt "Your question here"
loracraft model test <session_id> -p "Question" -t 0.8 -m 512

# Interactive testing session
loracraft model interactive <session_id>

# Compare multiple models
loracraft model compare <session1> <session2> <session3> --prompt "Test question"

# Delete a model
loracraft model delete <session_id>
loracraft model delete <session_id> --force  # Skip confirmation
```

### Export Commands

```bash
# List export formats
loracraft export formats

# List exports for a session
loracraft export list <session_id>

# Create an export
loracraft export create <session_id>
loracraft export create <session_id> --format gguf --quantization q4_k_m

# Batch export
loracraft export batch <session1> <session2> <session3> --format gguf -q q4_k_m

# Check export status
loracraft export status <session_id>
```

### Configuration Commands

```bash
# Show CLI config
loracraft config show

# Set a config value
loracraft config set server_url http://localhost:5000
loracraft config set timeout 60

# Get a config value
loracraft config get server_url
```

## Configuration

The CLI stores its configuration in `~/.loracraft/config.yaml`.

### Default Configuration

```yaml
server_url: http://localhost:5000
timeout: 30
default_format: table
color: true
verbose: false
```

### Training Configuration Example

Create a YAML or JSON file with your training configuration:

```yaml
# config.yaml
model_name: unsloth/Qwen2.5-1.5B
dataset_path: openai/gsm8k
dataset_config: main

# LoRA settings
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0

# Training parameters
num_epochs: 3
batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002
warmup_steps: 10
max_sequence_length: 2048
max_new_tokens: 512

# GRPO settings
kl_penalty: 0.05
reward_preset: "math"

# Field mapping
instruction_field: question
response_field: answer
```

## Live Training Monitor

The live training monitor provides a real-time dashboard in your terminal:

```bash
loracraft train monitor <session_id>
```

Features:
- Real-time metrics display
- Live log streaming
- Progress bar
- Status updates
- Auto-refresh every 2 seconds

Press `Ctrl+C` to exit the monitor (training continues in the background).

## Examples

### Complete Training Workflow

```bash
# 1. Check system
loracraft system status

# 2. Download dataset
loracraft dataset download openai/gsm8k --config main

# 3. Preview data
loracraft dataset preview openai/gsm8k --config main --samples 5

# 4. Start training with monitoring
loracraft train start my_config.yaml --watch

# 5. Test the model
loracraft model test <session_id> --prompt "If John has 5 apples..."

# 6. Export model
loracraft export create <session_id> --format gguf --quantization q4_k_m
```

### Batch Model Comparison

```bash
# Test multiple models on the same prompt
loracraft model compare session1 session2 session3 \
  --prompt "Explain quantum computing" \
  --temperature 0.7 \
  --max-tokens 512
```

### Interactive Testing

```bash
# Start interactive session
loracraft model interactive <session_id>

# Then enter prompts interactively:
Prompt: What is 2+2?
Response: 2+2 = 4

Prompt: Solve: x + 5 = 10
Response: x = 5

Prompt: exit
```

## Advanced Usage

### Custom Server URL

```bash
# Connect to a remote server
loracraft --server http://remote-server:5000 train list
```

### Verbose Mode

```bash
# Enable detailed logging
loracraft --verbose train start config.yaml
```

### Following Logs in Real-time

```bash
# Stream logs as they arrive
loracraft train logs <session_id> --follow
```

### Batch Operations

```bash
# Export multiple models at once
loracraft export batch session1 session2 session3 \
  --format gguf \
  --quantization q5_k_m
```

## Troubleshooting

### Connection Errors

If you see "Cannot connect to LoRA Craft server":

1. Ensure the Flask server is running: `python server.py`
2. Check the server URL: `loracraft config get server_url`
3. Update if needed: `loracraft config set server_url http://localhost:5000`

### Import Errors

If you get import errors:

1. Install CLI dependencies: `pip install -r cli/requirements.txt`
2. Or install the package: `pip install -e .`

### WebSocket Issues

If live monitoring doesn't work:

1. Check that the server supports WebSocket connections
2. Try using polling mode instead: `loracraft train logs <session_id> --follow`

## Development

### Project Structure

```
cli/
├── __init__.py           # Package initialization
├── main.py              # CLI entry point
├── client.py            # API client
├── config.py            # Configuration management
├── commands/            # Command modules
│   ├── __init__.py
│   ├── dataset.py       # Dataset commands
│   ├── train.py         # Training commands
│   ├── model.py         # Model commands
│   └── export.py        # Export commands
└── utils/               # Utilities
    ├── __init__.py
    ├── display.py       # Display helpers
    ├── formatters.py    # Data formatters
    └── monitor.py       # Live monitoring
```

### Contributing

Contributions are welcome! To add new commands:

1. Create a new command module in `cli/commands/`
2. Define command group with `@click.group()`
3. Add commands with `@command.command()`
4. Register in `cli/main.py`

## License

MIT License - see LICENSE file for details

## Support

- **GitHub Issues**: https://github.com/jwest33/lora_craft/issues
- **Documentation**: https://loracraft.org
- **Discord**: Coming soon

---

**Note**: This CLI tool requires a running LoRA Craft server. Make sure to start the Flask app before using CLI commands.
