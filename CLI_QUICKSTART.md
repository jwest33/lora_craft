# LoRA Craft CLI - Quick Start Guide

This guide will walk you through using the LoRA Craft CLI to train a model without touching the web UI.

## Installation

### Step 1: Install the CLI

From the LoRA Craft project root:

```bash
pip install -e .
```

This installs the `loracraft` command globally.

### Step 2: Start the Server

The CLI requires the LoRA Craft Flask server to be running:

```bash
python server.py
```

Leave this running in a separate terminal.

### Step 3: Verify Installation

```bash
loracraft --help
```

You should see the CLI help menu with all available commands.

## Your First Training Session

### 1. Check System Status

First, verify your system is ready:

```bash
loracraft system status
```

This shows your GPU, VRAM, and RAM status.

### 2. Browse Available Datasets

```bash
loracraft dataset list
```

Filter by category:

```bash
loracraft dataset list --category math
```

### 3. Download a Dataset

Let's use the GSM8K math dataset:

```bash
loracraft dataset download openai/gsm8k --config main
```

Preview some samples:

```bash
loracraft dataset preview openai/gsm8k --config main --samples 5
```

Check the fields:

```bash
loracraft dataset fields openai/gsm8k --config main
```

### 4. Create a Training Configuration

Use the example config as a template:

```bash
cp cli/examples/example_config.yaml my_config.yaml
```

Edit `my_config.yaml` to customize your training. Key settings:

- `model_name`: Choose your base model (e.g., "unsloth/Qwen2.5-1.5B")
- `dataset_path`: Dataset name (e.g., "openai/gsm8k")
- `num_epochs`: How many epochs to train (3 is a good start)
- `reward_preset`: Reward function for your task ("math", "code", "reasoning", "qa")

### 5. Start Training

Start training with live monitoring:

```bash
loracraft train start my_config.yaml --watch
```

Or start without monitoring:

```bash
loracraft train start my_config.yaml
```

You'll get a session ID. Save this - you'll need it for testing and export!

### 6. Monitor Training (Optional)

If you didn't use `--watch`, you can monitor anytime:

```bash
loracraft train monitor <session_id>
```

This shows:
- Real-time metrics (loss, rewards, KL divergence)
- Live logs
- Training progress

Press `Ctrl+C` to exit (training continues).

### 7. View Training Status

Check status anytime:

```bash
loracraft train status <session_id>
```

View logs:

```bash
loracraft train logs <session_id>
```

Follow logs in real-time:

```bash
loracraft train logs <session_id> --follow
```

### 8. Test Your Model

Once training is complete, test it:

```bash
loracraft model test <session_id> --prompt "What is 15 * 24?"
```

Interactive testing session:

```bash
loracraft model interactive <session_id>
```

Then enter prompts interactively:

```
Prompt: What is 15 * 24?
Response: 15 * 24 = 360

Prompt: If John has 5 apples and gets 3 more, how many does he have?
Response: 5 + 3 = 8 apples

Prompt: exit
```

### 9. Export Your Model

Export to GGUF format for use with llama.cpp, Ollama, or LM Studio:

```bash
loracraft export create <session_id> --format gguf --quantization q4_k_m
```

Quantization options:
- `q4_k_m`: 4-bit (smallest, good quality) - **Recommended**
- `q5_k_m`: 5-bit (balanced)
- `q8_0`: 8-bit (larger, better quality)
- `f16`: 16-bit (largest, best quality)

Check export status:

```bash
loracraft export status <session_id>
```

## Common Workflows

### Training on Custom Data

1. Prepare your dataset as JSON, JSONL, CSV, or Parquet
2. Upload it:

```bash
loracraft dataset upload /path/to/my_dataset.json
```

3. Check the fields:

```bash
loracraft dataset fields uploads/my_dataset.json
```

4. Update your config with the correct field mappings
5. Start training as usual

### Comparing Multiple Models

Train several models with different configurations, then compare:

```bash
loracraft model compare <session1> <session2> <session3> \
  --prompt "Solve: 2x + 5 = 15" \
  --temperature 0.7
```

### Batch Export

Export multiple models at once:

```bash
loracraft export batch <session1> <session2> <session3> \
  --format gguf \
  --quantization q4_k_m
```

### List All Training Sessions

```bash
loracraft train list
```

Filter by status:

```bash
loracraft train list --status-filter running
loracraft train list --status-filter completed
```

### List All Trained Models

```bash
loracraft model list
```

Get detailed info:

```bash
loracraft model info <session_id>
```

## Tips & Tricks

### 1. Use Config Files

Save your configurations for reproducibility:

```bash
# Start with a config file
loracraft train start configs/math_training.yaml

# Reuse for different datasets
loracraft train start configs/code_training.yaml
```

### 2. Monitor in Background

Start training in one terminal, monitor in another:

```bash
# Terminal 1
loracraft train start my_config.yaml

# Terminal 2 (use the session ID from terminal 1)
loracraft train monitor <session_id>
```

### 3. Chain Commands

Stop training and immediately export:

```bash
loracraft train stop <session_id> --force
loracraft export create <session_id> --format gguf -q q4_k_m
```

### 4. Remote Server

Connect to a remote LoRA Craft server:

```bash
loracraft --server http://remote-server:5000 train list
```

Or set it permanently:

```bash
loracraft config set server_url http://remote-server:5000
```

### 5. Verbose Mode

For debugging, enable verbose output:

```bash
loracraft --verbose train start my_config.yaml
```

## Troubleshooting

### "Cannot connect to LoRA Craft server"

Make sure the Flask server is running:

```bash
python server.py
```

Check the server URL:

```bash
loracraft config get server_url
```

### "Dataset not found"

Download the dataset first:

```bash
loracraft dataset download <dataset_name>
```

### Training Stuck or Slow

Monitor system resources:

```bash
loracraft system status
```

Check if GPU is being used. If not, you may be in CPU mode (much slower).

### Export Failed

Make sure training is complete:

```bash
loracraft train status <session_id>
```

Status should be "completed" before exporting.

## Next Steps

- **Read the full CLI README**: `cli/README.md`
- **Explore example configs**: `cli/examples/`
- **Check the main docs**: `README.md`
- **Visit the website**: https://loracraft.org

## Quick Command Reference

| Task | Command |
|------|---------|
| Check system | `loracraft system status` |
| List datasets | `loracraft dataset list` |
| Download dataset | `loracraft dataset download <name>` |
| Start training | `loracraft train start config.yaml` |
| Monitor training | `loracraft train monitor <session_id>` |
| Test model | `loracraft model test <session_id> -p "prompt"` |
| Export model | `loracraft export create <session_id>` |
| List models | `loracraft model list` |
| Get help | `loracraft --help` |

---

**Happy Training! 🚀**

For issues or questions:
- GitHub: https://github.com/jwest33/lora_craft/issues
- Docs: https://loracraft.org
