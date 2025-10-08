---
layout: default
title: Quick Start - LoRA Craft
---

# Quick Start Guide

Get from zero to your first fine-tuned model in minutes.


## Before You Begin

Ensure you have:
- NVIDIA GPU with 8GB+ VRAM
- 32GB+ System RAM
- Python 3.11 or higher
- 20GB+ free disk space

Not sure if your system is ready? Check the [Prerequisites](documentation.html#prerequisites).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jwest33/lora_craft.git
cd lora_craft
```

### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see `CUDA available: True`.

---

## Starting the Application

### 1. Launch the Server

```bash
python flask_app.py
```

### 2. Open the Interface

Navigate to `http://localhost:5000` in your web browser.

You should see the LoRA Craft interface with tabs for Model, Dataset, Config, Reward, and Training.

---

## Training Your First Model

Follow this 7-step workflow to train a math reasoning model.

### Step 1: Select Your Model

1. Click the **Model** tab
2. Choose **Recommended** preset
3. Select **Qwen3** model family
4. Choose **Qwen/Qwen2.5-1.5B-Instruct**
5. Click **Load Model**

**Why Qwen3 1.5B?**
- Small enough for most GPUs
- Fast training (minutes, not hours)
- Strong baseline performance

### Step 2: Choose a Dataset

1. Click the **Dataset** tab
2. Select **Public Datasets**
3. Filter by **Math** category
4. Choose **GSM8K** (8,500 grade school math problems)
5. Click **Load Dataset**
6. Preview samples to verify data format

**What is GSM8K?**
- Grade school math word problems
- Requires multi-step reasoning
- Perfect for testing GRPO training

### Step 3: Configure Training

1. Click the **Config** tab
2. Use these beginner-friendly settings:

**Training Duration:**
- Epochs: `1`
- Samples per epoch: `500` (subset for quick test)

**Batch Settings:**
- Batch size: `1`
- Gradient accumulation: `4`

**Learning Rate:**
- Learning rate: `0.0002`
- Warmup steps: `10`
- Scheduler: `constant`

**Generation:**
- Max sequence length: `2048`
- Max new tokens: `512`
- Temperature: `0.7`

**Pre-training:**
- Enabled: `Yes`
- Epochs: `1`
- Max samples: `100`

### Step 4: Select Reward Function

1. Click the **Reward** tab
2. Choose **Preset Library**
3. Select **Math & Science** category
4. Pick **Math Problem Solver** reward
5. Verify field mappings:
   - Instruction → `question`
   - Response → `answer`
6. Click **Test Reward** with a sample to verify

**How Rewards Work:**
The reward function checks if the model's answer matches the expected solution, rewarding correct answers with 1.0 and incorrect with 0.0.

### Step 5: Start Training

1. Click the **Training** tab
2. Review your configuration summary
3. Click **Start Training**
4. Watch the real-time metrics appear

**What to Watch:**
- **Mean Reward**: Should increase over time (target: 0.5+)
- **Training Loss**: Should decrease
- **KL Divergence**: Should stay relatively stable (< 0.1)

Training 500 samples on a 1.5B model takes approximately 10-15 minutes on a modern GPU.

### Step 6: Export Your Model

Once training completes:

1. Navigate to the **Export** section
2. Choose format:
   - **HuggingFace**: For Python/API use
   - **GGUF (Q4_K_M)**: For llama.cpp/Ollama/LM Studio
3. Click **Export Model**
4. Wait for conversion (1-2 minutes)

Your model is saved in `exports/<session_id>/`

### Step 7: Test Your Model

1. Click the **Test** tab
2. Select your newly trained model
3. Enter a test problem:
   ```
   Sarah has 5 apples. She buys 3 more apples.
   Then she gives 2 apples to her friend.
   How many apples does Sarah have now?
   ```
4. Click **Generate**
5. Compare the output to the base model

**Expected Improvement:**
Your fine-tuned model should show structured reasoning and correct answers more consistently than the base model.

---

## Quick Workflow Summary

```
1. Select Model → Qwen3 1.5B
2. Load Dataset → GSM8K (Math)
3. Configure Training → 1 epoch, 500 samples
4. Choose Reward → Math & Science
5. Start Training → ~15 minutes
6. Export Model → GGUF format
7. Test Output → Verify improvement
```

---

## Common First-Time Issues

### Training is too slow
- Reduce samples per epoch to 200
- Check GPU is being used (metrics should show GPU memory usage)
- Reduce max sequence length to 1024

### Out of memory errors
- Reduce batch size to 1
- Increase gradient accumulation to 8
- Use smaller model (Qwen3 0.6B)

### Rewards stay at 0.0
- Check field mappings match your dataset
- Verify reward function with test button
- Try a different reward function from presets

### Model outputs are gibberish
- Enable pre-training (helps model learn format)
- Increase pre-training epochs to 2
- Check system prompt matches expected output format

---

## Next Steps

### Try Different Tasks

**Code Generation**
- Dataset: Code Alpaca
- Reward: Code Generation
- Model: Qwen3 1.5B or Phi-2

**Question Answering**
- Dataset: SQuAD v2
- Reward: Question Answering
- Model: Llama 3.2 3B

**Creative Writing**
- Dataset: Alpaca
- Reward: Creative Writing
- Model: Mistral 7B

### Scale Up

Once comfortable with the basics:
- Train on full datasets (remove sample limits)
- Increase epochs to 3-5 for better results
- Try larger models (3B-7B parameters)
- Experiment with custom reward functions

### Deploy Your Models

**Use with Ollama:**
```bash
ollama create math-tutor -f exports/<session_id>/Modelfile
ollama run math-tutor "Solve: 15 × 12 ="
```

**Use with llama.cpp:**
```bash
./main -m exports/<session_id>/model-q4_k_m.gguf \
  -p "Calculate the area of a circle with radius 7"
```

**Integrate via API:**
Load your HuggingFace format model in any Python application with the Transformers library.

---

## Learn More

<div class="quickstart-nav">
  <div class="nav-card">
    <h3>Explore Features</h3>
    <p>Deep dive into GRPO, reward functions, and advanced training options</p>
    <a href="features.html">View Features →</a>
  </div>

  <div class="nav-card">
    <h3>Read Full Documentation</h3>
    <p>Complete technical reference, API docs, and troubleshooting</p>
    <a href="documentation.html">View Docs →</a>
  </div>

  <div class="nav-card">
    <h3>See Use Cases</h3>
    <p>Real-world applications and example configurations</p>
    <a href="use-cases.html">View Use Cases →</a>
  </div>
</div>

---

## Need Help?

- **Documentation**: [Full technical guide](documentation.html)
- **GitHub Issues**: [Report bugs or request features](https://github.com/jwest33/lora_craft/issues)
- **Discussions**: [Ask questions and share tips](https://github.com/jwest33/lora_craft/discussions)

Happy fine-tuning! 🎉
