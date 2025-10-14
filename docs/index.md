---
layout: default
title: LoRA Craft - LLM Fine-Tuning
---

<div class="hero-section">
  <div class="content-logo">
    <img src="lora_craft.png" alt="LoRA Craft">
    <h1>LoRA Craft</h1>
  </div>

  <p class="hero-description">
    Fine-tune large language models into specialized assistants with a web-based interface.
    No ML expertise required—just choose your model, dataset, and reward function.
  </p>

  <div class="alpha-notice" style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin: 20px 0; color: #856404;">
    <strong>⚠️ Alpha Software Notice:</strong> LoRA Craft is currently in active development and should be considered alpha software. Features may change, and you may encounter bugs or instability. Thank you for your patience!
  </div>

  <div class="cta-buttons">
    <a href="quickstart.html" class="btn-primary">Get Started</a>
    <a href="https://github.com/jwest33/lora_craft" class="btn-secondary" target="_blank">
      <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
      </svg>
      View on GitHub
    </a>
  </div>
</div>

## What is LoRA Craft?

LoRA Craft is a web-based application for fine-tuning large language models using reinforcement learning. It combines GRPO (Group Relative Policy Optimization) with LoRA adapters to enable efficient training on consumer GPUs, requiring no machine learning expertise or code.

---

## Why Choose LoRA Craft?

<div class="benefits-grid">
  <div class="benefit-card">
    <div class="benefit-icon">◎</div>
    <h3>No-Code Training</h3>
    <p>Configure and train models through an intuitive web interface. No Python scripts, no command-line complexity—just point, click, and train.</p>
  </div>

  <div class="benefit-card">
    <div class="benefit-icon">⬢</div>
    <h3>Docker Ready</h3>
    <p>Start in minutes with Docker—no manual dependency setup required. Works on Windows (WSL2), Linux, and macOS with automatic GPU detection.</p>
  </div>

  <div class="benefit-card">
    <div class="benefit-icon">▸</div>
    <h3>Efficient Fine-Tuning</h3>
    <p>Uses LoRA adapters to train on consumer GPUs (4-8GB VRAM). Fine-tune 7B models on your desktop with GRPO reinforcement learning.</p>
  </div>

  <div class="benefit-card">
    <div class="benefit-icon">▥</div>
    <h3>Real-Time Monitoring</h3>
    <p>Watch your model improve with live metrics: rewards, loss, KL divergence, and more. Interactive charts show training progress as it happens.</p>
  </div>

  <div class="benefit-card">
    <div class="benefit-icon">◆</div>
    <h3>Ready-to-Use Rewards</h3>
    <p>Choose from pre-built reward functions for math, coding, reasoning, and more. Or create custom rewards for your specific task.</p>
  </div>
</div>

---

## See It In Action

<div class="screenshot-showcase">
  <div class="screenshot-item">
    <img src="example_model_selection.png" alt="Model Selection Interface" class="clickable-image">
    <h4>1. Select Your Model</h4>
    <p>Choose from Qwen, Llama, Mistral, and Phi models. Configure LoRA parameters or use recommended defaults.</p>
  </div>

  <div class="screenshot-item">
    <img src="example_dataset_selection.png" alt="Dataset Configuration" class="clickable-image">
    <h4>2. Choose Your Dataset</h4>
    <p>Browse curated datasets or upload your own. Auto-detects field mappings for instant configuration.</p>
  </div>

  <div class="screenshot-item">
    <img src="example_reward_catalog.png" alt="Reward Function Library" class="clickable-image">
    <h4>3. Pick a Reward Function</h4>
    <p>Select from categorized reward functions optimized for math, coding, reasoning, and creative tasks.</p>
  </div>

  <div class="screenshot-item">
    <img src="example_training_metrics.png" alt="Training Dashboard" class="clickable-image">
    <h4>4. Monitor Training</h4>
    <p>Track real-time metrics with interactive charts. Watch rewards increase and loss decrease as your model learns.</p>
  </div>

  <div class="screenshot-item">
    <img src="example_stock_trade_question.png" alt="Model Testing Interface" class="clickable-image">
    <h4>5. Test & Compare Models</h4>
    <p>Evaluate your fine-tuned models against base models with interactive testing, side-by-side comparison, batch testing, and reward function evaluation.</p>
  </div>
</div>

---

## Popular Use Cases

<div class="use-cases-grid">
  <div class="use-case-card">
    <h3>∑ Math & Science</h3>
    <p>Train models to solve equations, prove theorems, and explain scientific concepts with step-by-step reasoning.</p>
    <a href="use-cases.html#math-science">Learn more →</a>
  </div>

  <div class="use-case-card">
    <h3>‹/› Code Generation</h3>
    <p>Create coding assistants that generate clean, efficient code with proper documentation and error handling.</p>
    <a href="use-cases.html#coding">Learn more →</a>
  </div>

  <div class="use-case-card">
    <h3>? Question Answering</h3>
    <p>Build specialized Q&A systems for domains like medicine, law, or customer support with accurate, relevant answers.</p>
    <a href="use-cases.html#qa">Learn more →</a>
  </div>

  <div class="use-case-card">
    <h3>✎ Custom Tasks</h3>
    <p>Fine-tune for any task with custom reward functions: summarization, translation, creative writing, and more.</p>
    <a href="use-cases.html#custom">Learn more →</a>
  </div>
</div>

---

## What You Can Do

- **Fine-tune for specific tasks**: Math reasoning, code generation, question answering, instruction-following
- **Use curated datasets**: 7 pre-configured datasets including Alpaca, GSM8K, OpenMath, Code Alpaca, Dolly 15k, Orca Math, and SQuAD
- **Upload custom data**: Train on your own JSON, JSONL, CSV, or Parquet datasets
- **Monitor in real-time**: Track loss, rewards, KL divergence, and other metrics through WebSocket-based updates
- **Test and compare models**: Interactive evaluation interface with side-by-side comparisons and batch testing
- **Export anywhere**: Convert models to GGUF format for llama.cpp, Ollama, or LM Studio
- **Save configurations**: Reuse training setups for reproducibility across experiments

---

## Powered by GRPO

LoRA Craft uses **Group Relative Policy Optimization (GRPO)**, a reinforcement learning algorithm that teaches models to maximize rewards rather than just imitate examples.

**How it works:**
1. Model generates multiple responses for each prompt
2. Reward function scores each response based on your criteria
3. Model learns to prefer high-reward responses
4. Training continues until the model consistently produces quality outputs

This approach enables models to learn complex behaviors and improve beyond the quality of training data.

[Learn more about GRPO →](features.html#grpo)

---

## Ready to Get Started?

<div class="final-cta">
  <h3>Start fine-tuning your first model in minutes</h3>
  <p>Choose Docker for the easiest setup with zero configuration, or install natively for maximum control. Both methods support full GPU acceleration.</p>
  <a href="quickstart.html" class="btn-primary">Quick Start Guide</a>
  <a href="documentation.html" class="btn-secondary">Full Documentation</a>
</div>

---

## Open Source & Community-Driven

LoRA Craft is MIT licensed and built with:
- [Unsloth](https://github.com/unslothai/unsloth) - Optimized training framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - Model library
- [Qwen](https://github.com/QwenLM/Qwen) - High-performance language models ([Documentation](https://qwen.readthedocs.io/))
- [Flask](https://flask.palletsprojects.com/) - Web framework
