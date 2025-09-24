#!/usr/bin/env python3
"""
GRPO Fine-Tuner - Main Entry Point

A GUI application for GRPO (Group Relative Policy Optimization) fine-tuning
of Qwen and LLaMA models with full dataset/prompt customization.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import json
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings in production
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'peft': 'PEFT',
        'accelerate': 'Accelerate',
        'tkinter': 'Tkinter',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'numpy': 'NumPy'
    }

    missing = []
    for package, name in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing.append(f"{name} ({package})")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def setup_environment():
    """Setup application environment."""
    # Create necessary directories
    directories = ['configs', 'logs', 'outputs', 'checkpoints', 'exports', 'cache']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)

    # Setup logging
    from utils.logging_config import setup_logging
    setup_logging(log_level="INFO", log_dir="logs")

    # Set environment variables for better performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # CUDA optimization
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        import torch
        if torch.cuda.is_available():
            # Use first available GPU by default
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run_gui_mode(config_file: str = None):
    """Run application in GUI mode.

    Args:
        config_file: Optional configuration file to load
    """
    import tkinter as tk
    from gui.app import GRPOFineTunerApp

    # Create root window
    root = tk.Tk()

    # Create application
    app = GRPOFineTunerApp(root)

    # Load configuration if provided
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            app.config = config
            app._apply_config_to_ui()
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")

    icon_image = tk.PhotoImage(file='assets/icon.png')
    root.iconphoto(True, icon_image)
    
    # Start application
    root.mainloop()


def run_headless_mode(config_file: str):
    """Run application in headless mode (no GUI).

    Args:
        config_file: Configuration file path
    """
    from core import (
        GRPOTrainer,
        GRPOConfig,
        DatasetHandler,
        DatasetConfig,
        PromptTemplate,
        TemplateConfig,
        CustomRewardBuilder,
        SystemConfig
    )

    # Load configuration
    if not Path(config_file).exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    print("Starting GRPO training in headless mode...")

    # System configuration
    system_config = SystemConfig()
    print(system_config.get_system_summary())

    # Create GRPO configuration
    grpo_config = GRPOConfig(
        model_name=config.get('model_name', 'qwen2.5-0.5b'),
        num_train_epochs=config.get('num_epochs', 3),
        per_device_train_batch_size=config.get('batch_size', 4),
        learning_rate=config.get('learning_rate', 2e-4),
        lora_r=config.get('lora_rank', 16),
        lora_alpha=config.get('lora_alpha', 16),
        output_dir=config.get('output_dir', './outputs'),
    )

    # Create trainer
    trainer = GRPOTrainer(grpo_config, system_config)

    # Setup model
    print("Loading model...")
    trainer.setup_model()

    # Load dataset
    print("Loading dataset...")
    dataset_config = DatasetConfig(
        source_type=config.get('dataset_source', 'huggingface'),
        source_path=config.get('dataset_path', 'nvidia/OpenMathReasoning'),
        split=config.get('dataset_split', 'train'),
        instruction_field=config.get('instruction_field', 'instruction'),
        response_field=config.get('response_field', 'response'),
    )

    dataset_handler = DatasetHandler(dataset_config)
    dataset = dataset_handler.load()
    print(f"Loaded {len(dataset)} samples")

    # Setup prompt template
    template_config = TemplateConfig(
        name="training",
        description="Training template",
        instruction_template="{instruction}",
        response_template="{response}",
        reasoning_start_marker=config.get('reasoning_start', '[THINKING]'),
        reasoning_end_marker=config.get('reasoning_end', '[/THINKING]'),
    )
    template = PromptTemplate(template_config)

    # Setup reward function
    reward_builder = CustomRewardBuilder()
    reward_builder.add_binary_reward("exact_match", weight=1.0)

    # Start training
    print("Starting training...")
    try:
        # Pre-fine-tuning phase
        if config.get('pre_finetune', True):
            print("Running pre-fine-tuning phase...")
            trainer.pre_fine_tune(dataset, template, epochs=1)

        # GRPO training
        print("Running GRPO training...")
        metrics = trainer.grpo_train(dataset, template, reward_builder)

        print("Training completed!")
        print(f"Final metrics: {metrics}")

        # Export model
        if config.get('export_model', True):
            export_path = config.get('export_path', './exports/model')
            export_format = config.get('export_format', 'safetensors')
            print(f"Exporting model to {export_path} in {export_format} format...")
            trainer.export_model(export_path, format=export_format)

    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
    finally:
        trainer.cleanup()


def create_example_config():
    """Create an example configuration file."""
    example_config = {
        "model_name": "qwen2.5-0.5b",
        "dataset_source": "huggingface",
        "dataset_path": "tatsu-lab/alpaca",
        "dataset_split": "train",
        "instruction_field": "instruction",
        "response_field": "output",
        "template_name": "default",
        "reasoning_start": "[THINKING]",
        "reasoning_end": "[/THINKING]",
        "lora_rank": 16,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "num_epochs": 3,
        "temperature": 0.7,
        "top_p": 0.95,
        "num_generations": 4,
        "kl_penalty": 0.01,
        "clip_range": 0.2,
        "use_flash_attention": False,
        "gradient_checkpointing": False,
        "mixed_precision": True,
        "output_dir": "./outputs",
        "export_model": True,
        "export_path": "./exports/model",
        "export_format": "safetensors",
        "pre_finetune": True
    }

    config_path = Path("configs/example.json")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(example_config, f, indent=2)

    print(f"Created example configuration at {config_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GRPO Fine-Tuner - Train Qwen and LLaMA models with GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path (JSON format)'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no GUI)'
    )

    parser.add_argument(
        '--create-example-config',
        action='store_true',
        help='Create an example configuration file'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='GRPO Fine-Tuner v1.0.0'
    )

    args = parser.parse_args()

    # Set debug mode
    if args.debug:
        os.environ['GRPO_DEBUG'] = '1'
        logging.getLogger().setLevel(logging.DEBUG)

    # Create example config
    if args.create_example_config:
        create_example_config()
        return

    # Check dependencies
    check_dependencies()

    # Setup environment
    setup_environment()

    # Run application
    if args.headless:
        if not args.config:
            print("Error: Configuration file required in headless mode")
            print("Use --config <file> to specify configuration")
            sys.exit(1)
        run_headless_mode(args.config)
    else:
        run_gui_mode(args.config)


if __name__ == "__main__":
    main()
