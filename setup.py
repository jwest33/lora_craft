"""
Setup script for LoRA Craft CLI

Install the CLI tool with:
    pip install -e .

This will make the 'loracraft' command available globally.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "cli" / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="loracraft-cli",
    version="0.1.0",
    description="Command-line interface for LoRA Craft GRPO training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LoRA Craft Team",
    author_email="",
    url="https://github.com/jwest33/lora_craft",
    packages=find_packages(include=["cli", "cli.*"]),
    install_requires=[
        "click>=8.1.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "rich>=13.7.0",
        "websocket-client>=1.7.0",
        "python-socketio[client]>=5.11.0",
    ],
    entry_points={
        "console_scripts": [
            "loracraft=cli.main:main",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="lora fine-tuning grpo machine-learning nlp cli",
)
