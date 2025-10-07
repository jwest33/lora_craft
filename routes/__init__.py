"""
Routes package for Flask blueprint organization.

This module exports all route blueprints for the application:
- system: System information and status
- configs: Configuration management
- rewards: Reward function management
- exports: Model export operations
- templates: Prompt and chat template management
- datasets: Dataset management and operations
- training: Training session management
- models: Model testing and comparison
"""

from .system import system_bp
from .configs import configs_bp
from .rewards import rewards_bp
from .exports import exports_bp
from .templates import templates_bp
from .datasets import datasets_bp
from .training import training_bp
from .models import models_bp

__all__ = [
    'system_bp',
    'configs_bp',
    'rewards_bp',
    'exports_bp',
    'templates_bp',
    'datasets_bp',
    'training_bp',
    'models_bp',
]
