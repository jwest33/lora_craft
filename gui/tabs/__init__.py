"""GUI tabs for the application."""

from .dataset_tab import DatasetTab
from .model_training_tab import ModelTrainingTab  # Combined tab
from .grpo_tab import GRPOTab  # Keep for backward compatibility
from .system_tab import SystemTab
from .monitoring_tab import MonitoringTab
from .export_tab import ExportTab

__all__ = [
    'DatasetTab',
    'ModelTrainingTab',  # New combined tab
    'ModelTab',  # Legacy
    'GRPOTab',  # Legacy
    'SystemTab',
    'MonitoringTab',
    'ExportTab'
]
