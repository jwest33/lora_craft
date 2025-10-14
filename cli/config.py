"""
Configuration management for LoRA Craft CLI.

Handles loading and saving CLI configuration from ~/.loracraft/config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class CLIConfig:
    """Manages CLI configuration."""

    def __init__(self):
        """Initialize configuration manager."""
        self.config_dir = Path.home() / '.loracraft'
        self.config_file = self.config_dir / 'config.yaml'
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return self._get_default_config()

        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            return {**self._get_default_config(), **config}
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_file}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'server_url': 'http://localhost:5001',
            'timeout': 30,
            'default_format': 'table',  # table, json, yaml
            'color': True,
            'verbose': False
        }

    def save(self):
        """Save current configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_file, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value

    @property
    def server_url(self) -> str:
        """Get server URL."""
        return self.config.get('server_url', 'http://localhost:5001')

    @server_url.setter
    def server_url(self, url: str):
        """Set server URL."""
        self.config['server_url'] = url

    @property
    def timeout(self) -> int:
        """Get request timeout."""
        return self.config.get('timeout', 30)

    @property
    def verbose(self) -> bool:
        """Get verbose mode."""
        return self.config.get('verbose', False)

    @verbose.setter
    def verbose(self, value: bool):
        """Set verbose mode."""
        self.config['verbose'] = value
