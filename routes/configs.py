"""
Configuration routes for training config management.

This module provides endpoints for:
- Validating training configurations
- Saving and loading configurations
- Listing and deleting saved configurations
"""

import os
import json
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

from utils.logging_config import get_logger
from utils.validators import validate_training_config

logger = get_logger(__name__)

# Create blueprint
configs_bp = Blueprint('configs', __name__, url_prefix='/api')


def get_configs_dir():
    """Get the configs directory path."""
    return Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs'))


@configs_bp.route('/config/validate', methods=['POST'])
def validate_config():
    """Validate training configuration."""
    try:
        config = request.json
        valid, errors = validate_training_config(config)

        if valid:
            return jsonify({'valid': True, 'message': 'Configuration is valid'})
        else:
            return jsonify({'valid': False, 'errors': errors}), 400
    except Exception as e:
        logger.error(f"Error validating config: {e}")
        return jsonify({'error': str(e)}), 500


@configs_bp.route('/config/save', methods=['POST'])
def save_config():
    """Save configuration to file."""
    try:
        config = request.json
        filename = config.get('filename', f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        configs_dir = get_configs_dir()
        configs_dir.mkdir(exist_ok=True)
        filepath = configs_dir / filename

        logger.info(f"Saving config to: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved successfully: {filename}")

        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath),
            'message': f'Configuration saved as {filename}'
        })
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify({'error': str(e)}), 500


@configs_bp.route('/config/load/<filename>', methods=['GET'])
def load_config(filename):
    """Load configuration from file."""
    try:
        configs_dir = get_configs_dir()
        filepath = configs_dir / filename

        if not filepath.exists():
            return jsonify({'error': 'Configuration file not found'}), 404

        with open(filepath, 'r') as f:
            config = json.load(f)

        return jsonify(config)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({'error': str(e)}), 500


@configs_bp.route('/configs/list', methods=['GET'])
def list_configs():
    """List all saved configurations."""
    try:
        configs_dir = get_configs_dir()
        configs = []

        if configs_dir.exists():
            for config_file in configs_dir.glob('*.json'):
                # Get file info
                stat = config_file.stat()
                configs.append({
                    'name': config_file.stem,
                    'filename': config_file.name,
                    'modified': stat.st_mtime,
                    'size': stat.st_size
                })

        # Sort by modified time (newest first)
        configs.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify(configs)
    except Exception as e:
        logger.error(f"Error listing configs: {e}")
        return jsonify({'error': str(e)}), 500


@configs_bp.route('/configs/delete/<filename>', methods=['DELETE'])
def delete_config(filename):
    """Delete a configuration file."""
    try:
        configs_dir = get_configs_dir()
        filepath = configs_dir / filename

        if not filepath.exists():
            return jsonify({'error': 'Configuration file not found'}), 404

        filepath.unlink()

        return jsonify({'message': f'Configuration {filename} deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting config: {e}")
        return jsonify({'error': str(e)}), 500


# Additional endpoints for configuration management

@configs_bp.route('/configurations', methods=['GET'])
def get_configurations():
    """Get list of saved configurations with metadata."""
    try:
        configs_dir = Path('./configs')
        if not configs_dir.exists():
            configs_dir.mkdir(parents=True, exist_ok=True)

        configs = []
        for config_file in configs_dir.glob('*.json'):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    configs.append({
                        'id': config_file.stem,
                        'name': config_data.get('name', config_file.stem),
                        'description': config_data.get('description', ''),
                        'timestamp': config_data.get('timestamp', config_file.stat().st_mtime * 1000)
                    })
            except Exception as e:
                logger.warning(f"Failed to load config {config_file}: {e}")

        return jsonify({'configurations': configs})
    except Exception as e:
        logger.error(f"Error getting configurations: {e}")
        return jsonify({'configurations': []})


@configs_bp.route('/save_configuration', methods=['POST'])
def save_configuration():
    """Save a configuration with metadata."""
    try:
        data = request.json
        config_name = data.get('name', f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        configs_dir = Path('./configs')
        configs_dir.mkdir(parents=True, exist_ok=True)

        config_file = configs_dir / f"{config_name}.json"
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)

        return jsonify({'success': True, 'config_id': config_name})
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@configs_bp.route('/configuration/<config_id>', methods=['GET', 'DELETE'])
def manage_configuration(config_id):
    """Get or delete a configuration by ID."""
    try:
        config_file = Path('./configs') / f"{config_id}.json"

        if request.method == 'GET':
            if not config_file.exists():
                return jsonify({'error': 'Configuration not found', 'success': False}), 404

            with open(config_file, 'r') as f:
                config_data = json.load(f)

            return jsonify({'success': True, 'configuration': config_data})

        elif request.method == 'DELETE':
            if config_file.exists():
                config_file.unlink()
            return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Error managing configuration: {e}")
        return jsonify({'error': str(e), 'success': False}), 500
