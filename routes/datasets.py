"""
Dataset routes for dataset management and operations.

This module provides endpoints for:
- HuggingFace dataset downloading and caching
- Local dataset file uploads
- Dataset field detection and validation
- Dataset sampling and preview
- Cache management
"""

import os
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from services import DatasetService
from core import DatasetHandler
from constants import POPULAR_DATASETS
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create blueprint
datasets_bp = Blueprint('datasets', __name__, url_prefix='/api/datasets')

# Initialize service
dataset_service = DatasetService()


@datasets_bp.route('/list', methods=['GET'])
def list_popular_datasets():
    """Get list of popular datasets with their cache status."""
    try:
        # Check cache status for each dataset
        handler = DatasetHandler()

        datasets_with_status = []
        for dataset_path, info in POPULAR_DATASETS.items():
            # Extract dataset_config if it exists (for multi-config datasets like GSM8K)
            dataset_config = info.get('dataset_config')

            # Check cache status with config to differentiate between different configs
            is_cached = handler.is_cached(dataset_path, dataset_config)
            cache_info = handler.get_cache_info(dataset_path, dataset_config) if is_cached else None

            dataset_entry = {
                'path': dataset_path,
                'name': info['name'],
                'size': info['size'],
                'category': info['category'],
                'is_cached': is_cached,
                'cache_info': cache_info.to_dict() if cache_info else None
            }

            # Include dataset_config in response if present
            if dataset_config:
                dataset_entry['dataset_config'] = dataset_config

            datasets_with_status.append(dataset_entry)

        return jsonify({
            'datasets': datasets_with_status
        })

    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/status/<path:dataset_name>', methods=['GET'])
def get_dataset_status(dataset_name):
    """Check if a dataset is cached and get its info."""
    try:
        from core import DatasetConfig

        # Get optional dataset_config from query params
        dataset_config = request.args.get('dataset_config')

        # Replace forward slash with safe separator
        safe_name = dataset_name.replace('/', '__')

        # Create temporary handler to check cache
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name.replace('__', '/'),
            dataset_config=dataset_config,
            use_cache=True
        )

        handler = DatasetHandler(config)
        normalized_name = dataset_name.replace('__', '/')
        is_cached = handler.is_cached(normalized_name, dataset_config)

        cache_info = None
        if is_cached:
            cache_info = handler.get_cache_info(normalized_name, dataset_config)

        return jsonify({
            'dataset': dataset_name,
            'is_cached': is_cached,
            'cache_info': cache_info.to_dict() if cache_info else None
        })

    except Exception as e:
        logger.error(f"Failed to get dataset status: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/download', methods=['POST'])
def download_dataset():
    """Download a dataset from HuggingFace and cache it."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        dataset_config = data.get('dataset_config')
        force_download = data.get('force_download', False)
        custom_field_mapping = data.get('custom_field_mapping')

        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        # Use the service to download
        success, message, info = dataset_service.download_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            force_download=force_download,
            custom_field_mapping=custom_field_mapping
        )

        if success:
            return jsonify({
                'success': True,
                'message': message,
                'dataset_info': info
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 500

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/sample', methods=['POST'])
def sample_dataset():
    """Get a sample of dataset entries."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        dataset_config = data.get('dataset_config')
        sample_size = data.get('sample_size', 5)

        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        result = dataset_service.sample_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            sample_size=sample_size
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to sample dataset: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/detect-fields', methods=['POST'])
def detect_fields():
    """Detect dataset fields and suggest mappings."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        dataset_config = data.get('dataset_config')
        is_local = data.get('is_local', False)

        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        result = dataset_service.detect_fields(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            is_local=is_local
        )

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to detect fields: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/cache/info', methods=['GET'])
def get_cache_info():
    """Get information about dataset cache."""
    try:
        result = dataset_service.get_cache_info()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the dataset cache."""
    try:
        result = dataset_service.clear_cache()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({'error': str(e)}), 500


# Upload routes

@datasets_bp.route('/upload', methods=['POST'])
def upload_dataset():
    """Upload a dataset file (JSON, JSONL, CSV, or Parquet)."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get allowed extensions from config
        allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'json', 'jsonl', 'csv', 'parquet'})

        # Validate file extension
        if not dataset_service.validate_file_extension(file.filename, allowed_extensions):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            }), 400

        # Generate secure filename with timestamp to avoid conflicts
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = original_filename.rsplit('.', 1)[0]
        extension = original_filename.rsplit('.', 1)[1]
        filename = f"{base_name}_{timestamp}.{extension}"

        # Save file
        upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # Get file info using service
        try:
            dataset_info = dataset_service.get_upload_info(filename)

            return jsonify({
                'success': True,
                'message': 'Dataset uploaded successfully',
                'filename': filename,
                'filepath': filepath,
                'dataset_info': dataset_info
            })

        except Exception as info_error:
            logger.warning(f"Could not get upload info: {info_error}")
            # Still return success since file was uploaded
            return jsonify({
                'success': True,
                'message': 'Dataset uploaded successfully',
                'filename': filename,
                'filepath': filepath,
                'dataset_info': {
                    'file_size': os.path.getsize(filepath),
                    'error': str(info_error)
                }
            })

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/uploaded', methods=['GET'])
def list_uploaded_datasets():
    """List all uploaded datasets."""
    try:
        result = dataset_service.list_uploaded_datasets()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to list uploaded datasets: {e}")
        return jsonify({'error': str(e)}), 500


@datasets_bp.route('/uploaded/<filename>', methods=['DELETE'])
def delete_uploaded_dataset(filename):
    """Delete an uploaded dataset file."""
    try:
        success, message = dataset_service.delete_uploaded_dataset(filename)

        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'error': message
            }), 404

    except Exception as e:
        logger.error(f"Failed to delete uploaded dataset: {e}")
        return jsonify({'error': str(e)}), 500
