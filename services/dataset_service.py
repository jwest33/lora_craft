"""
Dataset service for dataset management operations.

This module provides business logic for:
- Dataset downloading with progress tracking
- Dataset caching and cache management
- Field detection and mapping suggestion
- Dataset sampling and preview
- Uploaded dataset management
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

from core import DatasetHandler, DatasetConfig
from utils.logging_config import get_logger
from constants import POPULAR_DATASETS

logger = get_logger(__name__)


class DatasetService:
    """Service for managing datasets."""

    def __init__(self, upload_folder: str = 'uploads', cache_folder: str = 'datasets_cache'):
        """
        Initialize dataset service.

        Args:
            upload_folder: Path to uploaded files directory
            cache_folder: Path to dataset cache directory
        """
        self.upload_folder = Path(upload_folder)
        self.cache_folder = Path(cache_folder)

        # Ensure directories exist
        self.upload_folder.mkdir(exist_ok=True)
        self.cache_folder.mkdir(exist_ok=True)

    def get_dataset_status(self, dataset_name: str) -> Dict[str, Any]:
        """
        Check if a dataset is cached and get its info.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with cache status and info
        """
        try:
            # Replace forward slash with safe separator for file paths
            safe_name = dataset_name.replace('/', '__')

            # Create temporary handler to check cache
            config = DatasetConfig(
                source_type='huggingface',
                source_path=dataset_name,
                use_cache=True
            )

            handler = DatasetHandler(config)
            cache_info = handler.get_cache_info(dataset_name)

            if cache_info:
                return {
                    'cached': True,
                    'info': cache_info.to_dict()
                }
            else:
                return {'cached': False}

        except Exception as e:
            logger.error(f"Error getting dataset status: {e}")
            raise

    def download_dataset(self, dataset_name: str, dataset_config: Optional[str],
                        force_download: bool, custom_field_mapping: Optional[Dict],
                        progress_callback: Optional[Callable] = None) -> Tuple[Any, Dict]:
        """
        Download a dataset with optional progress tracking.

        Args:
            dataset_name: Name of the dataset to download
            dataset_config: Optional config for multi-config datasets
            force_download: Force re-download even if cached
            custom_field_mapping: Optional custom field mapping
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (dataset, cache_info)

        Raises:
            ValueError: If dataset name is invalid
            Exception: If download fails
        """
        if not dataset_name:
            raise ValueError('Dataset name required')

        # Check if dataset has a custom default split or field mapping
        default_split = 'train'
        dataset_info = POPULAR_DATASETS.get(dataset_name, {})
        if 'default_split' in dataset_info:
            default_split = dataset_info['default_split']

        # Get field mappings - prefer custom, then predefined, then default
        if custom_field_mapping:
            instruction_field = custom_field_mapping.get('instruction', 'instruction')
            response_field = custom_field_mapping.get('response', 'response')
        else:
            field_mapping = dataset_info.get('field_mapping', {})
            instruction_field = field_mapping.get('instruction', 'instruction')
            response_field = field_mapping.get('response', 'response')

        # Create dataset config
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            split=default_split,
            subset=dataset_config,  # This maps to 'name' parameter in HF load_dataset
            instruction_field=instruction_field,
            response_field=response_field,
            use_cache=True,
            force_download=force_download
        )

        # Load dataset
        handler = DatasetHandler(config, progress_callback=progress_callback)
        dataset = handler.load()

        # Get cache info
        cache_info = handler.get_cache_info(dataset_name)

        return dataset, cache_info

    def download_dataset_background(self, dataset_name: str, dataset_config: Optional[str],
                                   force_download: bool, custom_field_mapping: Optional[Dict],
                                   session_id: str, emit_callback: Callable):
        """
        Download dataset in background thread with WebSocket updates.

        Args:
            dataset_name: Name of the dataset
            dataset_config: Optional config for multi-config datasets
            force_download: Force re-download
            custom_field_mapping: Optional custom field mapping
            session_id: Session ID for progress tracking
            emit_callback: Callback to emit WebSocket events
        """
        def progress_callback(info):
            emit_callback(session_id, 'dataset_progress', info)

        def download_task():
            try:
                dataset, cache_info = self.download_dataset(
                    dataset_name, dataset_config, force_download,
                    custom_field_mapping, progress_callback
                )

                emit_callback(session_id, 'dataset_complete', {
                    'dataset_name': dataset_name,
                    'samples': len(dataset) if dataset else 0,
                    'cache_info': cache_info.to_dict() if cache_info else None
                })
            except Exception as e:
                logger.error(f"Dataset download failed: {e}")
                emit_callback(session_id, 'dataset_error', {'error': str(e)})

        thread = threading.Thread(target=download_task, daemon=True)
        thread.start()

    def sample_dataset(self, dataset_name: str, dataset_config: Optional[str],
                      sample_size: int = 5) -> Dict[str, Any]:
        """
        Get a sample of a dataset for preview.

        Args:
            dataset_name: Name of the dataset
            dataset_config: Optional config for multi-config datasets
            sample_size: Number of samples to return

        Returns:
            Dictionary with sample data

        Raises:
            ValueError: If dataset name is invalid
        """
        if not dataset_name:
            raise ValueError('Dataset name required')

        # Check for custom default split
        default_split = 'train'
        dataset_info = POPULAR_DATASETS.get(dataset_name, {})
        if 'default_split' in dataset_info:
            default_split = dataset_info['default_split']

        # Get field mappings
        field_mapping = dataset_info.get('field_mapping', {})
        instruction_field = field_mapping.get('instruction', 'instruction')
        response_field = field_mapping.get('response', 'response')

        # Create config for sampling
        config = DatasetConfig(
            source_type='huggingface',
            source_path=dataset_name,
            split=default_split,
            subset=dataset_config,
            instruction_field=instruction_field,
            response_field=response_field,
            max_samples=sample_size,
            use_cache=True
        )

        # Load sample
        handler = DatasetHandler(config)
        dataset = handler.load()

        # Extract samples
        samples = []
        for i in range(min(sample_size, len(dataset))):
            samples.append(dataset[i])

        return {
            'samples': samples,
            'total_shown': len(samples),
            'instruction_field': instruction_field,
            'response_field': response_field
        }

    def detect_fields(self, dataset_name: str, dataset_config: Optional[str] = None,
                     is_local: bool = False, upload_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect available fields in a dataset for mapping.

        Args:
            dataset_name: Name or path of the dataset
            dataset_config: Optional config for multi-config datasets
            is_local: Whether this is a local uploaded file
            upload_folder: Upload folder path if needed

        Returns:
            Dictionary with columns, suggested mappings, and sample data

        Raises:
            ValueError: If dataset name is invalid
            FileNotFoundError: If local file not found
        """
        if not dataset_name:
            raise ValueError('Dataset name required')

        # Check if this is an uploaded file
        if is_local or dataset_name.startswith(str(self.upload_folder)) or '\\uploads\\' in dataset_name or '/uploads/' in dataset_name:
            return self._detect_fields_local(dataset_name)

        # Check for predefined mapping
        dataset_info = POPULAR_DATASETS.get(dataset_name, {})
        default_split = dataset_info.get('default_split', 'train')

        try:
            from datasets import load_dataset

            # Special handling for known datasets
            if 'squad' in dataset_name.lower():
                dataset_name = 'squad'
                default_split = 'train'

            kwargs = {
                'path': dataset_name,
                'split': f"{default_split}[:1]",
                'streaming': False,
                'trust_remote_code': False
            }

            if dataset_config:
                kwargs['name'] = dataset_config

            # Try to load
            try:
                dataset = load_dataset(**kwargs)
            except Exception:
                # Retry without slice notation
                kwargs['split'] = default_split
                dataset = load_dataset(**kwargs)
                if hasattr(dataset, 'select'):
                    dataset = dataset.select(range(1))

            # Get column names
            columns = []
            if hasattr(dataset, 'column_names'):
                columns = dataset.column_names
            elif hasattr(dataset, 'features'):
                columns = list(dataset.features.keys())

            # Suggest mappings
            suggested_mappings = self._suggest_field_mappings(columns)

            # Get sample data
            sample_data = {}
            if len(dataset) > 0:
                sample = dataset[0]
                for col in columns[:5]:
                    if col in sample:
                        value = str(sample[col])[:100]
                        sample_data[col] = value

            return {
                'columns': columns,
                'suggested_mappings': suggested_mappings,
                'sample_data': sample_data,
                'predefined_mapping': dataset_info.get('field_mapping', {})
            }

        except Exception as e:
            logger.error(f"Failed to detect fields for {dataset_name}: {e}")
            raise

    def _detect_fields_local(self, filepath: str) -> Dict[str, Any]:
        """
        Detect fields in a local uploaded file.

        Args:
            filepath: Path to the local file

        Returns:
            Dictionary with columns, suggested mappings, and sample data
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f'File not found: {filepath}')

        columns = []
        sample_data = []
        extension = filepath.suffix[1:].lower()

        try:
            if extension == 'csv':
                import pandas as pd
                df = pd.read_csv(filepath, nrows=5)
                columns = list(df.columns)
                sample_data = df.head(3).to_dict('records')

            elif extension == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data_json = json.load(f)
                    if isinstance(data_json, list) and len(data_json) > 0:
                        columns = list(data_json[0].keys())
                        sample_data = data_json[:3]

            elif extension == 'jsonl':
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_lines = []
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        first_lines.append(json.loads(line))
                    if first_lines:
                        columns = list(first_lines[0].keys())
                        sample_data = first_lines

            elif extension == 'parquet':
                import pandas as pd
                df = pd.read_parquet(filepath, engine='auto')
                columns = list(df.columns)
                sample_data = df.head(3).to_dict('records')

            suggested_mappings = self._suggest_field_mappings(columns)

            return {
                'columns': columns,
                'suggested_mappings': suggested_mappings,
                'sample_data': sample_data,
                'is_local': True
            }

        except Exception as e:
            logger.error(f"Failed to read local file {filepath}: {e}")
            raise

    def _suggest_field_mappings(self, columns: List[str]) -> Dict[str, Optional[str]]:
        """
        Suggest field mappings based on column names.

        Args:
            columns: List of column names

        Returns:
            Dictionary with suggested instruction and response field names
        """
        suggested_mappings = {'instruction': None, 'response': None}

        instruction_patterns = ['instruction', 'prompt', 'question', 'input', 'text', 'problem', 'query', 'user']
        response_patterns = ['response', 'answer', 'output', 'completion', 'label', 'generated_solution', 'solution', 'reply', 'assistant']

        for col in columns:
            col_lower = col.lower()
            if not suggested_mappings['instruction']:
                for pattern in instruction_patterns:
                    if pattern in col_lower:
                        suggested_mappings['instruction'] = col
                        break
            if not suggested_mappings['response']:
                for pattern in response_patterns:
                    if pattern in col_lower:
                        suggested_mappings['response'] = col
                        break

        return suggested_mappings

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset cache.

        Returns:
            Dictionary with cache size and dataset count
        """
        try:
            total_size = 0
            dataset_count = 0

            if self.cache_folder.exists():
                for item in self.cache_folder.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
                    elif item.is_dir():
                        dataset_count += 1

            return {
                'cache_size_bytes': total_size,
                'cache_size_mb': round(total_size / (1024 * 1024), 2),
                'dataset_count': dataset_count,
                'cache_path': str(self.cache_folder)
            }
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            raise

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the dataset cache.

        Returns:
            Dictionary with results of cache clearing

        Raises:
            Exception: If clearing fails
        """
        try:
            import shutil

            if self.cache_folder.exists():
                # Get size before clearing
                total_size = sum(f.stat().st_size for f in self.cache_folder.rglob('*') if f.is_file())

                # Clear cache
                shutil.rmtree(self.cache_folder)
                self.cache_folder.mkdir(exist_ok=True)

                logger.info(f"Cleared dataset cache: {total_size} bytes")

                return {
                    'success': True,
                    'bytes_cleared': total_size,
                    'mb_cleared': round(total_size / (1024 * 1024), 2)
                }
            else:
                return {
                    'success': True,
                    'bytes_cleared': 0,
                    'message': 'Cache directory did not exist'
                }

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise

    def list_uploaded_datasets(self) -> Dict[str, Any]:
        """
        List all uploaded dataset files.

        Returns:
            Dictionary with list of uploaded files and metadata
        """
        try:
            from datetime import datetime
            from constants import ALLOWED_EXTENSIONS

            files = []

            if self.upload_folder.exists():
                for file_path in self.upload_folder.iterdir():
                    # Only include files with allowed extensions
                    if file_path.is_file() and file_path.suffix[1:].lower() in ALLOWED_EXTENSIONS:
                        stat = file_path.stat()
                        files.append({
                            'filename': file_path.name,
                            'filepath': str(file_path),
                            'relative_path': f"uploads/{file_path.name}",
                            'size_mb': round(stat.st_size / (1024 * 1024), 2),
                            'uploaded_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'extension': file_path.suffix[1:].lower()
                        })

            # Sort by upload time (most recent first)
            files.sort(key=lambda x: x['uploaded_at'], reverse=True)

            return {
                'files': files,
                'count': len(files)
            }

        except Exception as e:
            logger.error(f"Failed to list uploaded datasets: {e}")
            raise

    def delete_uploaded_dataset(self, filename: str) -> bool:
        """
        Delete an uploaded dataset file.

        Args:
            filename: Name of the file to delete

        Returns:
            True if deleted successfully, False if not found

        Raises:
            ValueError: If trying to access file outside upload directory
        """
        try:
            file_path = self.upload_folder / filename

            # Security check: ensure file is within upload directory
            if not str(file_path.resolve()).startswith(str(self.upload_folder.resolve())):
                raise ValueError('Invalid file path')

            if not file_path.exists():
                return False

            file_path.unlink()
            logger.info(f"Deleted uploaded dataset: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete uploaded dataset {filename}: {e}")
            raise

    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set) -> bool:
        """
        Check if a file extension is allowed.

        Args:
            filename: Name of the file
            allowed_extensions: Set of allowed extensions

        Returns:
            True if file extension is allowed
        """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    def get_upload_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about an uploaded dataset file.

        Args:
            file_path: Path to the uploaded file

        Returns:
            Dictionary with file info and preview

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

        # Get basic file info
        stat = file_path.stat()
        info = {
            'name': file_path.name,
            'path': str(file_path),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': file_path.suffix[1:].lower()
        }

        # Try to get preview
        try:
            fields_info = self._detect_fields_local(str(file_path))
            info.update({
                'columns': fields_info.get('columns', []),
                'suggested_mappings': fields_info.get('suggested_mappings', {}),
                'preview': fields_info.get('sample_data', [])
            })
        except Exception as e:
            logger.warning(f"Could not get preview for {file_path}: {e}")
            info['preview_error'] = str(e)

        return info
