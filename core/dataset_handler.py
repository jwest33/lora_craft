"""Dataset handler for multiple data sources."""

import json
import csv
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
import requests
from io import StringIO, BytesIO
import numpy as np
import time
import shutil
from datetime import datetime
import hashlib
import pickle

from utils.validators import Validators
from utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    source_type: str  # 'huggingface', 'local', 'api', 'direct'
    source_path: str  # Path, name, or URL
    split: str = 'train'
    subset: Optional[str] = None
    streaming: bool = False
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42

    # Field mapping
    instruction_field: str = 'instruction'
    response_field: str = 'response'
    system_field: Optional[str] = None
    additional_fields: List[str] = field(default_factory=list)

    # Preprocessing
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    remove_empty: bool = True
    lowercase: bool = False
    strip_whitespace: bool = True

    # API specific
    api_headers: Dict[str, str] = field(default_factory=dict)
    api_params: Dict[str, Any] = field(default_factory=dict)

    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    force_download: bool = False
    sample_size: Optional[int] = None  # For preview/sampling


@dataclass
class DatasetStatistics:
    """Statistics for a loaded dataset."""
    total_samples: int
    avg_instruction_length: float
    avg_response_length: float
    min_instruction_length: int
    max_instruction_length: int
    min_response_length: int
    max_response_length: int
    empty_instructions: int
    empty_responses: int
    unique_instructions: int
    field_coverage: Dict[str, float]  # Percentage of non-empty values per field


@dataclass
class CacheInfo:
    """Information about cached dataset."""
    dataset_name: str
    cache_path: Path
    size_bytes: int
    download_date: datetime
    samples: int
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dataset_name': self.dataset_name,
            'cache_path': str(self.cache_path),
            'size_mb': round(self.size_bytes / (1024 * 1024), 2),
            'download_date': self.download_date.isoformat(),
            'samples': self.samples,
            'checksum': self.checksum
        }


class DatasetHandler:
    """Handles loading and processing of datasets from multiple sources."""

    SUPPORTED_FILE_EXTENSIONS = {
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.txt': 'text'
    }

    def __init__(self, config: Optional[DatasetConfig] = None, progress_callback: Optional[Callable] = None):
        """Initialize dataset handler.

        Args:
            config: Dataset configuration
            progress_callback: Callback function for progress updates
        """
        self.config = config or DatasetConfig(source_type='direct', source_path='')
        self.dataset = None
        self.statistics = None
        self._field_mapping = {}
        self.progress_callback = progress_callback
        self._cache_dir = Path(self.config.cache_dir or './cache/datasets')
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_info_file = self._cache_dir / 'cache_info.json'
        self._load_cache_info()

    def _load_cache_info(self):
        """Load cache information from file."""
        self.cache_info = {}
        if self._cache_info_file.exists():
            try:
                with open(self._cache_info_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        value['download_date'] = datetime.fromisoformat(value['download_date'])
                        value['cache_path'] = Path(value['cache_path'])
                        self.cache_info[key] = CacheInfo(**value)
            except Exception as e:
                logger.warning(f"Failed to load cache info: {e}")
                self.cache_info = {}

    def _save_cache_info(self):
        """Save cache information to file."""
        try:
            data = {key: info.to_dict() for key, info in self.cache_info.items()}
            with open(self._cache_info_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache info: {e}")

    def _get_cache_key(self, dataset_name: str) -> str:
        """Generate cache key for dataset."""
        return hashlib.md5(dataset_name.encode()).hexdigest()

    def is_cached(self, dataset_name: str) -> bool:
        """Check if dataset is cached."""
        cache_key = self._get_cache_key(dataset_name)
        cache_path = self._cache_dir / f"{cache_key}.pkl"
        return cache_path.exists() and cache_key in self.cache_info

    def get_cache_info(self, dataset_name: str) -> Optional[CacheInfo]:
        """Get cache information for dataset."""
        cache_key = self._get_cache_key(dataset_name)
        return self.cache_info.get(cache_key)

    def clear_cache(self, dataset_name: Optional[str] = None):
        """Clear cache for specific dataset or all cached datasets."""
        if dataset_name:
            # Clear specific dataset
            cache_key = self._get_cache_key(dataset_name)
            cache_path = self._cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for {dataset_name}")
            if cache_key in self.cache_info:
                del self.cache_info[cache_key]
                self._save_cache_info()
        else:
            # Clear all cache
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_info = {}
            self._save_cache_info()
            logger.info("Cleared all dataset cache")

    def get_total_cache_size(self) -> int:
        """Get total size of cached datasets in bytes."""
        total_size = 0
        for info in self.cache_info.values():
            if info.cache_path.exists():
                total_size += info.size_bytes
        return total_size

    def _report_progress(self, message: str, progress: float = None, status: str = "loading"):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback({
                'message': message,
                'progress': progress,
                'status': status,
                'timestamp': time.time()
            })

    def load(self, config: Optional[DatasetConfig] = None) -> Dataset:
        """Load dataset from configured source.

        Args:
            config: Optional config to override current config

        Returns:
            Loaded dataset
        """
        if config:
            self.config = config

        logger.info(f"Loading dataset from {self.config.source_type}: {self.config.source_path}")

        if self.config.source_type == 'huggingface':
            self.dataset = self._load_huggingface()
        elif self.config.source_type == 'local':
            self.dataset = self._load_local()
        elif self.config.source_type == 'api':
            self.dataset = self._load_api()
        elif self.config.source_type == 'direct':
            self.dataset = self._load_direct()
        else:
            raise ValueError(f"Unknown source type: {self.config.source_type}")

        # Apply preprocessing
        if self.dataset:
            self.dataset = self._preprocess_dataset(self.dataset)
            self.statistics = self._calculate_statistics(self.dataset)

        return self.dataset

    def _load_huggingface(self) -> Dataset:
        """Load dataset from HuggingFace Hub with caching support."""
        try:
            # Validate dataset name
            valid, msg = Validators.validate_dataset_name(self.config.source_path)
            if not valid:
                raise ValueError(msg)

            cache_key = self._get_cache_key(self.config.source_path)
            cache_path = self._cache_dir / f"{cache_key}.pkl"

            # Check cache unless force download
            if self.config.use_cache and not self.config.force_download and self.is_cached(self.config.source_path):
                self._report_progress(f"Loading {self.config.source_path} from cache...", 0.1, "cache_loading")

                try:
                    with open(cache_path, 'rb') as f:
                        dataset = pickle.load(f)

                    self._report_progress(f"Loaded from cache successfully", 1.0, "cached")
                    logger.info(f"Loaded dataset from cache: {self.config.source_path}")

                    # Apply sampling if requested
                    if self.config.sample_size and len(dataset) > self.config.sample_size:
                        dataset = dataset.select(range(self.config.sample_size))
                        logger.info(f"Sampled {self.config.sample_size} examples for preview")

                    return dataset
                except Exception as e:
                    logger.warning(f"Failed to load from cache, will download: {e}")
                    self.clear_cache(self.config.source_path)

            # Download dataset
            self._report_progress(f"Downloading {self.config.source_path}...", 0.2, "downloading")
            logger.info(f"Starting download of dataset: {self.config.source_path}")

            # Load dataset - this will download if not cached by HF
            kwargs = {
                'path': self.config.source_path,
                'split': self.config.split,
                'streaming': self.config.streaming,
                'trust_remote_code': False  # Security: don't trust remote code
            }

            if self.config.subset:
                kwargs['name'] = self.config.subset

            # For sampling, limit the split
            if self.config.sample_size and not self.config.streaming:
                # Modify split to limit samples
                split_str = f"{self.config.split}[:{self.config.sample_size}]"
                kwargs['split'] = split_str

            self._report_progress(f"Loading dataset...", 0.5, "loading")

            # Load the dataset (this blocks until download is complete when streaming=False)
            dataset = load_dataset(**kwargs)

            # Handle DatasetDict
            if isinstance(dataset, DatasetDict):
                if self.config.split in dataset:
                    dataset = dataset[self.config.split]
                else:
                    # Use first available split
                    available_splits = list(dataset.keys())
                    logger.warning(f"Split '{self.config.split}' not found. Available splits: {available_splits}")
                    dataset = dataset[available_splits[0]] if available_splits else None

            if dataset is None:
                raise ValueError("Failed to load dataset - no data returned")

            self._report_progress(f"Processing dataset...", 0.8, "processing")

            # Ensure dataset is fully loaded (not lazy)
            if not self.config.streaming:
                # Force dataset to materialize by accessing length
                dataset_len = len(dataset)
                logger.info(f"Dataset loaded successfully with {dataset_len} samples")

                # Save to cache if not sampling
                if self.config.use_cache and not self.config.sample_size:
                    try:
                        self._report_progress(f"Caching dataset...", 0.9, "caching")

                        with open(cache_path, 'wb') as f:
                            pickle.dump(dataset, f)

                        # Calculate size
                        size_bytes = cache_path.stat().st_size

                        # Store cache info
                        self.cache_info[cache_key] = CacheInfo(
                            dataset_name=self.config.source_path,
                            cache_path=cache_path,
                            size_bytes=size_bytes,
                            download_date=datetime.now(),
                            samples=dataset_len,
                            checksum=cache_key
                        )
                        self._save_cache_info()

                        logger.info(f"Cached dataset: {self.config.source_path} ({size_bytes / 1024 / 1024:.2f} MB)")
                    except Exception as e:
                        logger.warning(f"Failed to cache dataset: {e}")

                # Limit samples if specified (and not already limited by split)
                if self.config.max_samples and dataset_len > self.config.max_samples and not self.config.sample_size:
                    dataset = dataset.select(range(self.config.max_samples))
                    logger.info(f"Limited dataset to {self.config.max_samples} samples")

                # Shuffle if requested
                if self.config.shuffle and not self.config.sample_size:
                    dataset = dataset.shuffle(seed=self.config.seed)

            self._report_progress(f"Dataset ready!", 1.0, "completed")
            return dataset

        except Exception as e:
            self._report_progress(f"Failed: {str(e)}", 0, "error")
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise

    def _load_local(self) -> Dataset:
        """Load dataset from local file."""
        path = Path(self.config.source_path)

        # Validate file path
        valid, msg = Validators.validate_file_path(
            str(path),
            must_exist=True,
            extensions=list(self.SUPPORTED_FILE_EXTENSIONS.keys())
        )
        if not valid:
            raise ValueError(msg)

        file_type = self.SUPPORTED_FILE_EXTENSIONS.get(path.suffix.lower())

        if file_type == 'json':
            return self._load_json(path)
        elif file_type == 'jsonl':
            return self._load_jsonl(path)
        elif file_type == 'csv':
            return self._load_csv(path)
        elif file_type == 'parquet':
            return self._load_parquet(path)
        elif file_type == 'text':
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load_json(self, path: Path) -> Dataset:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            # Assume it's a dictionary of lists
            return Dataset.from_dict(data)
        else:
            raise ValueError("JSON file must contain a list or dictionary")

    def _load_jsonl(self, path: Path) -> Dataset:
        """Load JSONL file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        return Dataset.from_list(data)

    def _load_csv(self, path: Path) -> Dataset:
        """Load CSV file."""
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)

    def _load_parquet(self, path: Path) -> Dataset:
        """Load Parquet file."""
        table = pq.read_table(path)
        df = table.to_pandas()
        return Dataset.from_pandas(df)

    def _load_text(self, path: Path) -> Dataset:
        """Load text file (one sample per line)."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append({'text': line.strip()})

        return Dataset.from_list(data)

    def _load_api(self) -> Dataset:
        """Load dataset from API endpoint."""
        # Validate URL
        valid, msg = Validators.validate_url(self.config.source_path)
        if not valid:
            raise ValueError(msg)

        try:
            response = requests.get(
                self.config.source_path,
                headers=self.config.api_headers,
                params=self.config.api_params
            )
            response.raise_for_status()

            # Try to parse as JSON
            data = response.json()

            if isinstance(data, list):
                return Dataset.from_list(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    # Common API response format
                    return Dataset.from_list(data['data'])
                else:
                    return Dataset.from_dict(data)
            else:
                raise ValueError("API response must be JSON list or dictionary")

        except requests.RequestException as e:
            logger.error(f"Failed to load from API: {e}")
            raise

    def _load_direct(self) -> Dataset:
        """Load dataset from direct input (for testing)."""
        if not self.config.source_path:
            # Create empty dataset
            return Dataset.from_list([])

        # Try to parse as JSON
        try:
            data = json.loads(self.config.source_path)
            if isinstance(data, list):
                return Dataset.from_list(data)
            else:
                return Dataset.from_dict(data)
        except json.JSONDecodeError:
            # Treat as single text sample
            return Dataset.from_list([{'text': self.config.source_path}])

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing to dataset."""
        if not dataset or len(dataset) == 0:
            return dataset

        # Map fields if needed
        dataset = self._map_fields(dataset)

        # Filter by length
        if self.config.max_length or self.config.min_length:
            dataset = self._filter_by_length(dataset)

        # Remove empty samples
        if self.config.remove_empty:
            dataset = self._remove_empty(dataset)

        # Apply text transformations
        if self.config.lowercase or self.config.strip_whitespace:
            dataset = self._transform_text(dataset)

        # Limit samples
        if self.config.max_samples and not self.config.streaming:
            if len(dataset) > self.config.max_samples:
                dataset = dataset.select(range(self.config.max_samples))

        return dataset

    def _map_fields(self, dataset: Dataset) -> Dataset:
        """Map dataset fields to standard format."""
        columns = dataset.column_names if hasattr(dataset, 'column_names') else []

        # Auto-detect fields if not specified
        if self.config.instruction_field not in columns:
            # Try to find instruction field
            instruction_candidates = ['instruction', 'prompt', 'question', 'input', 'text']
            for candidate in instruction_candidates:
                if candidate in columns:
                    self._field_mapping[self.config.instruction_field] = candidate
                    break

        if self.config.response_field not in columns:
            # Try to find response field
            response_candidates = ['response', 'answer', 'output', 'completion', 'label']
            for candidate in response_candidates:
                if candidate in columns:
                    self._field_mapping[self.config.response_field] = candidate
                    break

        # Apply field mapping
        if self._field_mapping:
            def map_fields(example):
                mapped = {}
                for new_field, old_field in self._field_mapping.items():
                    if old_field in example:
                        mapped[new_field] = example[old_field]
                # Keep other fields
                for field in example:
                    if field not in self._field_mapping.values():
                        mapped[field] = example[field]
                return mapped

            dataset = dataset.map(map_fields)

        return dataset

    def _filter_by_length(self, dataset: Dataset) -> Dataset:
        """Filter dataset by text length."""
        def check_length(example):
            instruction_len = len(str(example.get(self.config.instruction_field, '')))
            response_len = len(str(example.get(self.config.response_field, '')))

            if self.config.min_length:
                if instruction_len < self.config.min_length or response_len < self.config.min_length:
                    return False

            if self.config.max_length:
                if instruction_len > self.config.max_length or response_len > self.config.max_length:
                    return False

            return True

        return dataset.filter(check_length)

    def _remove_empty(self, dataset: Dataset) -> Dataset:
        """Remove samples with empty fields."""
        def not_empty(example):
            instruction = str(example.get(self.config.instruction_field, '')).strip()
            response = str(example.get(self.config.response_field, '')).strip()
            return bool(instruction) and bool(response)

        return dataset.filter(not_empty)

    def _transform_text(self, dataset: Dataset) -> Dataset:
        """Apply text transformations."""
        def transform(example):
            if self.config.instruction_field in example:
                text = str(example[self.config.instruction_field])
                if self.config.strip_whitespace:
                    text = text.strip()
                if self.config.lowercase:
                    text = text.lower()
                example[self.config.instruction_field] = text

            if self.config.response_field in example:
                text = str(example[self.config.response_field])
                if self.config.strip_whitespace:
                    text = text.strip()
                if self.config.lowercase:
                    text = text.lower()
                example[self.config.response_field] = text

            return example

        return dataset.map(transform)

    def _calculate_statistics(self, dataset: Dataset) -> DatasetStatistics:
        """Calculate dataset statistics."""
        if not dataset or len(dataset) == 0:
            return DatasetStatistics(
                total_samples=0,
                avg_instruction_length=0,
                avg_response_length=0,
                min_instruction_length=0,
                max_instruction_length=0,
                min_response_length=0,
                max_response_length=0,
                empty_instructions=0,
                empty_responses=0,
                unique_instructions=0,
                field_coverage={}
            )

        # Extract fields
        instructions = []
        responses = []
        for example in dataset:
            instructions.append(str(example.get(self.config.instruction_field, '')))
            responses.append(str(example.get(self.config.response_field, '')))

        # Calculate lengths
        instruction_lengths = [len(i) for i in instructions]
        response_lengths = [len(r) for r in responses]

        # Count empties
        empty_instructions = sum(1 for i in instructions if not i.strip())
        empty_responses = sum(1 for r in responses if not r.strip())

        # Calculate field coverage
        field_coverage = {}
        if hasattr(dataset, 'column_names'):
            for field in dataset.column_names:
                non_empty = sum(1 for ex in dataset if ex.get(field, '') and str(ex[field]).strip())
                field_coverage[field] = (non_empty / len(dataset)) * 100

        return DatasetStatistics(
            total_samples=len(dataset),
            avg_instruction_length=np.mean(instruction_lengths) if instruction_lengths else 0,
            avg_response_length=np.mean(response_lengths) if response_lengths else 0,
            min_instruction_length=min(instruction_lengths) if instruction_lengths else 0,
            max_instruction_length=max(instruction_lengths) if instruction_lengths else 0,
            min_response_length=min(response_lengths) if response_lengths else 0,
            max_response_length=max(response_lengths) if response_lengths else 0,
            empty_instructions=empty_instructions,
            empty_responses=empty_responses,
            unique_instructions=len(set(instructions)),
            field_coverage=field_coverage
        )

    def get_preview(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Get preview of dataset samples.

        Args:
            num_samples: Number of samples to preview

        Returns:
            List of sample dictionaries
        """
        if not self.dataset:
            return []

        samples = []
        for i, example in enumerate(self.dataset):
            if i >= num_samples:
                break
            samples.append(dict(example))

        return samples

    def export(self, path: str, format: str = 'jsonl'):
        """Export dataset to file.

        Args:
            path: Output file path
            format: Export format ('json', 'jsonl', 'csv', 'parquet')
        """
        if not self.dataset:
            raise ValueError("No dataset loaded")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            self.dataset.to_json(path)
        elif format == 'jsonl':
            self.dataset.to_json(path, lines=True)
        elif format == 'csv':
            self.dataset.to_csv(path)
        elif format == 'parquet':
            self.dataset.to_parquet(path)
        else:
            raise ValueError(f"Unknown export format: {format}")

        logger.info(f"Dataset exported to {path}")

    def apply_template(self, template_func: callable) -> Dataset:
        """Apply template function to dataset.

        Args:
            template_func: Function to apply to each sample

        Returns:
            Transformed dataset
        """
        if not self.dataset:
            raise ValueError("No dataset loaded")

        self.dataset = self.dataset.map(template_func)
        return self.dataset

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train/validation/test sets.

        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set

        Returns:
            Tuple of (train, validation, test) datasets
        """
        if not self.dataset:
            raise ValueError("No dataset loaded")

        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be less than 1.0")

        total = len(self.dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        # Shuffle before splitting
        dataset = self.dataset.shuffle(seed=self.config.seed)

        train = dataset.select(range(train_size))
        val = dataset.select(range(train_size, train_size + val_size))
        test = dataset.select(range(train_size + val_size, total))

        return train, val, test


if __name__ == "__main__":
    # Test dataset handler
    config = DatasetConfig(
        source_type='direct',
        source_path=json.dumps([
            {"instruction": "What is 2+2?", "response": "4"},
            {"instruction": "What is the capital of France?", "response": "Paris"},
        ])
    )

    handler = DatasetHandler(config)
    dataset = handler.load()

    print(f"Loaded {len(dataset)} samples")
    print("Preview:")
    for sample in handler.get_preview(2):
        print(sample)

    if handler.statistics:
        print(f"\nStatistics:")
        print(f"  Total samples: {handler.statistics.total_samples}")
        print(f"  Avg instruction length: {handler.statistics.avg_instruction_length:.1f}")
        print(f"  Avg response length: {handler.statistics.avg_response_length:.1f}")
