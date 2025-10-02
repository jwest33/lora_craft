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
    """Configuration for dataset loading and processing.

    Note on max_samples:
        max_samples represents the number of samples to use PER EPOCH, not total samples
        across all epochs. For example, if max_samples=1000 and num_epochs=3, the trainer
        will use 1000 samples per epoch for a total of 3000 training steps (assuming batch_size=1).

        This matches the behavior of TRL's GRPOTrainer, where the dataset is cycled through
        for each epoch. If max_samples is None, all available samples will be used per epoch.
    """
    source_type: str  # 'huggingface', 'local', 'api', 'direct'
    source_path: str  # Path, name, or URL
    split: str = 'train'
    subset: Optional[str] = None
    streaming: bool = False
    max_samples: Optional[int] = None  # Samples per epoch (NOT total across all epochs)
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
                        if isinstance(value, dict):
                            value['download_date'] = datetime.fromisoformat(value['download_date'])
                            value['cache_path'] = Path(value['cache_path'])
                            # Convert size_mb back to size_bytes if needed (for backward compatibility)
                            if 'size_mb' in value and 'size_bytes' not in value:
                                value['size_bytes'] = int(value['size_mb'] * 1024 * 1024)
                                del value['size_mb']
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
            if self.progress_callback:
                self.progress_callback({
                    'message': 'Preprocessing dataset...',
                    'progress': 0.95,
                    'status': 'preprocessing',
                    'timestamp': time.time()
                })

            self.dataset = self._preprocess_dataset(self.dataset)

            # Skip statistics calculation - not needed for training and slows down large datasets
            # Statistics are cosmetic (only used for UI display)
            self.statistics = None
            logger.info("Skipped statistics calculation to speed up dataset loading")

            if self.progress_callback:
                self.progress_callback({
                    'message': 'Dataset ready!',
                    'progress': 1.0,
                    'status': 'completed',
                    'timestamp': time.time()
                })

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
            self._report_progress(f"Preparing to download {self.config.source_path}...", 0.2, "downloading")
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

            self._report_progress(f"Downloading dataset from HuggingFace...", 0.3, "downloading")
            self._report_progress(f"This may take a few moments for large datasets...", 0.4, "downloading")

            # Try to load the dataset, handling multi-config datasets
            try:
                self._report_progress(f"Loading dataset configuration...", 0.5, "loading")
                # Load the dataset (this blocks until download is complete when streaming=False)
                dataset = load_dataset(**kwargs)
                self._report_progress(f"Dataset loaded successfully", 0.7, "processing")
            except ValueError as e:
                # Check if it's a split error or config error
                error_msg = str(e)
                if "Unknown split" in error_msg:
                    # Extract available splits from error message
                    import re
                    splits_match = re.search(r"Should be one of \[(.+?)\]", error_msg)
                    if splits_match:
                        available_splits = [s.strip().strip("'") for s in splits_match.group(1).split(',')]
                        logger.warning(f"Split '{self.config.split}' not found. Available splits: {available_splits}")
                        self._report_progress(f"Using '{available_splits[0]}' split instead of '{self.config.split}'...", 0.55, "loading")

                        # Use first available split
                        kwargs['split'] = available_splits[0]
                        dataset = load_dataset(**kwargs)
                        self._report_progress(f"Dataset loaded with '{available_splits[0]}' split", 0.7, "processing")
                    else:
                        raise
                elif "Config name is missing" in error_msg:
                    logger.warning(f"Dataset requires config, trying with 'all' config: {error_msg}")
                    self._report_progress(f"Dataset requires config selection...", 0.55, "loading")

                    # Extract available configs from error message
                    import re
                    configs_match = re.search(r"available configs: \[(.*?)\]", error_msg)
                    if configs_match:
                        available_configs = [c.strip().strip("'") for c in configs_match.group(1).split(',')]
                        # Try with 'all' first, then 'default', then first available
                        for config in ['all', 'default'] + available_configs:
                            if config in available_configs:
                                logger.info(f"Auto-selecting config: {config}")
                                self._report_progress(f"Using '{config}' config...", 0.6, "loading")
                                kwargs['name'] = config
                                dataset = load_dataset(**kwargs)
                                self._report_progress(f"Dataset loaded with '{config}' config", 0.7, "processing")
                                break
                    else:
                        # Fallback to 'all' config
                        kwargs['name'] = 'all'
                        self._report_progress(f"Using default 'all' config...", 0.6, "loading")
                        dataset = load_dataset(**kwargs)
                        self._report_progress(f"Dataset loaded successfully", 0.7, "processing")
                else:
                    # Re-raise if it's not a config error
                    raise

            # Handle DatasetDict
            if isinstance(dataset, DatasetDict):
                if self.config.split in dataset:
                    dataset = dataset[self.config.split]
                else:
                    # Use first available split
                    available_splits = list(dataset.keys())
                    logger.warning(f"Split '{self.config.split}' not found. Available splits: {available_splits}")
                    if available_splits:
                        # If 'train' was requested but not found, try common alternatives
                        if self.config.split == 'train' and 'cot' in available_splits:
                            # Special case for OpenMathReasoning and similar datasets
                            dataset = dataset['cot']
                            logger.info(f"Using 'cot' split as alternative to 'train'")
                        else:
                            dataset = dataset[available_splits[0]]
                            logger.info(f"Using '{available_splits[0]}' split")
                    else:
                        dataset = None

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

                # max_samples now represents "samples per epoch"
                # TRL will handle repeating the dataset for multiple epochs
                # We only limit if we want fewer samples per epoch than available
                if self.config.max_samples and dataset_len > self.config.max_samples:
                    dataset = dataset.select(range(self.config.max_samples))
                    logger.info(f"Limited dataset to {self.config.max_samples} samples per epoch")

                # Shuffle if requested
                if self.config.shuffle and not self.config.sample_size:
                    dataset = dataset.shuffle(seed=self.config.seed)

            self._report_progress(f"Dataset loaded, processing...", 0.9, "loaded")
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
        """Load CSV file with efficient pandas preprocessing."""
        logger.info(f"Reading CSV file: {path}")

        # Read CSV with Python engine for better handling of messy/malformed files
        # The C engine is faster but stricter about format
        df = pd.read_csv(
            path,
            engine='python',  # Python engine handles messy CSVs better
            dtype=str,  # Don't infer types - much faster
            skipinitialspace=True,  # Ignore spaces after delimiter
            on_bad_lines='skip',  # Skip malformed lines instead of crashing
            encoding='utf-8',  # Explicit encoding
        )

        logger.info(f"CSV loaded: {len(df):,} rows, {len(df.columns)} columns")

        # Sanity check - if we only got 1 row from what should be a large file, something is wrong
        if len(df) < 10:
            logger.warning(f"Only {len(df)} rows loaded from CSV. File may be malformed.")
            logger.warning("Checking file structure...")

            # Try to debug by reading first few lines raw
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline() for _ in range(5)]
                    logger.warning(f"First 5 lines of file:")
                    for i, line in enumerate(first_lines):
                        logger.warning(f"  Line {i}: {repr(line[:100])}")  # Show first 100 chars with escape codes
            except Exception as e:
                logger.error(f"Could not read file for debugging: {e}")

        # Apply preprocessing in pandas (vectorized - much faster than row-by-row)
        df = self._preprocess_dataframe(df)

        # Mark as preprocessed to skip slow HuggingFace Dataset operations
        self._csv_preprocessed = True

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

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to pandas DataFrame (vectorized - much faster)."""
        if df.empty:
            return df

        original_size = len(df)
        logger.info(f"Preprocessing DataFrame with {original_size:,} rows and {len(df.columns)} columns")

        # 1. Rename columns if needed (instant operation)
        instruction_field = self.config.instruction_field
        response_field = self.config.response_field

        if instruction_field not in df.columns or response_field not in df.columns:
            logger.info(f"Field mapping needed. Looking for {instruction_field} and {response_field} in {df.columns.tolist()}")

            # Try to auto-map fields
            columns_lower = {col.lower(): col for col in df.columns}

            if instruction_field.lower() in columns_lower:
                actual_instruction = columns_lower[instruction_field.lower()]
                if actual_instruction != instruction_field:
                    df = df.rename(columns={actual_instruction: instruction_field})
                    logger.info(f"Mapped '{actual_instruction}' to '{instruction_field}'")

            if response_field.lower() in columns_lower:
                actual_response = columns_lower[response_field.lower()]
                if actual_response != response_field:
                    df = df.rename(columns={actual_response: response_field})
                    logger.info(f"Mapped '{actual_response}' to '{response_field}'")

        # 2. Remove empty samples (vectorized boolean indexing)
        if self.config.remove_empty:
            if instruction_field in df.columns and response_field in df.columns:
                before_filter = len(df)
                df = df[
                    (df[instruction_field].astype(str).str.strip() != '') &
                    (df[response_field].astype(str).str.strip() != '')
                ]
                removed = before_filter - len(df)
                if removed > 0:
                    logger.info(f"Removed {removed:,} empty samples ({before_filter:,} → {len(df):,})")

        # 3. Filter by length (vectorized)
        if self.config.max_length or self.config.min_length:
            if instruction_field in df.columns and response_field in df.columns:
                before_filter = len(df)
                mask = pd.Series([True] * len(df))

                instruction_lengths = df[instruction_field].astype(str).str.len()
                response_lengths = df[response_field].astype(str).str.len()

                if self.config.min_length:
                    mask &= (instruction_lengths >= self.config.min_length) & (response_lengths >= self.config.min_length)

                if self.config.max_length:
                    mask &= (instruction_lengths <= self.config.max_length) & (response_lengths <= self.config.max_length)

                df = df[mask]
                removed = before_filter - len(df)
                if removed > 0:
                    logger.info(f"Filtered {removed:,} samples by length ({before_filter:,} → {len(df):,})")

        # 4. Text transformations (vectorized)
        if self.config.lowercase or self.config.strip_whitespace:
            if instruction_field in df.columns:
                if self.config.strip_whitespace:
                    df[instruction_field] = df[instruction_field].astype(str).str.strip()
                if self.config.lowercase:
                    df[instruction_field] = df[instruction_field].astype(str).str.lower()

            if response_field in df.columns:
                if self.config.strip_whitespace:
                    df[response_field] = df[response_field].astype(str).str.strip()
                if self.config.lowercase:
                    df[response_field] = df[response_field].astype(str).str.lower()

            logger.info("Applied text transformations")

        # 5. Limit samples (if needed)
        if self.config.max_samples and len(df) > self.config.max_samples:
            df = df.head(self.config.max_samples)
            logger.info(f"Limited dataset from {original_size:,} to {self.config.max_samples:,} samples")

        final_size = len(df)
        if final_size != original_size:
            logger.info(f"Preprocessing complete: {original_size:,} → {final_size:,} samples ({final_size/original_size*100:.1f}% retained)")

        return df

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing to dataset (for non-CSV sources)."""
        if not dataset or len(dataset) == 0:
            return dataset

        # Check if this is from CSV (already preprocessed in pandas)
        if hasattr(self, '_csv_preprocessed') and self._csv_preprocessed:
            logger.info("Skipping dataset preprocessing (already done in pandas)")
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
                original_size = len(dataset)
                dataset = dataset.select(range(self.config.max_samples))
                logger.info(f"Limited dataset from {original_size} to {self.config.max_samples} samples (in preprocessing)")

        return dataset

    def _map_fields(self, dataset: Dataset) -> Dataset:
        """Map dataset fields to standard format."""
        columns = dataset.column_names if hasattr(dataset, 'column_names') else []

        # If the configured fields exist in the dataset, use them directly
        # Otherwise, try to auto-detect

        # For instruction field
        if self.config.instruction_field in columns:
            # Field exists, use it directly (no mapping needed)
            # Store the actual field name for filtering operations
            self._actual_instruction_field = self.config.instruction_field
        elif self.config.instruction_field != 'instruction':
            # User specified a custom field name, map it to 'instruction'
            if self.config.instruction_field in columns:
                self._field_mapping['instruction'] = self.config.instruction_field
                self._actual_instruction_field = 'instruction'  # After mapping
            else:
                logger.warning(f"Configured instruction field '{self.config.instruction_field}' not found in columns: {columns[:10]}")
                # Try auto-detection
                instruction_candidates = ['instruction', 'prompt', 'question', 'input', 'text', 'problem']
                for candidate in instruction_candidates:
                    if candidate in columns:
                        self._field_mapping['instruction'] = candidate
                        self._actual_instruction_field = 'instruction'
                        break
        else:
            # Default 'instruction' field not found, try auto-detect
            instruction_candidates = ['prompt', 'question', 'input', 'text', 'problem']
            for candidate in instruction_candidates:
                if candidate in columns:
                    self._field_mapping['instruction'] = candidate
                    self._actual_instruction_field = 'instruction'
                    break
            else:
                # No mapping found, keep original
                self._actual_instruction_field = self.config.instruction_field

        # For response field
        if self.config.response_field in columns:
            # Field exists, use it directly (no mapping needed)
            self._actual_response_field = self.config.response_field
        elif self.config.response_field != 'response':
            # User specified a custom field name, map it to 'response'
            if self.config.response_field in columns:
                self._field_mapping['response'] = self.config.response_field
                self._actual_response_field = 'response'  # After mapping
            else:
                logger.warning(f"Configured response field '{self.config.response_field}' not found in columns: {columns[:10]}")
                # Try auto-detection
                response_candidates = ['response', 'answer', 'output', 'completion', 'label', 'generated_solution', 'solution']
                for candidate in response_candidates:
                    if candidate in columns:
                        self._field_mapping['response'] = candidate
                        self._actual_response_field = 'response'
                        break
        else:
            # Default 'response' field not found, try auto-detect
            response_candidates = ['answer', 'output', 'completion', 'label', 'generated_solution', 'solution']
            for candidate in response_candidates:
                if candidate in columns:
                    self._field_mapping['response'] = candidate
                    self._actual_response_field = 'response'
                    break
            else:
                # No mapping found, keep original
                self._actual_response_field = self.config.response_field

        # Special handling for SQuAD dataset
        if 'squad' in self.config.source_path.lower() and 'answers' in columns:
            def process_squad(example):
                # SQuAD has answers as a dict with 'text' and 'answer_start' lists
                if 'answers' in example and isinstance(example['answers'], dict):
                    # Extract first answer text if available
                    if 'text' in example['answers'] and example['answers']['text']:
                        answer_text = example['answers']['text'][0] if isinstance(example['answers']['text'], list) else example['answers']['text']
                    else:
                        answer_text = ""
                    example['answers'] = answer_text
                return example

            dataset = dataset.map(process_squad)
            logger.info("Applied SQuAD-specific processing for answers field")

        # Apply field mapping if needed
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
        # After field mapping, we should have 'instruction' and 'response' fields
        # If mapping occurred, use those; otherwise use the configured fields
        if hasattr(self, '_field_mapping') and self._field_mapping:
            # Field mapping was applied, so use the mapped field names
            instruction_field = 'instruction' if 'instruction' in self._field_mapping else self.config.instruction_field
            response_field = 'response' if 'response' in self._field_mapping else self.config.response_field
        else:
            # No mapping, use configured fields
            instruction_field = self.config.instruction_field
            response_field = self.config.response_field

        # Check if the fields exist at all
        if instruction_field not in dataset.column_names or response_field not in dataset.column_names:
            logger.warning(f"Fields not found for filtering. Looking for {instruction_field} and {response_field} in {dataset.column_names[:10]}")
            # If fields don't exist, don't filter
            return dataset

        total_samples = len(dataset)

        # Report start of filtering
        if self.progress_callback:
            self.progress_callback({
                'message': f'Filtering empty samples from {total_samples:,} records...',
                'progress': 0,
                'status': 'filtering',
                'timestamp': time.time()
            })

        def not_empty(example, idx):
            # idx is passed as an integer when with_indices=True
            # Report progress periodically
            if self.progress_callback and idx % 10000 == 0:
                progress = idx / total_samples
                self.progress_callback({
                    'message': f'Filtering: {idx:,}/{total_samples:,} samples processed',
                    'progress': progress,
                    'status': 'filtering',
                    'timestamp': time.time()
                })

            instruction = str(example.get(instruction_field, '')).strip()
            response = str(example.get(response_field, '')).strip()
            return bool(instruction) and bool(response)

        # Use with_indices to track progress
        return dataset.filter(not_empty, with_indices=True)

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
        """Calculate dataset statistics using efficient vectorized operations."""
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

        # Convert to pandas for efficient vectorized operations
        df = dataset.to_pandas()

        # Get column names, handling missing fields gracefully
        instruction_field = self.config.instruction_field
        response_field = self.config.response_field

        # Check if fields exist in dataframe
        if instruction_field not in df.columns:
            logger.warning(f"Instruction field '{instruction_field}' not found in dataset. Available: {df.columns.tolist()}")
            instruction_field = df.columns[0] if len(df.columns) > 0 else None

        if response_field not in df.columns:
            logger.warning(f"Response field '{response_field}' not found in dataset. Available: {df.columns.tolist()}")
            response_field = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        if instruction_field is None or response_field is None:
            logger.error("Could not determine instruction/response fields")
            return DatasetStatistics(
                total_samples=len(df),
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

        # Vectorized operations - much faster than iteration
        instruction_col = df[instruction_field].astype(str)
        response_col = df[response_field].astype(str)

        # Calculate lengths (vectorized)
        instruction_lengths = instruction_col.str.len()
        response_lengths = response_col.str.len()

        # Count empties (vectorized)
        empty_instructions = (instruction_col.str.strip() == '').sum()
        empty_responses = (response_col.str.strip() == '').sum()

        # Field coverage (vectorized)
        field_coverage = {}
        for col in df.columns:
            non_empty = (df[col].astype(str).str.strip() != '').sum()
            field_coverage[col] = (non_empty / len(df)) * 100

        return DatasetStatistics(
            total_samples=len(df),
            avg_instruction_length=float(instruction_lengths.mean()) if len(instruction_lengths) > 0 else 0,
            avg_response_length=float(response_lengths.mean()) if len(response_lengths) > 0 else 0,
            min_instruction_length=int(instruction_lengths.min()) if len(instruction_lengths) > 0 else 0,
            max_instruction_length=int(instruction_lengths.max()) if len(instruction_lengths) > 0 else 0,
            min_response_length=int(response_lengths.min()) if len(response_lengths) > 0 else 0,
            max_response_length=int(response_lengths.max()) if len(response_lengths) > 0 else 0,
            empty_instructions=int(empty_instructions),
            empty_responses=int(empty_responses),
            unique_instructions=int(instruction_col.nunique()),
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
