"""Dataset handler for multiple data sources."""

import json
import csv
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple, Union
from dataclasses import dataclass, field
import logging
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
import requests
from io import StringIO, BytesIO
import numpy as np

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


class DatasetHandler:
    """Handles loading and processing of datasets from multiple sources."""

    SUPPORTED_FILE_EXTENSIONS = {
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.txt': 'text'
    }

    def __init__(self, config: Optional[DatasetConfig] = None):
        """Initialize dataset handler.

        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig(source_type='direct', source_path='')
        self.dataset = None
        self.statistics = None
        self._field_mapping = {}

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
        """Load dataset from HuggingFace Hub."""
        try:
            # Validate dataset name
            valid, msg = Validators.validate_dataset_name(self.config.source_path)
            if not valid:
                raise ValueError(msg)

            logger.info(f"Starting download of dataset: {self.config.source_path}")

            # Load dataset - this will download if not cached
            kwargs = {
                'path': self.config.source_path,
                'split': self.config.split,
                'streaming': self.config.streaming,
                'trust_remote_code': False  # Security: don't trust remote code
            }

            if self.config.subset:
                kwargs['name'] = self.config.subset

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

            # Ensure dataset is fully loaded (not lazy)
            if not self.config.streaming:
                # Force dataset to materialize by accessing length
                dataset_len = len(dataset)
                logger.info(f"Dataset loaded successfully with {dataset_len} samples")

                # Limit samples if specified
                if self.config.max_samples and dataset_len > self.config.max_samples:
                    dataset = dataset.select(range(self.config.max_samples))
                    logger.info(f"Limited dataset to {self.config.max_samples} samples")

                # Shuffle if requested
                if self.config.shuffle:
                    dataset = dataset.shuffle(seed=self.config.seed)

            return dataset

        except Exception as e:
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
