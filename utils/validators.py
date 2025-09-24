"""Input validators and sanitizers for the application."""

import re
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import torch


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class Validators:
    """Collection of validators for different input types."""

    @staticmethod
    def validate_model_name(model_name: str, allow_local: bool = True) -> Tuple[bool, str]:
        """Validate model name or path.

        Args:
            model_name: Model name or path
            allow_local: Whether to allow local paths

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not model_name or not model_name.strip():
            return False, "Model name cannot be empty"

        model_name = model_name.strip()

        # Check for HuggingFace model format (org/model)
        hf_pattern = r'^[\w\-\.]+/[\w\-\.]+$'
        if re.match(hf_pattern, model_name):
            return True, ""

        # Check for local path
        if allow_local:
            path = Path(model_name)
            if path.exists() and path.is_dir():
                # Check if it looks like a model directory
                required_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
                if any((path / f).exists() for f in required_files):
                    return True, ""
                return False, "Directory exists but doesn't appear to contain a model"

        # Check against known model patterns
        known_patterns = [
            r'^qwen2\.5-\d+(\.\d+)?b(-instruct)?$',
            r'^llama-3\.\d+-\d+b(-instruct)?$',
            r'^mistral-\d+b(-instruct)?$',
            r'^phi-\d+(-mini)?$',
            r'^gemma-\d+b(-it)?$',
        ]

        for pattern in known_patterns:
            if re.match(pattern, model_name.lower()):
                return True, ""

        return False, f"Invalid model name format: {model_name}"

    @staticmethod
    def validate_dataset_name(dataset_name: str) -> Tuple[bool, str]:
        """Validate HuggingFace dataset name.

        Args:
            dataset_name: Dataset name

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not dataset_name or not dataset_name.strip():
            return False, "Dataset name cannot be empty"

        dataset_name = dataset_name.strip()

        # HuggingFace dataset format
        hf_pattern = r'^[\w\-\.]+(/[\w\-\.]+)?$'
        if re.match(hf_pattern, dataset_name):
            return True, ""

        return False, f"Invalid dataset name format: {dataset_name}"

    @staticmethod
    def validate_file_path(file_path: str,
                          must_exist: bool = True,
                          extensions: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Validate file path.

        Args:
            file_path: File path to validate
            must_exist: Whether file must exist
            extensions: Allowed file extensions

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path or not file_path.strip():
            return False, "File path cannot be empty"

        path = Path(file_path.strip())

        if must_exist and not path.exists():
            return False, f"File does not exist: {file_path}"

        if path.exists() and not path.is_file():
            return False, f"Path is not a file: {file_path}"

        if extensions:
            ext = path.suffix.lower()
            if ext not in extensions:
                return False, f"Invalid file extension. Expected one of {extensions}, got {ext}"

        # Check for path traversal attempts
        try:
            resolved = path.resolve()
            if ".." in str(path):
                return False, "Path traversal detected"
        except Exception as e:
            return False, f"Invalid path: {e}"

        return True, ""

    @staticmethod
    def validate_directory_path(dir_path: str,
                               must_exist: bool = False,
                               create_if_missing: bool = False) -> Tuple[bool, str]:
        """Validate directory path.

        Args:
            dir_path: Directory path to validate
            must_exist: Whether directory must exist
            create_if_missing: Create directory if it doesn't exist

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not dir_path or not dir_path.strip():
            return False, "Directory path cannot be empty"

        path = Path(dir_path.strip())

        if must_exist and not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    return True, ""
                except Exception as e:
                    return False, f"Failed to create directory: {e}"
            return False, f"Directory does not exist: {dir_path}"

        if path.exists() and not path.is_dir():
            return False, f"Path is not a directory: {dir_path}"

        return True, ""

    @staticmethod
    def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Validate URL.

        Args:
            url: URL to validate
            allowed_schemes: Allowed URL schemes (default: ['http', 'https'])

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url or not url.strip():
            return False, "URL cannot be empty"

        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']

        try:
            result = urlparse(url.strip())
            if not result.scheme:
                return False, "URL must include scheme (e.g., https://)"

            if result.scheme not in allowed_schemes:
                return False, f"URL scheme must be one of {allowed_schemes}"

            if not result.netloc:
                return False, "URL must include domain"

            return True, ""
        except Exception as e:
            return False, f"Invalid URL: {e}"

    @staticmethod
    def validate_json(json_str: str) -> Tuple[bool, str, Optional[Dict]]:
        """Validate JSON string.

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_json)
        """
        if not json_str or not json_str.strip():
            return False, "JSON string cannot be empty", None

        try:
            parsed = json.loads(json_str.strip())
            return True, "", parsed
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", None

    @staticmethod
    def validate_yaml(yaml_str: str) -> Tuple[bool, str, Optional[Dict]]:
        """Validate YAML string.

        Args:
            yaml_str: YAML string to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_yaml)
        """
        if not yaml_str or not yaml_str.strip():
            return False, "YAML string cannot be empty", None

        try:
            parsed = yaml.safe_load(yaml_str.strip())
            return True, "", parsed
        except yaml.YAMLError as e:
            return False, f"Invalid YAML: {e}", None

    @staticmethod
    def validate_numeric(value: Any,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None,
                        allow_float: bool = True,
                        allow_negative: bool = True) -> Tuple[bool, str, Optional[float]]:
        """Validate numeric value.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_float: Whether to allow float values
            allow_negative: Whether to allow negative values

        Returns:
            Tuple of (is_valid, error_message, parsed_value)
        """
        try:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return False, "Value cannot be empty", None

            if allow_float:
                num_val = float(value)
            else:
                num_val = int(value)
                if isinstance(value, float) and value != num_val:
                    return False, "Value must be an integer", None

            if not allow_negative and num_val < 0:
                return False, "Value cannot be negative", None

            if min_val is not None and num_val < min_val:
                return False, f"Value must be at least {min_val}", None

            if max_val is not None and num_val > max_val:
                return False, f"Value must be at most {max_val}", None

            return True, "", num_val
        except (ValueError, TypeError) as e:
            return False, f"Invalid numeric value: {e}", None

    @staticmethod
    def validate_batch_size(batch_size: Any) -> Tuple[bool, str, Optional[int]]:
        """Validate batch size.

        Args:
            batch_size: Batch size to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_value)
        """
        return Validators.validate_numeric(
            batch_size,
            min_val=1,
            max_val=512,
            allow_float=False,
            allow_negative=False
        )

    @staticmethod
    def validate_learning_rate(lr: Any) -> Tuple[bool, str, Optional[float]]:
        """Validate learning rate.

        Args:
            lr: Learning rate to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_value)
        """
        return Validators.validate_numeric(
            lr,
            min_val=1e-10,
            max_val=1.0,
            allow_float=True,
            allow_negative=False
        )

    @staticmethod
    def validate_sequence_length(seq_len: Any) -> Tuple[bool, str, Optional[int]]:
        """Validate sequence length.

        Args:
            seq_len: Sequence length to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_value)
        """
        valid, msg, value = Validators.validate_numeric(
            seq_len,
            min_val=32,
            max_val=32768,
            allow_float=False,
            allow_negative=False
        )

        if valid:
            # Check if power of 2 or common value
            common_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
            if value not in common_lengths:
                return True, f"Warning: {value} is not a common sequence length", value

        return valid, msg, value

    @staticmethod
    def validate_lora_rank(rank: Any) -> Tuple[bool, str, Optional[int]]:
        """Validate LoRA rank.

        Args:
            rank: LoRA rank to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_value)
        """
        valid, msg, value = Validators.validate_numeric(
            rank,
            min_val=1,
            max_val=512,
            allow_float=False,
            allow_negative=False
        )

        if valid:
            # Check if power of 2
            if value & (value - 1) != 0:
                # Not a power of 2, but still valid
                pass

        return valid, msg, value

    @staticmethod
    def validate_template(template: str) -> Tuple[bool, str]:
        """Validate prompt template.

        Args:
            template: Template string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not template or not template.strip():
            return False, "Template cannot be empty"

        template = template.strip()

        # Check for required placeholders
        required_placeholders = ['{instruction}', '{response}']
        missing = [p for p in required_placeholders if p not in template]

        if missing:
            return False, f"Template missing required placeholders: {missing}"

        # Check for balanced braces
        open_braces = template.count('{')
        close_braces = template.count('}')

        if open_braces != close_braces:
            return False, "Template has unbalanced braces"

        # Check for valid placeholder format
        placeholder_pattern = r'\{[a-zA-Z_][a-zA-Z0-9_]*\}'
        placeholders = re.findall(r'\{[^}]+\}', template)

        for p in placeholders:
            if not re.match(placeholder_pattern, p):
                return False, f"Invalid placeholder format: {p}"

        return True, ""

    @staticmethod
    def validate_python_code(code: str, safe_mode: bool = True) -> Tuple[bool, str]:
        """Validate Python code.

        Args:
            code: Python code to validate
            safe_mode: Check for potentially dangerous operations

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Code cannot be empty"

        code = code.strip()

        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        if safe_mode:
            # Check for dangerous operations
            dangerous_patterns = [
                r'\b(exec|eval|__import__|compile|open|file)\s*\(',
                r'\bos\.(system|popen|exec)',
                r'\bsubprocess\.',
                r'\b(input|raw_input)\s*\(',
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code):
                    return False, f"Code contains potentially dangerous operation: {pattern}"

        return True, ""


class Sanitizers:
    """Collection of sanitizers for different input types."""

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 255) -> str:
        """Sanitize filename for safe file operations.

        Args:
            filename: Filename to sanitize
            max_length: Maximum length of filename

        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)

        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')

        # Limit length
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            name = name[:max_length - len(ext)]
            filename = name + ext

        # Ensure not empty
        if not filename:
            filename = "unnamed"

        return filename

    @staticmethod
    def sanitize_path(path: str) -> str:
        """Sanitize file path.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path
        """
        # Resolve path and remove any traversal attempts
        try:
            clean_path = Path(path).resolve()
            return str(clean_path)
        except Exception:
            # If path resolution fails, do basic sanitization
            path = path.replace('..', '')
            path = path.replace('~', '')
            return path

    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize text for HTML display.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#39;',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def sanitize_json_key(key: str) -> str:
        """Sanitize JSON key.

        Args:
            key: Key to sanitize

        Returns:
            Sanitized key
        """
        # Remove invalid characters for JSON keys
        key = re.sub(r'[^\w\-\.]', '_', key)

        # Ensure doesn't start with number
        if key and key[0].isdigit():
            key = '_' + key

        return key or 'unnamed'


def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate complete training configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Validate model
    if 'model_name' in config:
        valid, msg = Validators.validate_model_name(config['model_name'])
        if not valid:
            errors.append(f"Model: {msg}")

    # Validate dataset - REQUIRED for training
    dataset_configured = False
    dataset_source = config.get('dataset_source', '')

    # Check for HuggingFace dataset
    if 'HuggingFace' in dataset_source or dataset_source == 'huggingface_hub':
        # Check both possible keys: dataset_name and dataset_path
        dataset_value = config.get('dataset_name') or config.get('dataset_path')
        if dataset_value and dataset_value.strip():
            valid, msg = Validators.validate_dataset_name(dataset_value)
            if not valid:
                errors.append(f"Dataset: {msg}")
            else:
                dataset_configured = True
        else:
            errors.append("HuggingFace dataset not specified")

    # Check for local dataset file
    elif 'Local' in dataset_source or dataset_source == 'local_file':
        # Check both possible keys: dataset_file and dataset_path
        dataset_value = config.get('dataset_file') or config.get('dataset_path')
        if dataset_value and dataset_value.strip():
            valid, msg = Validators.validate_file_path(
                dataset_value,
                must_exist=True,
                extensions=['.json', '.jsonl', '.csv', '.parquet']
            )
            if not valid:
                errors.append(f"Dataset file: {msg}")
            else:
                dataset_configured = True
        else:
            errors.append("Local dataset file not specified")

    # Check for API endpoint
    elif 'API' in dataset_source:
        dataset_value = config.get('dataset_path')
        if dataset_value and dataset_value.strip():
            dataset_configured = True
        else:
            errors.append("API endpoint not specified")

    # Check for direct input/custom dataset
    elif 'Direct' in dataset_source or 'Custom' in dataset_source:
        # For direct input, we don't need a dataset path
        dataset_configured = True

    # Check if dataset_path exists regardless of source (catch-all)
    elif config.get('dataset_path') and config.get('dataset_path').strip():
        dataset_configured = True

    # No dataset configured at all
    if not dataset_configured and not config.get('dataset_path'):
        errors.append("No dataset configured. Please select a dataset source and provide dataset details.")

    # Validate numeric parameters
    numeric_params = {
        'batch_size': Validators.validate_batch_size,
        'learning_rate': Validators.validate_learning_rate,
        'max_sequence_length': Validators.validate_sequence_length,
        'lora_rank': Validators.validate_lora_rank,
    }

    for param, validator in numeric_params.items():
        if param in config:
            valid, msg, _ = validator(config[param])
            if not valid:
                errors.append(f"{param}: {msg}")

    # Validate paths
    if 'output_dir' in config:
        valid, msg = Validators.validate_directory_path(
            config['output_dir'],
            create_if_missing=True
        )
        if not valid:
            errors.append(f"Output directory: {msg}")

    return len(errors) == 0, errors


if __name__ == "__main__":
    # Test validators
    print("Testing validators...")

    # Test model name
    valid, msg = Validators.validate_model_name("qwen2.5-3b-instruct")
    print(f"Model 'qwen2.5-3b-instruct': {valid} - {msg}")

    # Test file path
    valid, msg = Validators.validate_file_path("test.json", must_exist=False, extensions=['.json'])
    print(f"File 'test.json': {valid} - {msg}")

    # Test numeric
    valid, msg, val = Validators.validate_batch_size("32")
    print(f"Batch size '32': {valid} - {msg} - {val}")

    # Test sanitizers
    print("\nTesting sanitizers...")
    print(f"Filename: {Sanitizers.sanitize_filename('test<>file:name.txt')}")
    print(f"Path: {Sanitizers.sanitize_path('/path/../to/file')}")
