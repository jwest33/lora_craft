"""Model export module for exporting trained models to various formats."""

import os
import shutil
import json
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import safetensors.torch

from .gguf_converter import GGUFConverter

logger = logging.getLogger(__name__)


class ModelExporter:
    """Handle export of trained models to various formats."""

    SUPPORTED_FORMATS = {
        'safetensors': 'SafeTensors format (efficient, safe serialization)',
        'huggingface': 'HuggingFace format (standard transformers format)',
        'gguf': 'GGUF format (for llama.cpp and local inference)',
        'merged': 'Merged model (LoRA weights merged with base model)',
        'onnx': 'ONNX format (cross-platform deployment)'
    }

    GGUF_QUANTIZATIONS = {
        'f16': 'No quantization (largest, best quality)',
        'q8_0': '8-bit quantization (very good quality)',
        'q6_k': '6-bit quantization (good quality)',
        'q5_k_m': '5-bit quantization (balanced)',
        'q4_k_m': '4-bit quantization (recommended)',
        'q4_0': '4-bit quantization (smaller)',
        'q3_k_m': '3-bit quantization (smallest)'
    }

    def __init__(self, export_dir: str = "./exports"):
        """Initialize the ModelExporter.

        Args:
            export_dir: Base directory for exports
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.gguf_converter = GGUFConverter()

    def export_model(
        self,
        model_path: str,
        session_id: str,
        export_format: str,
        export_name: Optional[str] = None,
        quantization: Optional[str] = None,
        merge_lora: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Export a trained model to specified format.

        Args:
            model_path: Path to the trained model checkpoint
            session_id: Training session ID
            export_format: Format to export to
            export_name: Optional custom name for export
            quantization: Quantization level for GGUF format
            merge_lora: Whether to merge LoRA weights with base model
            progress_callback: Callback for progress updates (message, progress_pct)

        Returns:
            Tuple of (success, export_path, metadata)
        """
        if export_format not in self.SUPPORTED_FORMATS:
            return False, "", {"error": f"Unsupported format: {export_format}"}

        # Generate export name and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = export_name or f"model_{timestamp}"
        export_path = self.export_dir / session_id / export_format / export_name
        export_path.mkdir(parents=True, exist_ok=True)

        try:
            if progress_callback:
                progress_callback(f"Starting {export_format} export...", 0)

            # Load model and tokenizer
            logger.info(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Check if this is a LoRA model
            is_lora = (Path(model_path) / "adapter_config.json").exists()

            if export_format == 'safetensors':
                success, path = self._export_safetensors(
                    model_path, export_path, tokenizer, is_lora, progress_callback
                )
            elif export_format == 'huggingface':
                success, path = self._export_huggingface(
                    model_path, export_path, tokenizer, is_lora, merge_lora, progress_callback
                )
            elif export_format == 'gguf':
                success, path = self._export_gguf(
                    model_path, export_path, quantization or 'q4_k_m',
                    is_lora, merge_lora, progress_callback
                )
            elif export_format == 'merged':
                if not is_lora:
                    return False, "", {"error": "Model is not a LoRA model, cannot merge"}
                success, path = self._export_merged(
                    model_path, export_path, tokenizer, progress_callback
                )
            elif export_format == 'onnx':
                success, path = self._export_onnx(
                    model_path, export_path, tokenizer, is_lora, progress_callback
                )
            else:
                return False, "", {"error": f"Export format {export_format} not implemented"}

            if success:
                # Create metadata file
                metadata = self._create_metadata(
                    export_format, export_name, model_path,
                    quantization, merge_lora, is_lora
                )
                with open(export_path / "export_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Calculate file sizes
                total_size = sum(f.stat().st_size for f in export_path.rglob('*') if f.is_file())
                metadata['total_size_bytes'] = total_size
                metadata['total_size_mb'] = round(total_size / (1024 * 1024), 2)

                if progress_callback:
                    progress_callback(f"Export completed successfully!", 100)

                return True, str(path), metadata
            else:
                return False, "", {"error": "Export failed"}

        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False, "", {"error": str(e)}

    def _export_safetensors(
        self,
        model_path: str,
        export_path: Path,
        tokenizer,
        is_lora: bool,
        progress_callback: Optional[Callable]
    ) -> Tuple[bool, str]:
        """Export model in SafeTensors format."""
        try:
            if progress_callback:
                progress_callback("Loading model for SafeTensors export...", 20)

            if is_lora:
                # Load base model and LoRA adapter
                with open(Path(model_path) / "adapter_config.json", 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")

                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                model = PeftModel.from_pretrained(model, model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            if progress_callback:
                progress_callback("Saving in SafeTensors format...", 60)

            # Save model in safetensors format
            output_file = export_path / "model.safetensors"
            safetensors.torch.save_file(
                model.state_dict(),
                output_file,
                metadata={"format": "pt"}
            )

            # Save tokenizer and config
            tokenizer.save_pretrained(export_path)
            model.config.save_pretrained(export_path)

            if progress_callback:
                progress_callback("SafeTensors export complete", 100)

            return True, str(export_path)

        except Exception as e:
            logger.error(f"SafeTensors export failed: {str(e)}")
            return False, ""

    def _export_huggingface(
        self,
        model_path: str,
        export_path: Path,
        tokenizer,
        is_lora: bool,
        merge_lora: bool,
        progress_callback: Optional[Callable]
    ) -> Tuple[bool, str]:
        """Export model in HuggingFace format."""
        try:
            if progress_callback:
                progress_callback("Loading model for HuggingFace export...", 20)

            if is_lora:
                # Load base model and LoRA adapter
                with open(Path(model_path) / "adapter_config.json", 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")

                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                model = PeftModel.from_pretrained(model, model_path)

                if merge_lora:
                    if progress_callback:
                        progress_callback("Merging LoRA weights...", 40)
                    model = model.merge_and_unload()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            if progress_callback:
                progress_callback("Saving in HuggingFace format...", 60)

            # Save model and tokenizer
            model.save_pretrained(export_path)
            tokenizer.save_pretrained(export_path)

            # Create README
            readme_content = f"""# Exported Model

This model was exported from GRPO Fine-Tuner.

## Format
HuggingFace Transformers format

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{export_path}")
tokenizer = AutoTokenizer.from_pretrained("{export_path}")
```

## Export Details
- Exported at: {datetime.now().isoformat()}
- LoRA merged: {merge_lora if is_lora else 'N/A'}
"""
            with open(export_path / "README.md", 'w') as f:
                f.write(readme_content)

            if progress_callback:
                progress_callback("HuggingFace export complete", 100)

            return True, str(export_path)

        except Exception as e:
            logger.error(f"HuggingFace export failed: {str(e)}")
            return False, ""

    def _export_gguf(
        self,
        model_path: str,
        export_path: Path,
        quantization: str,
        is_lora: bool,
        merge_lora: bool,
        progress_callback: Optional[Callable]
    ) -> Tuple[bool, str]:
        """Export model in GGUF format using llama.cpp converter."""
        try:
            # First, we need to export to HuggingFace format
            hf_path = export_path / "hf_temp"
            hf_path.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback("Preparing model for GGUF conversion...", 20)

            # Load and save in HF format (potentially merging LoRA)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if is_lora:
                with open(Path(model_path) / "adapter_config.json", 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")

                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                model = PeftModel.from_pretrained(model, model_path)

                if merge_lora or True:  # Always merge for GGUF
                    if progress_callback:
                        progress_callback("Merging LoRA weights for GGUF...", 30)
                    model = model.merge_and_unload()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            # Save in HF format
            model.save_pretrained(hf_path)
            tokenizer.save_pretrained(hf_path)

            # Delete model to free memory before conversion
            del model
            torch.cuda.empty_cache()

            # Map "none" to "f32" for full precision
            actual_quantization = "f32" if quantization == "none" else quantization

            if progress_callback:
                progress_callback(f"Converting to GGUF {actual_quantization.upper()} format...", 50)

            # Convert to GGUF
            output_file = export_path / f"model-{actual_quantization}.gguf"
            success, error = self.gguf_converter.convert_with_llama_cpp(
                hf_path,
                output_file,
                actual_quantization,
                lambda msg: progress_callback(msg, 70) if progress_callback else None
            )

            # Clean up temporary HF files
            shutil.rmtree(hf_path)

            if success:
                # Create info file
                quant_desc, size_info = GGUFConverter.get_quantization_info(actual_quantization)
                info_content = f"""# GGUF Model

## Quantization: {actual_quantization.upper()}
{quant_desc}
Approximate size: {size_info}

## Usage with llama.cpp
```bash
./llama-cli -m {output_file.name} -p "Your prompt here"
```

## Usage with llama-cpp-python
```python
from llama_cpp import Llama

llm = Llama(model_path="{output_file.name}")
output = llm("Your prompt here", max_tokens=100)
```
"""
                with open(export_path / "README.md", 'w') as f:
                    f.write(info_content)

                if progress_callback:
                    progress_callback("GGUF export complete", 100)

                return True, str(export_path)
            else:
                logger.error(f"GGUF conversion failed: {error}")
                return False, ""

        except Exception as e:
            logger.error(f"GGUF export failed: {str(e)}")
            return False, ""

    def _export_merged(
        self,
        model_path: str,
        export_path: Path,
        tokenizer,
        progress_callback: Optional[Callable]
    ) -> Tuple[bool, str]:
        """Export model with LoRA weights merged."""
        try:
            if progress_callback:
                progress_callback("Loading LoRA model...", 20)

            # Load base model and LoRA adapter
            with open(Path(model_path) / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            if progress_callback:
                progress_callback("Loading LoRA adapter...", 40)

            model = PeftModel.from_pretrained(model, model_path)

            if progress_callback:
                progress_callback("Merging LoRA weights with base model...", 60)

            # Merge and unload LoRA weights
            model = model.merge_and_unload()

            if progress_callback:
                progress_callback("Saving merged model...", 80)

            # Save merged model
            model.save_pretrained(export_path)
            tokenizer.save_pretrained(export_path)

            # Create info file
            with open(export_path / "merge_info.json", 'w') as f:
                json.dump({
                    "base_model": base_model_name,
                    "lora_model": str(model_path),
                    "merged_at": datetime.now().isoformat(),
                    "merge_type": "linear"
                }, f, indent=2)

            if progress_callback:
                progress_callback("Merged model export complete", 100)

            return True, str(export_path)

        except Exception as e:
            logger.error(f"Merged export failed: {str(e)}")
            return False, ""

    def _export_onnx(
        self,
        model_path: str,
        export_path: Path,
        tokenizer,
        is_lora: bool,
        progress_callback: Optional[Callable]
    ) -> Tuple[bool, str]:
        """Export model in ONNX format."""
        try:
            if progress_callback:
                progress_callback("ONNX export not yet implemented", 50)

            # TODO: Implement ONNX export
            # This requires additional dependencies and is more complex
            return False, ""

        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            return False, ""

    def _create_metadata(
        self,
        export_format: str,
        export_name: str,
        model_path: str,
        quantization: Optional[str],
        merge_lora: bool,
        is_lora: bool
    ) -> Dict[str, Any]:
        """Create metadata for the export."""
        metadata = {
            "export_name": export_name,
            "export_format": export_format,
            "export_timestamp": datetime.now().isoformat(),
            "source_model_path": str(model_path),
            "is_lora": is_lora,
            "lora_merged": merge_lora if is_lora else None,
            "quantization": quantization,
            "export_tool": "GRPO Fine-Tuner",
            "export_version": "1.0.0"
        }

        # Add format-specific metadata
        if export_format == 'gguf' and quantization:
            quant_desc, size_info = GGUFConverter.get_quantization_info(quantization)
            metadata["quantization_description"] = quant_desc
            metadata["approximate_size"] = size_info

        return metadata

    def create_archive(
        self,
        export_path: str,
        archive_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Create a zip archive of the exported model.

        Args:
            export_path: Path to the exported model directory
            archive_name: Optional name for the archive

        Returns:
            Tuple of (success, archive_path)
        """
        try:
            export_path = Path(export_path)
            if not export_path.exists():
                return False, "Export path does not exist"

            archive_name = archive_name or f"{export_path.name}.zip"
            archive_path = export_path.parent / archive_name

            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(export_path.parent)
                        zipf.write(file_path, arcname)

            # Calculate checksum
            with open(archive_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            # Create checksum file
            with open(archive_path.with_suffix('.sha256'), 'w') as f:
                f.write(f"{checksum}  {archive_name}\n")

            return True, str(archive_path)

        except Exception as e:
            logger.error(f"Archive creation failed: {str(e)}")
            return False, str(e)

    def list_exports(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """List all exports or exports for a specific session.

        Args:
            session_id: Optional session ID to filter exports

        Returns:
            Dictionary with export information
        """
        exports = []

        search_path = self.export_dir / session_id if session_id else self.export_dir

        for metadata_file in search_path.rglob("export_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    metadata["path"] = str(metadata_file.parent)
                    exports.append(metadata)
            except Exception as e:
                logger.warning(f"Could not read metadata from {metadata_file}: {e}")

        return {
            "exports": sorted(exports, key=lambda x: x.get("export_timestamp", ""), reverse=True),
            "total": len(exports)
        }

    def cleanup_old_exports(self, days: int = 30) -> int:
        """Clean up exports older than specified days.

        Args:
            days: Number of days to keep exports

        Returns:
            Number of exports deleted
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for metadata_file in self.export_dir.rglob("export_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    export_date = datetime.fromisoformat(metadata.get("export_timestamp"))

                    if export_date < cutoff_date:
                        export_dir = metadata_file.parent
                        shutil.rmtree(export_dir)
                        deleted_count += 1
                        logger.info(f"Deleted old export: {export_dir}")
            except Exception as e:
                logger.warning(f"Could not process export for cleanup: {e}")

        return deleted_count