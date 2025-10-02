"""GGUF converter module for exporting models to GGUF format using llama.cpp."""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class GGUFConverter:
    """Handle conversion of models to GGUF format using llama.cpp tools."""

    @staticmethod
    def find_llama_cpp() -> Optional[Path]:
        """Find llama.cpp installation.

        Returns:
            Path to llama.cpp directory if found, None otherwise
        """
        # Common paths for llama.cpp installation
        llama_cpp_paths = [
            Path("tools/llama.cpp"),
            Path("tools\\llama.cpp"),
            Path("C:/llama.cpp"),
            Path("~/llama.cpp").expanduser(),
            Path("./llama.cpp"),
            Path("../llama.cpp"),
        ]

        # Find llama.cpp installation
        for path in llama_cpp_paths:
            # Check for new or old convert script names
            if path.exists() and (
                (path / "convert_hf_to_gguf.py").exists() or
                (path / "convert-hf-to-gguf.py").exists() or
                (path / "convert.py").exists()
            ):
                logger.info(f"Found llama.cpp at: {path}")
                return path

        logger.warning("llama.cpp not found in common locations")
        return None

    @staticmethod
    def convert_with_llama_cpp(
        model_dir: Path,
        output_file: Path,
        quantization: str = "q4_k_m",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """Convert model using llama.cpp tools.

        Args:
            model_dir: Directory containing HF model
            output_file: Output GGUF file path
            quantization: Quantization method (q8_0, q6_k, q5_k_m, q4_k_m, q4_0, f16)
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success, error_message)
        """
        # Find llama.cpp installation
        llama_cpp_dir = GGUFConverter.find_llama_cpp()

        if not llama_cpp_dir:
            error_msg = (
                "llama.cpp not found! Please install it."
            )
            logger.error(error_msg)
            return False, error_msg

        # Find the convert script (try all possible names)
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            convert_script = llama_cpp_dir / "convert-hf-to-gguf.py"
        if not convert_script.exists():
            convert_script = llama_cpp_dir / "convert.py"

        if not convert_script.exists():
            error_msg = f"Convert script not found in {llama_cpp_dir}"
            logger.error(error_msg)
            return False, error_msg

        # Find quantize executable for Windows
        quantize_exe = llama_cpp_dir / "llama-quantize.exe"
        if not quantize_exe.exists():
            quantize_exe = llama_cpp_dir / "quantize.exe"
        if not quantize_exe.exists():
            quantize_exe = llama_cpp_dir / "llama-quantize"
        if not quantize_exe.exists():
            quantize_exe = llama_cpp_dir / "quantize"

        logger.info(f"Using convert script: {convert_script.name}")
        if progress_callback:
            progress_callback(f"Using {convert_script.name}")

        # Convert to GGUF
        if progress_callback:
            progress_callback(f"Converting to GGUF {quantization.upper()} format...")

        # Build the command based on the script type
        cmd = [
            sys.executable,
            str(convert_script),
            str(model_dir),
            "--outfile", str(output_file),
            "--outtype", quantization.lower()
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"Initial conversion failed: {result.stderr}")

            # If the error is about outtype, try without it (older script format)
            if "--outtype" in result.stderr or "unrecognized arguments" in result.stderr:
                if progress_callback:
                    progress_callback("Retrying with older script format...")

                # Older scripts: convert to F16 first, then quantize
                f16_file = output_file.parent / f"{output_file.stem}_f16.gguf"
                cmd = [
                    sys.executable,
                    str(convert_script),
                    str(model_dir),
                    "--outfile", str(f16_file)
                ]

                logger.info(f"Retrying: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    error_msg = f"Conversion failed: {result.stderr}"
                    logger.error(error_msg)
                    return False, error_msg

                # Now quantize if needed
                if quantization != "f16":
                    if quantize_exe.exists():
                        if progress_callback:
                            progress_callback(f"Quantizing to {quantization}...")

                        result = subprocess.run([
                            str(quantize_exe),
                            str(f16_file),
                            str(output_file),
                            quantization
                        ], capture_output=True, text=True)

                        if result.returncode == 0:
                            f16_file.unlink(missing_ok=True)
                            logger.info("Quantization successful")
                        else:
                            error_msg = f"Quantization error: {result.stderr}"
                            logger.error(error_msg)
                            f16_file.unlink(missing_ok=True)
                            return False, error_msg
                    else:
                        logger.warning("Quantize tool not found, output will be F16")
                        if progress_callback:
                            progress_callback("Warning: Quantize tool not found, using F16")
                        f16_file.rename(output_file)
                else:
                    # Just rename if F16 was requested
                    f16_file.rename(output_file)
            else:
                error_msg = f"Conversion failed: {result.stderr}"
                logger.error(error_msg)
                return False, error_msg

        if progress_callback:
            progress_callback("GGUF conversion completed successfully!")

        return True, ""

    @staticmethod
    def get_quantization_info(quantization: str) -> Tuple[str, str]:
        """Get information about quantization method.

        Args:
            quantization: Quantization method

        Returns:
            Tuple of (description, approximate_size)
        """
        quant_info = {
            "f32": ("32-bit floating point (full precision)", "~4GB per billion parameters"),
            "f16": ("16-bit floating point (half precision)", "~2GB per billion parameters"),
            "q8_0": ("8-bit quantization (excellent quality)", "~1GB per billion parameters"),
            "auto": ("Automatic selection by converter", "Varies based on model"),
        }

        return quant_info.get(quantization, ("Unknown quantization", "Unknown size"))
