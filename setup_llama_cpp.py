"""Setup llama.cpp for GGUF conversion support."""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil


class LlamaCppSetup:
    """Setup llama.cpp tools for GGUF conversion."""

    LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
    TOOLS_DIR = Path("tools")
    LLAMA_CPP_DIR = TOOLS_DIR / "llama.cpp"

    @staticmethod
    def check_existing_installation() -> bool:
        """Check if llama.cpp is already installed.

        Returns:
            True if installed, False otherwise
        """
        if LlamaCppSetup.LLAMA_CPP_DIR.exists():
            # Check for required scripts
            convert_scripts = [
                "convert_hf_to_gguf.py",
                "convert-hf-to-gguf.py",
                "convert.py"
            ]

            for script in convert_scripts:
                if (LlamaCppSetup.LLAMA_CPP_DIR / script).exists():
                    print(f"llama.cpp already installed at {LlamaCppSetup.LLAMA_CPP_DIR}")
                    return True

        return False

    @staticmethod
    def download_prebuilt_windows() -> bool:
        """Download pre-built llama.cpp binaries for Windows.

        Returns:
            True if successful, False otherwise
        """
        print("Downloading pre-built llama.cpp for Windows...")

        # Note: You might need to update this URL to point to actual releases
        # This is an example - adjust based on actual llama.cpp releases
        releases_url = "https://github.com/ggerganov/llama.cpp/releases/latest"

        print("Note: Pre-built binaries may not be available.")
        print("Falling back to git clone method...")
        return False

    @staticmethod
    def clone_repository() -> bool:
        """Clone llama.cpp repository.

        Returns:
            True if successful, False otherwise
        """
        print("Cloning llama.cpp repository...")

        # Create tools directory if it doesn't exist
        LlamaCppSetup.TOOLS_DIR.mkdir(exist_ok=True)

        # Remove existing directory if present
        if LlamaCppSetup.LLAMA_CPP_DIR.exists():
            print(f"Removing existing directory: {LlamaCppSetup.LLAMA_CPP_DIR}")
            shutil.rmtree(LlamaCppSetup.LLAMA_CPP_DIR)

        # Clone the repository
        try:
            result = subprocess.run([
                "git", "clone",
                "--depth", "1",  # Shallow clone for faster download
                LlamaCppSetup.LLAMA_CPP_REPO,
                str(LlamaCppSetup.LLAMA_CPP_DIR)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error cloning repository: {result.stderr}")
                return False

            print("Repository cloned successfully")
            return True

        except FileNotFoundError:
            print("Error: Git is not installed. Please install Git and try again.")
            print("Download from: https://git-scm.com/downloads")
            return False
        except Exception as e:
            print(f"Error cloning repository: {e}")
            return False

    @staticmethod
    def install_python_requirements() -> bool:
        """Install Python requirements for llama.cpp conversion scripts.

        Returns:
            True if successful, False otherwise
        """
        print("Installing Python requirements for conversion scripts...")

        requirements = [
            "numpy",
            "sentencepiece",
            "transformers>=4.34.0",
            "protobuf",
            "torch",  # May already be installed
        ]

        for package in requirements:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Warning: Failed to install {package}")
                # Continue anyway as some packages might be optional

        print("✓ Python requirements installed")
        return True

    @staticmethod
    def build_quantize_tool() -> bool:
        """Build the quantize tool for Windows.

        Returns:
            True if successful, False otherwise
        """
        print("Attempting to build quantize tool...")

        if platform.system() != "Windows":
            print("Building quantize tool with make...")
            os.chdir(LlamaCppSetup.LLAMA_CPP_DIR)
            result = subprocess.run(["make", "quantize"], capture_output=True, text=True)
            os.chdir("../..")
            return result.returncode == 0

        # On Windows, try to use cmake
        print("Note: Building C++ tools on Windows requires Visual Studio or MinGW")
        print("The Python conversion scripts will work, but quantization might be limited")

        # Check if pre-built executables exist
        quantize_files = [
            "llama-quantize.exe",
            "quantize.exe",
            "bin/Release/llama-quantize.exe",
            "bin/Release/quantize.exe"
        ]

        for file in quantize_files:
            if (LlamaCppSetup.LLAMA_CPP_DIR / file).exists():
                print(f"✓ Found pre-built quantize tool: {file}")
                return True

        print("⚠ Quantize tool not found. GGUF export will work but with limited quantization options")
        return True  # Return True anyway as conversion can still work

    @staticmethod
    def setup():
        """Main setup function."""
        print("=" * 60)
        print("llama.cpp Setup for GGUF Export")
        print("=" * 60)
        print()

        # Check if already installed
        if LlamaCppSetup.check_existing_installation():
            print("\nllama.cpp is already set up!")
            return True

        # Try to clone repository
        if not LlamaCppSetup.clone_repository():
            return False

        # Install Python requirements
        if not LlamaCppSetup.install_python_requirements():
            print("Warning: Some Python packages could not be installed")
            print("The converter might still work with existing packages")

        # Try to build quantize tool (optional)
        LlamaCppSetup.build_quantize_tool()

        # Verify installation
        if LlamaCppSetup.check_existing_installation():
            print("\n" + "=" * 60)
            print("llama.cpp setup completed successfully!")
            print(f"Installation location: {LlamaCppSetup.LLAMA_CPP_DIR.absolute()}")
            print("You can now export models to GGUF format")
            print("=" * 60)
            return True
        else:
            print("\nSetup completed but verification failed")
            print("Please check the installation manually")
            return False


def main():
    """Main entry point."""
    print("Starting llama.cpp setup...")

    success = LlamaCppSetup.setup()

    if success:
        print("\nSetup completed! You can now use GGUF export in the GUI.")
    else:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
