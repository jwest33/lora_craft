"""System configuration and hardware detection module."""

import platform
import psutil
import torch
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import os
import warnings

try:
    # nvidia-ml-py package provides the pynvml module
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("nvidia-ml-py not available. NVIDIA GPU detection will be limited.")

try:
    import gpustat
    GPUSTAT_AVAILABLE = True
except ImportError:
    GPUSTAT_AVAILABLE = False


@dataclass
class GPUInfo:
    """GPU information container."""
    index: int
    name: str
    memory_total: int  # in MB
    memory_free: int   # in MB
    memory_used: int   # in MB
    utilization: float # percentage
    cuda_capability: Optional[Tuple[int, int]] = None
    driver_version: Optional[str] = None


@dataclass
class SystemInfo:
    """System information container."""
    os: str
    os_version: str
    python_version: str
    cpu_count: int
    cpu_model: str
    ram_total: int  # in MB
    ram_available: int  # in MB
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    gpu_count: int = 0
    gpus: list = None


@dataclass
class TrainingConfig:
    """Optimized training configuration based on system capabilities."""
    batch_size: int
    gradient_accumulation_steps: int
    max_sequence_length: int
    use_mixed_precision: bool
    use_gradient_checkpointing: bool
    use_cpu_offload: bool
    use_flash_attention: bool
    num_workers: int
    pin_memory: bool

    # Memory limits
    max_memory_per_gpu: Optional[Dict[int, str]] = None
    cpu_memory_limit: Optional[str] = None

    # Optimization settings
    optimizer_offload: bool = False
    activation_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SystemConfig:
    """System configuration and hardware detection."""

    # Model size estimates (in MB)
    MODEL_SIZES = {
        "unsloth/phi-4-reasoning": 30720,  # ~15B params, ~30GB VRAM
        "unsloth/Qwen3-0.6B": 1229,       # 0.6B params, ~1.2GB VRAM
        "unsloth/Qwen3-1.7B": 3482,       # 1.7B params, ~3.4GB VRAM
        "unsloth/Qwen3-4B": 8192,          # 4B params, ~8GB VRAM
        "unsloth/Qwen3-8B": 16384,         # 8B params, ~16GB VRAM
        "unsloth/Llama-3.2-1B-Instruct": 2048,  # 1B params, ~2GB VRAM
        "unsloth/Llama-3.2-3B-Instruct": 6144,  # 3B params, ~6GB VRAM
    }

    def __init__(self):
        """Initialize system configuration."""
        self.system_info = self._detect_system()
        self.gpu_info = self._detect_gpus()
        self._initialized = True

    def _detect_system(self) -> SystemInfo:
        """Detect system configuration."""
        # Try to get CPU info
        cpu_model = "Unknown"
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            cpu_model = cpu_info.get('brand_raw', 'Unknown')
        except ImportError:
            # cpuinfo not installed, try alternative methods
            import platform
            cpu_model = platform.processor() or "Unknown"

        # Get memory info
        mem = psutil.virtual_memory()

        # Get PyTorch info
        cuda_available = torch.cuda.is_available()
        cuda_version = None
        cudnn_version = None

        if cuda_available:
            cuda_version = torch.version.cuda
            cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None

        return SystemInfo(
            os=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            cpu_count=psutil.cpu_count(logical=True),
            cpu_model=cpu_model,
            ram_total=mem.total // (1024 * 1024),  # Convert to MB
            ram_available=mem.available // (1024 * 1024),
            pytorch_version=torch.__version__,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            gpu_count=torch.cuda.device_count() if cuda_available else 0,
            gpus=[]
        )

    def _detect_gpus(self) -> list:
        """Detect GPU configuration."""
        gpus = []

        if not self.system_info.cuda_available:
            return gpus

        # Try nvidia-ml-py (pynvml) first for detailed info
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    # Get CUDA capability
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

                    # Get driver version
                    driver = pynvml.nvmlDriverGetVersion()
                    if isinstance(driver, bytes):
                        driver = driver.decode('utf-8')

                    gpu = GPUInfo(
                        index=i,
                        name=name,
                        memory_total=mem_info.total // (1024 * 1024),
                        memory_free=mem_info.free // (1024 * 1024),
                        memory_used=mem_info.used // (1024 * 1024),
                        utilization=util.gpu,
                        cuda_capability=(major, minor),
                        driver_version=driver
                    )
                    gpus.append(gpu)

                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"Error detecting GPUs with nvidia-ml-py: {e}")

        # Fallback to torch for basic info
        if not gpus and self.system_info.cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu = GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total=props.total_memory // (1024 * 1024),
                    memory_free=0,  # Can't get from torch directly
                    memory_used=0,
                    utilization=0.0,
                    cuda_capability=(props.major, props.minor)
                )
                gpus.append(gpu)

        self.system_info.gpus = gpus
        return gpus

    def get_optimal_config(self,
                          model_size: str = "unsloth/Qwen3-1.7B",
                          task: str = "training") -> TrainingConfig:
        """Get optimal training configuration based on system capabilities.

        Args:
            model_size: Model identifier or size
            task: Task type ("training", "inference")

        Returns:
            TrainingConfig with optimized settings
        """
        # Get model memory requirements
        model_memory = self.MODEL_SIZES.get(model_size, 3482)  # Default to Qwen3-1.7B size

        # Calculate available memory
        if self.gpu_info:
            # Use first GPU as reference
            gpu = self.gpu_info[0]
            available_vram = gpu.memory_free
            total_vram = gpu.memory_total
        else:
            available_vram = 0
            total_vram = 0

        available_ram = self.system_info.ram_available

        # Determine batch size based on available memory
        if task == "training":
            # Training needs more memory for gradients and optimizer states
            memory_per_sample = model_memory * 3 // 1024  # Rough estimate
            if available_vram > 0:
                max_batch_size = max(1, available_vram // (memory_per_sample * 2))
            else:
                max_batch_size = max(1, available_ram // (memory_per_sample * 4))
        else:
            # Inference needs less memory
            memory_per_sample = model_memory // 1024
            if available_vram > 0:
                max_batch_size = max(1, available_vram // memory_per_sample)
            else:
                max_batch_size = max(1, available_ram // (memory_per_sample * 2))

        # Determine optimal batch size and gradient accumulation
        # For GRPO with Qwen3-1.7B, use smaller batches
        if "Qwen3" in model_size and task == "training":
            # GRPO works best with batch_size=1 and gradient accumulation
            batch_size = 1
            gradient_accumulation = 1  # Can increase to 4 for smoother training
        elif max_batch_size >= 8:
            batch_size = 8
            gradient_accumulation = 1
        elif max_batch_size >= 4:
            batch_size = 4
            gradient_accumulation = 2
        elif max_batch_size >= 2:
            batch_size = 2
            gradient_accumulation = 4
        else:
            batch_size = 1
            gradient_accumulation = 4

        # Determine sequence length based on memory
        # For GRPO reasoning, we want 2048 for reasoning traces
        if "Qwen3" in model_size and task == "training":
            max_seq_length = 2048  # Optimal for reasoning traces
        elif total_vram > 16000:  # 16GB+
            max_seq_length = 2048
        elif total_vram > 8000:  # 8GB+
            max_seq_length = 1024
        elif total_vram > 4000:  # 4GB+
            max_seq_length = 512
        else:
            max_seq_length = 256

        # Determine optimization flags
        # For Qwen3 models, use 16-bit LoRA (not 4-bit) for better performance
        use_mixed_precision = self.system_info.cuda_available and total_vram > 2000
        # Enable gradient checkpointing for memory efficiency ("unsloth" mode)
        use_gradient_checkpointing = True  # Always use for GRPO
        use_cpu_offload = total_vram < 4000 and available_ram > 16000

        # Check for flash attention support (requires Ampere or newer)
        use_flash_attention = False
        if self.gpu_info and self.gpu_info[0].cuda_capability:
            major, minor = self.gpu_info[0].cuda_capability
            use_flash_attention = major >= 8  # Ampere and newer

        # Memory limits
        max_memory_per_gpu = None
        if self.gpu_info:
            max_memory_per_gpu = {
                i: f"{int(gpu.memory_total * 0.9)}MB"
                for i, gpu in enumerate(self.gpu_info)
            }

        cpu_memory_limit = f"{int(available_ram * 0.5)}MB"

        return TrainingConfig(
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            max_sequence_length=max_seq_length,
            use_mixed_precision=use_mixed_precision,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_cpu_offload=use_cpu_offload,
            use_flash_attention=use_flash_attention,
            num_workers=min(4, self.system_info.cpu_count // 2),
            pin_memory=self.system_info.cuda_available,
            max_memory_per_gpu=max_memory_per_gpu,
            cpu_memory_limit=cpu_memory_limit,
            optimizer_offload=use_cpu_offload,
            activation_checkpointing=use_gradient_checkpointing
        )

    def get_system_summary(self) -> str:
        """Get a human-readable system summary."""
        lines = [
            "System Configuration Summary",
            "=" * 40,
            f"OS: {self.system_info.os} {self.system_info.os_version}",
            f"Python: {self.system_info.python_version}",
            f"PyTorch: {self.system_info.pytorch_version}",
            f"CPU: {self.system_info.cpu_model} ({self.system_info.cpu_count} cores)",
            f"RAM: {self.system_info.ram_total / 1024:.1f} GB total, {self.system_info.ram_available / 1024:.1f} GB available",
            ""
        ]

        if self.system_info.cuda_available:
            lines.append(f"CUDA: {self.system_info.cuda_version}")
            if self.system_info.cudnn_version:
                lines.append(f"cuDNN: {self.system_info.cudnn_version}")
            lines.append(f"GPU Count: {self.system_info.gpu_count}")
            lines.append("")

            for gpu in self.gpu_info:
                lines.append(f"GPU {gpu.index}: {gpu.name}")
                lines.append(f"  Memory: {gpu.memory_total / 1024:.1f} GB total, {gpu.memory_free / 1024:.1f} GB free")
                lines.append(f"  Utilization: {gpu.utilization:.1f}%")
                if gpu.cuda_capability:
                    lines.append(f"  CUDA Capability: {gpu.cuda_capability[0]}.{gpu.cuda_capability[1]}")
                lines.append("")
        else:
            lines.append("No CUDA-capable GPU detected")

        return "\n".join(lines)

    def save_config(self, filepath: str):
        """Save system configuration to file."""
        config = {
            "system_info": asdict(self.system_info),
            "gpu_info": [asdict(gpu) for gpu in self.gpu_info]
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def validate_requirements(self, min_ram_gb: float = 8, min_vram_gb: float = 4) -> Tuple[bool, list]:
        """Validate system meets minimum requirements.

        Args:
            min_ram_gb: Minimum RAM in GB
            min_vram_gb: Minimum VRAM in GB

        Returns:
            Tuple of (meets_requirements, list_of_warnings)
        """
        warnings = []
        meets_requirements = True

        # Check RAM
        ram_gb = self.system_info.ram_total / 1024
        if ram_gb < min_ram_gb:
            warnings.append(f"RAM ({ram_gb:.1f} GB) is below recommended minimum ({min_ram_gb} GB)")
            meets_requirements = False

        # Check GPU
        if not self.system_info.cuda_available:
            warnings.append("No CUDA-capable GPU detected. Training will be slow on CPU.")
        elif self.gpu_info:
            vram_gb = self.gpu_info[0].memory_total / 1024
            if vram_gb < min_vram_gb:
                warnings.append(f"VRAM ({vram_gb:.1f} GB) is below recommended minimum ({min_vram_gb} GB)")
                meets_requirements = False

        # Check PyTorch
        if not torch.__version__.startswith(('2.', '3.')):
            warnings.append(f"PyTorch version {torch.__version__} may not be optimal. Consider upgrading to 2.0+")

        return meets_requirements, warnings


if __name__ == "__main__":
    # Test the system configuration
    config = SystemConfig()
    print(config.get_system_summary())
    print("\nOptimal Training Config for unsloth/Qwen3-1.7B:")
    optimal = config.get_optimal_config("unsloth/Qwen3-1.7B")
    for key, value in optimal.to_dict().items():
        print(f"  {key}: {value}")

    meets_req, warnings = config.validate_requirements()
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
