"""Hyperparameters and device configuration for the statistical language model."""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Try to import cupy for GPU support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_gpu_info() -> dict:
    """
    Get GPU information using nvidia-smi.

    Returns:
        Dictionary with GPU info or empty dict if no GPU
    """
    gpu_info = {
        "available": False,
        "name": None,
        "memory_total": None,
        "memory_used": None,
        "memory_free": None,
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split(", ")
                gpu_info["available"] = True
                gpu_info["name"] = parts[0]
                gpu_info["memory_total"] = float(parts[1])
                gpu_info["memory_used"] = float(parts[2])
                gpu_info["memory_free"] = float(parts[3])
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return gpu_info


def get_device() -> str:
    """
    Detect the best available device.

    Priority: CUDA GPU > CPU

    Returns:
        Device string ("cuda" or "cpu")
    """
    # Check for CUDA via PyTorch
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"

    # Check for CUDA via environment variable
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        if CUPY_AVAILABLE:
            return "cuda"

    # Check for nvidia-smi
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        if CUPY_AVAILABLE:
            return "cuda"
        else:
            print("GPU detected but cupy not installed. Using CPU.")
            print("Install cupy: pip install cupy-cuda11x")

    return "cpu"


def get_device_info(device: str) -> str:
    """
    Get detailed device information.

    Args:
        device: Device string

    Returns:
        Device description
    """
    if device == "cuda":
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            return (f"GPU: {gpu_info['name']} | "
                    f"Memory: {gpu_info['memory_free']:.0f}MB free / "
                    f"{gpu_info['memory_total']:.0f}MB total")
        return "GPU detected"
    return "CPU"


@dataclass
class Config:
    # Device
    device: str = field(default_factory=get_device)

    # Data
    vocab_size: int = 50000
    min_token_freq: int = 5
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Embeddings
    embed_dim: int = 300
    cooccurrence_window: int = 5
    ppmi_smooth: float = 1.0
    svd_n_components: int = 300

    # Markov chain
    max_ngram_order: int = 6
    kneser_ney_discount: float = 0.75

    # HMM
    hmm_n_states: int = 100
    hmm_n_iterations: int = 50
    hmm_tolerance: float = 1e-4

    # Topic model
    n_topics: int = 50
    lda_alpha: float = 0.1
    lda_beta: float = 0.01
    lda_n_iterations: int = 500
    lda_burn_in: int = 100

    # Interpolation
    interpolation_method: str = "em"
    n_component_models: int = 4

    # Training
    batch_size: int = 1024
    learning_rate: float = 0.01
    n_training_steps: int = 100000
    warmup_steps: int = 1000
    eval_interval: int = 1000

    # Temperature for sampling
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

    # Random seed
    seed: int = 42

    def print_device_info(self):
        """Print device information."""
        print("=" * 60)
        print("DEVICE CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.device.upper()}")
        print(f"Info: {get_device_info(self.device)}")

        if self.device == "cuda":
            gpu_info = get_gpu_info()
            if gpu_info["available"]:
                print(f"CuPy available: {CUPY_AVAILABLE}")
                if CUPY_AVAILABLE:
                    print(f"CuPy device: {cp.cuda.Device().id}")
        print("=" * 60)


def xp(device: str = "cpu"):
    """
    Get the appropriate array library for the device.

    Args:
        device: Device string

    Returns:
        numpy or cupy module
    """
    if device == "cuda" and CUPY_AVAILABLE:
        return cp
    import numpy as np
    return np


def to_device(array, device: str = "cpu"):
    """
    Move array to device.

    Args:
        array: numpy or cupy array
        device: Target device

    Returns:
        Array on target device
    """
    if device == "cuda" and CUPY_AVAILABLE:
        return cp.asarray(array)
    return array


def to_cpu(array, device: str = "cpu"):
    """
    Move array to CPU.

    Args:
        array: numpy or cupy array
        device: Source device

    Returns:
        numpy array
    """
    if device == "cuda" and CUPY_AVAILABLE:
        return cp.asnumpy(array)
    return array


def get_gpu_memory_usage() -> dict:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory usage
    """
    if not CUPY_AVAILABLE:
        return {"used": 0, "total": 0, "free": 0}

    try:
        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes() / 1024**2  # MB
        gpu_info = get_gpu_info()
        return {
            "used": used,
            "total": gpu_info.get("memory_total", 0),
            "free": gpu_info.get("memory_free", 0),
        }
    except Exception:
        return {"used": 0, "total": 0, "free": 0}
