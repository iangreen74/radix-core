"""
Executor modules for Radix

This package provides various execution backends for running jobs locally
with safety guards and comprehensive monitoring.
"""

from .threadpool import ThreadPoolExecutor as RadixThreadPoolExecutor

# Conditional imports for optional ML dependencies
try:
    from .ray_local import RayLocalExecutor

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    RayLocalExecutor = None

try:
    from .hf_runner import HuggingFaceRunner

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HuggingFaceRunner = None

# Conditional import for vLLM (graceful degradation)
try:
    from .vllm_local import VLLMLocalRunner

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VLLMLocalRunner = None

__all__ = [
    "RadixThreadPoolExecutor",
    "RayLocalExecutor",
    "HuggingFaceRunner",
    "VLLMLocalRunner",
    "RAY_AVAILABLE",
    "HF_AVAILABLE",
    "VLLM_AVAILABLE",
]
