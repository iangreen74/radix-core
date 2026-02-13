"""
vLLM Local Runner with Graceful Degradation

This module provides a wrapper around vLLM for high-performance inference
with graceful fallback when vLLM is not available.
"""

import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import threading

# Graceful vLLM import
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = SamplingParams = AsyncEngineArgs = None

from ..config_v2 import get_config
from ..logging import get_logger
from ..utils.timers import time_operation
from ..utils.randfail import seeded_failure
from ..dryrun import DryRunGuard

logger = get_logger(__name__)


@dataclass
class VLLMGenerationResult:
    """Result of vLLM text generation."""

    prompt: str
    generated_text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float
    finish_reason: str
    metadata: Dict[str, Any]


class VLLMLocalRunner:
    """
    vLLM runner for high-performance local inference.

    Features:
    - High-throughput batch inference
    - GPU memory optimization
    - Graceful degradation when vLLM unavailable
    - Comprehensive safety guards
    - Performance monitoring
    """

    def __init__(self,
                 model_name: str = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int = None,
                 enforce_eager: bool = True):
        """
        Initialize vLLM runner.

        Args:
            model_name: Model name/path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model sequence length
            enforce_eager: Use eager execution (safer for local)
        """
        if not VLLM_AVAILABLE:
            logger.warning("vLLM not available - runner will operate in fallback mode")
            self.available = False
            return

        self.config = get_config()
        self.available = True

        # Model configuration
        self.model_name = model_name or self.config.ml.default_model_name
        self.tensor_parallel_size = min(tensor_parallel_size, 1)  # Single GPU for safety
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len or self.config.ml.max_sequence_length
        self.enforce_eager = enforce_eager

        # vLLM engine
        self.llm: Optional[LLM] = None
        self.is_loaded = False

        # Default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,
            stop=None
        ) if VLLM_AVAILABLE else None

        # Statistics
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0

        # Thread safety
        self.lock = threading.RLock()

        if self.available:
            logger.info("vLLM runner initialized",
                       model_name=self.model_name,
                       tensor_parallel_size=self.tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization)
        else:
            logger.info("vLLM runner initialized in fallback mode")

    def load_model(self):
        """Load vLLM model with safety constraints."""
        if not self.available:
            logger.warning("Cannot load model - vLLM not available")
            return

        if self.is_loaded:
            return

        # Safety checks
        if not self.config.execution.enable_cuda:
            logger.warning("vLLM requires CUDA but CUDA is disabled")
            self.available = False
            return

        try:
            with time_operation(f"vllm_model_loading_{self.model_name}"):
                logger.info("Loading vLLM model",
                           model_name=self.model_name,
                           tensor_parallel_size=self.tensor_parallel_size)

                # Create vLLM engine
                self.llm = LLM(
                    model=self.model_name,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    enforce_eager=self.enforce_eager,
                    trust_remote_code=False,  # Safety: no remote code
                    disable_log_stats=False,
                    disable_log_requests=False
                )

                self.is_loaded = True

                logger.info("vLLM model loaded successfully",
                           model_name=self.model_name,
                           max_model_len=self.max_model_len)

        except Exception as e:
            logger.error("Failed to load vLLM model",
                        model_name=self.model_name,
                        error=str(e))
            self.available = False
            raise

    @DryRunGuard.protect
    def generate(self,
                prompts: Union[str, List[str]],
                sampling_params: Optional[SamplingParams] = None) -> Union[VLLMGenerationResult, List[VLLMGenerationResult]]:
        """
        Generate text using vLLM.

        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters for generation

        Returns:
            Generation result(s)
        """
        if not self.available:
            raise RuntimeError("vLLM not available - cannot generate text")

        if not self.is_loaded:
            self.load_model()

        # Check for failure injection
        seeded_failure("vllm_generation")

        # Handle single prompt
        single_prompt = isinstance(prompts, str)
        if single_prompt:
            prompts = [prompts]

        # Use default sampling params if not provided
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        results = []

        with time_operation(f"vllm_generation_batch_{len(prompts)}"):
            start_time = time.time()

            try:
                logger.info("Starting vLLM generation",
                           batch_size=len(prompts),
                           max_tokens=sampling_params.max_tokens,
                           temperature=sampling_params.temperature)

                # Generate with vLLM
                outputs = self.llm.generate(prompts, sampling_params)

                generation_time = (time.time() - start_time) * 1000

                # Process outputs
                for i, output in enumerate(outputs):
                    prompt = prompts[i]
                    generated_text = output.outputs[0].text
                    tokens_generated = len(output.outputs[0].token_ids)
                    finish_reason = output.outputs[0].finish_reason

                    tokens_per_second = tokens_generated / (generation_time / 1000) if generation_time > 0 else 0

                    result = VLLMGenerationResult(
                        prompt=prompt,
                        generated_text=generated_text,
                        tokens_generated=tokens_generated,
                        generation_time_ms=generation_time / len(prompts),  # Per-prompt time
                        tokens_per_second=tokens_per_second,
                        finish_reason=finish_reason,
                        metadata={
                            "model_name": self.model_name,
                            "tensor_parallel_size": self.tensor_parallel_size,
                            "sampling_params": {
                                "temperature": sampling_params.temperature,
                                "top_p": sampling_params.top_p,
                                "max_tokens": sampling_params.max_tokens
                            }
                        }
                    )

                    results.append(result)

                    logger.debug("vLLM generation completed",
                               prompt_length=len(prompt),
                               tokens_generated=tokens_generated,
                               finish_reason=finish_reason,
                               tokens_per_second=tokens_per_second)

                # Update statistics
                with self.lock:
                    self.total_requests += len(prompts)
                    self.total_tokens_generated += sum(r.tokens_generated for r in results)
                    self.total_generation_time += generation_time

                logger.info("vLLM batch generation completed",
                           batch_size=len(prompts),
                           total_tokens=sum(r.tokens_generated for r in results),
                           total_time_ms=generation_time)

            except Exception as e:
                logger.error("vLLM generation failed", error=str(e))

                # Create failed results
                for prompt in prompts:
                    result = VLLMGenerationResult(
                        prompt=prompt,
                        generated_text="",
                        tokens_generated=0,
                        generation_time_ms=0.0,
                        tokens_per_second=0.0,
                        finish_reason="error",
                        metadata={"error": str(e)}
                    )
                    results.append(result)

        return results[0] if single_prompt else results

    def create_sampling_params(self,
                             temperature: float = 0.7,
                             top_p: float = 0.9,
                             max_tokens: int = 50,
                             stop: Optional[List[str]] = None) -> Optional[SamplingParams]:
        """
        Create sampling parameters for generation.

        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            SamplingParams object or None if vLLM unavailable
        """
        if not self.available:
            return None

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        if not self.available:
            return {
                "runner_type": "vllm_local",
                "available": False,
                "error": "vLLM not installed or not available"
            }

        with self.lock:
            avg_generation_time = (self.total_generation_time / max(self.total_requests, 1))
            avg_tokens_per_request = (self.total_tokens_generated / max(self.total_requests, 1))
            avg_tokens_per_second = (self.total_tokens_generated /
                                   max(self.total_generation_time / 1000, 0.001))

            return {
                "runner_type": "vllm_local",
                "available": True,
                "is_loaded": self.is_loaded,
                "total_requests": self.total_requests,
                "total_tokens_generated": self.total_tokens_generated,
                "total_generation_time_ms": self.total_generation_time,
                "avg_generation_time_ms": avg_generation_time,
                "avg_tokens_per_request": avg_tokens_per_request,
                "avg_tokens_per_second": avg_tokens_per_second,
                "config": {
                    "model_name": self.model_name,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_model_len": self.max_model_len,
                    "enforce_eager": self.enforce_eager
                }
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.available or not self.is_loaded:
            return {"error": "Model not loaded or vLLM not available"}

        try:
            # Get basic model info
            return {
                "model_name": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "max_model_len": self.max_model_len,
                "loaded": self.is_loaded,
                "engine_type": "vllm"
            }
        except Exception as e:
            return {"error": str(e)}

    def unload_model(self):
        """Unload vLLM model to free memory."""
        if not self.available or not self.is_loaded:
            return

        try:
            logger.info("Unloading vLLM model", model_name=self.model_name)

            # vLLM doesn't have explicit unload, but we can clear the reference
            del self.llm
            self.llm = None
            self.is_loaded = False

            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("vLLM model unloaded successfully")

        except Exception as e:
            logger.error("Error unloading vLLM model", error=str(e))

    def __enter__(self):
        """Context manager entry."""
        if self.available:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.available:
            self.unload_model()


# Fallback function for when vLLM is not available
def create_vllm_fallback_message() -> str:
    """Create a helpful message when vLLM is not available."""
    return """
vLLM is not available in this environment.

To enable vLLM support:
1. Install vLLM: pip install vllm
2. Ensure CUDA is available and properly configured
3. Set ENABLE_CUDA=true in your configuration

vLLM provides high-performance inference but is optional.
You can use the HuggingFace runner as an alternative.
"""


# Convenience function to check vLLM availability
def check_vllm_availability() -> Dict[str, Any]:
    """Check if vLLM is available and properly configured."""
    result = {
        "vllm_available": VLLM_AVAILABLE,
        "cuda_available": False,
        "recommendations": []
    }

    if not VLLM_AVAILABLE:
        result["recommendations"].append("Install vLLM: pip install vllm")

    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            result["recommendations"].append("CUDA not available - vLLM requires GPU")
    except ImportError:
        result["recommendations"].append("PyTorch not available")

    config = get_config()
    if not config.execution.enable_cuda:
        result["recommendations"].append("Enable CUDA in configuration")

    if not result["recommendations"]:
        result["status"] = "ready"
    else:
        result["status"] = "not_ready"

    return result
