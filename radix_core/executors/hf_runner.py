"""
Hugging Face Runner for Text Generation and Embeddings

This module provides a safe wrapper around Hugging Face transformers with
accelerate integration for single GPU usage and comprehensive safety guards.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch

try:
    from accelerate import Accelerator
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        GenerationConfig,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = AutoModel = AutoModelForCausalLM = None
    AutoModelForSequenceClassification = pipeline = GenerationConfig = None
    Accelerator = None

from ..config import get_config
from ..dryrun import DryRunGuard
from ..logging import get_logger
from ..utils.randfail import seeded_failure
from ..utils.timers import time_operation

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    model_name: str
    model_type: str
    device: str
    precision: str
    memory_usage_mb: float
    loaded_at: datetime
    parameters: int


@dataclass
class GenerationResult:
    """Result of text generation."""

    input_text: str
    generated_text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float
    metadata: Dict[str, Any]


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    input_text: str
    embedding: List[float]
    embedding_dim: int
    generation_time_ms: float
    metadata: Dict[str, Any]


class HuggingFaceRunner:
    """
    Safe Hugging Face model runner with accelerate integration.

    Features:
    - Single GPU usage with CPU fallback
    - Memory-efficient model loading
    - Comprehensive safety guards
    - Performance monitoring
    - Batch processing support
    """

    def __init__(
        self,
        model_name: str = None,
        task: str = "text-generation",
        device: str = "auto",
        precision: str = None,
    ):
        """
        Initialize Hugging Face runner.

        Args:
            model_name: Model name/path (from config if None)
            task: Task type (text-generation, embeddings, classification)
            device: Device to use (auto, cpu, cuda)
            precision: Model precision (fp16, fp32, bf16)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers not available. Install with: pip install transformers accelerate"
            )

        self.config = get_config()

        # Model configuration
        ml_config = getattr(self.config, "ml", None)
        self.model_name = model_name or getattr(ml_config, "default_model_name", "gpt2")
        self.task = task
        self.device = device
        self.precision = precision or getattr(ml_config, "precision", "fp16")

        # Model state
        self.model = None
        self.tokenizer = None
        self.accelerator = None
        self.model_info: Optional[ModelInfo] = None
        self.is_loaded = False

        # Generation configuration
        self.generation_config = None

        # Statistics
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            "HuggingFace runner initialized",
            model_name=self.model_name,
            task=self.task,
            device=self.device,
            precision=self.precision,
        )

    def load_model(self):
        """Load model and tokenizer with safety checks."""
        if self.is_loaded:
            return

        try:
            with time_operation(f"model_loading_{self.model_name}"):
                logger.info(
                    "Loading model and tokenizer", model_name=self.model_name, task=self.task
                )

                # Initialize accelerator for device management
                enable_cuda = getattr(
                    self.config.execution,
                    "enable_cuda",
                    getattr(self.config.execution, "enable_gpu", False),
                )
                self.accelerator = Accelerator(
                    mixed_precision=self.precision if self.precision != "fp32" else "no",
                    cpu=not enable_cuda or self.device == "cpu",
                )

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=False,  # Safety: no remote code
                    local_files_only=False,  # Allow download for research
                    padding_side="left",
                )

                # Add pad token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model based on task
                if self.task == "text-generation":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=False,
                        torch_dtype=self._get_torch_dtype(),
                        device_map=(
                            "auto"
                            if getattr(
                                self.config.execution,
                                "enable_cuda",
                                getattr(self.config.execution, "enable_gpu", False),
                            )
                            else None
                        ),
                        low_cpu_mem_usage=True,
                    )
                elif self.task == "embeddings":
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=False,
                        torch_dtype=self._get_torch_dtype(),
                        device_map=(
                            "auto"
                            if getattr(
                                self.config.execution,
                                "enable_cuda",
                                getattr(self.config.execution, "enable_gpu", False),
                            )
                            else None
                        ),
                        low_cpu_mem_usage=True,
                    )
                elif self.task == "classification":
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        trust_remote_code=False,
                        torch_dtype=self._get_torch_dtype(),
                        device_map=(
                            "auto"
                            if getattr(
                                self.config.execution,
                                "enable_cuda",
                                getattr(self.config.execution, "enable_gpu", False),
                            )
                            else None
                        ),
                        low_cpu_mem_usage=True,
                    )
                else:
                    raise ValueError(f"Unsupported task: {self.task}")

                # Prepare model with accelerator
                if not getattr(
                    self.config.execution,
                    "enable_cuda",
                    getattr(self.config.execution, "enable_gpu", False),
                ):
                    self.model = self.model.to("cpu")

                self.model = self.accelerator.prepare(self.model)

                # Set up generation config for text generation
                if self.task == "text-generation":
                    self.generation_config = GenerationConfig(
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Calculate model info
                device_name = str(next(self.model.parameters()).device)
                param_count = sum(p.numel() for p in self.model.parameters())
                memory_usage = self._estimate_memory_usage()

                self.model_info = ModelInfo(
                    model_name=self.model_name,
                    model_type=self.task,
                    device=device_name,
                    precision=self.precision,
                    memory_usage_mb=memory_usage,
                    loaded_at=datetime.utcnow(),
                    parameters=param_count,
                )

                self.is_loaded = True

                logger.info(
                    "Model loaded successfully",
                    model_name=self.model_name,
                    device=device_name,
                    parameters=param_count,
                    memory_mb=memory_usage,
                )

        except Exception as e:
            logger.error("Failed to load model", model_name=self.model_name, error=str(e))
            raise

    @DryRunGuard.protect
    def generate_text(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """
        Generate text from prompts.

        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generation result(s)
        """
        if not self.is_loaded:
            self.load_model()

        if self.task != "text-generation":
            raise ValueError("Model not configured for text generation")

        # Check for failure injection
        seeded_failure("hf_text_generation")

        # Handle single prompt
        single_prompt = isinstance(prompts, str)
        if single_prompt:
            prompts = [prompts]

        # Update generation config if provided
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p

        results = []

        with time_operation(f"text_generation_batch_{len(prompts)}"):
            for prompt in prompts:
                start_time = time.time()

                try:
                    # Tokenize input
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=getattr(
                            getattr(self.config, "ml", None), "max_sequence_length", 512
                        ),
                    )

                    # Move to device
                    inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
                    input_length = inputs["input_ids"].shape[1]

                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=gen_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    # Decode output
                    generated_tokens = outputs[0][input_length:]
                    generated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )

                    generation_time = (time.time() - start_time) * 1000
                    tokens_generated = len(generated_tokens)
                    tokens_per_second = (
                        tokens_generated / (generation_time / 1000) if generation_time > 0 else 0
                    )

                    result = GenerationResult(
                        input_text=prompt,
                        generated_text=generated_text,
                        tokens_generated=tokens_generated,
                        generation_time_ms=generation_time,
                        tokens_per_second=tokens_per_second,
                        metadata={
                            "model_name": self.model_name,
                            "device": str(self.accelerator.device),
                            "precision": self.precision,
                            "max_new_tokens": gen_config.max_new_tokens,
                            "temperature": gen_config.temperature,
                        },
                    )

                    results.append(result)

                    # Update statistics
                    with self.lock:
                        self.total_generations += 1
                        self.total_tokens_generated += tokens_generated
                        self.total_generation_time += generation_time

                    logger.debug(
                        "Text generated",
                        prompt_length=len(prompt),
                        tokens_generated=tokens_generated,
                        generation_time_ms=generation_time,
                        tokens_per_second=tokens_per_second,
                    )

                except Exception as e:
                    logger.error("Text generation failed", prompt=prompt[:100], error=str(e))

                    # Create failed result
                    result = GenerationResult(
                        input_text=prompt,
                        generated_text="",
                        tokens_generated=0,
                        generation_time_ms=(time.time() - start_time) * 1000,
                        tokens_per_second=0.0,
                        metadata={"error": str(e)},
                    )
                    results.append(result)

        return results[0] if single_prompt else results

    @DryRunGuard.protect
    def generate_embeddings(
        self, texts: Union[str, List[str]]
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Embedding result(s)
        """
        if not self.is_loaded:
            self.load_model()

        if self.task != "embeddings":
            raise ValueError("Model not configured for embeddings")

        # Check for failure injection
        seeded_failure("hf_embedding_generation")

        # Handle single text
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        results = []

        with time_operation(f"embedding_generation_batch_{len(texts)}"):
            for text in texts:
                start_time = time.time()

                try:
                    # Tokenize input
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=getattr(
                            getattr(self.config, "ml", None), "max_sequence_length", 512
                        ),
                    )

                    # Move to device
                    inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}

                    # Generate embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                        # Use CLS token embedding or mean pooling
                        if hasattr(outputs, "last_hidden_state"):
                            # Mean pooling over sequence length
                            embeddings = outputs.last_hidden_state.mean(dim=1)
                        else:
                            # Use pooler output if available
                            embeddings = outputs.pooler_output

                        # Convert to list
                        embedding = embeddings.cpu().numpy().flatten().tolist()

                    generation_time = (time.time() - start_time) * 1000

                    result = EmbeddingResult(
                        input_text=text,
                        embedding=embedding,
                        embedding_dim=len(embedding),
                        generation_time_ms=generation_time,
                        metadata={
                            "model_name": self.model_name,
                            "device": str(self.accelerator.device),
                            "precision": self.precision,
                            "sequence_length": inputs["input_ids"].shape[1],
                        },
                    )

                    results.append(result)

                    logger.debug(
                        "Embedding generated",
                        text_length=len(text),
                        embedding_dim=len(embedding),
                        generation_time_ms=generation_time,
                    )

                except Exception as e:
                    logger.error("Embedding generation failed", text=text[:100], error=str(e))

                    # Create failed result
                    result = EmbeddingResult(
                        input_text=text,
                        embedding=[],
                        embedding_dim=0,
                        generation_time_ms=(time.time() - start_time) * 1000,
                        metadata={"error": str(e)},
                    )
                    results.append(result)

        return results[0] if single_text else results

    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from precision setting."""
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        return dtype_map.get(self.precision, torch.float16)

    def _estimate_memory_usage(self) -> float:
        """Estimate model memory usage in MB."""
        if self.model is None:
            return 0.0

        try:
            # Count parameters and estimate memory
            param_count = sum(p.numel() for p in self.model.parameters())

            # Bytes per parameter based on precision
            bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}.get(self.precision, 2)

            # Estimate: parameters + gradients + optimizer states + activations
            memory_bytes = param_count * bytes_per_param * 1.5  # Conservative estimate

            return memory_bytes / (1024 * 1024)  # Convert to MB

        except Exception:
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        with self.lock:
            avg_generation_time = self.total_generation_time / max(self.total_generations, 1)
            avg_tokens_per_generation = self.total_tokens_generated / max(self.total_generations, 1)
            avg_tokens_per_second = self.total_tokens_generated / max(
                self.total_generation_time / 1000, 0.001
            )

            return {
                "runner_type": "huggingface",
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "is_loaded": self.is_loaded,
                "model_info": self.model_info.__dict__ if self.model_info else None,
                "total_generations": self.total_generations,
                "total_tokens_generated": self.total_tokens_generated,
                "total_generation_time_ms": self.total_generation_time,
                "avg_generation_time_ms": avg_generation_time,
                "avg_tokens_per_generation": avg_tokens_per_generation,
                "avg_tokens_per_second": avg_tokens_per_second,
                "config": {
                    "model_name": self.model_name,
                    "task": self.task,
                    "device": self.device,
                    "precision": self.precision,
                },
            }

    def unload_model(self):
        """Unload model to free memory."""
        if not self.is_loaded:
            return

        try:
            logger.info("Unloading model", model_name=self.model_name)

            # Clear model and tokenizer
            del self.model
            del self.tokenizer
            del self.accelerator

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = None
            self.tokenizer = None
            self.accelerator = None
            self.model_info = None
            self.is_loaded = False

            logger.info("Model unloaded successfully")

        except Exception as e:
            logger.error("Error unloading model", error=str(e))

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
