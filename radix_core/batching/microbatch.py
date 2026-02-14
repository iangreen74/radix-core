"""
Microbatching for Memory-Efficient Processing

This module provides intelligent microbatching that fragments large batches
based on tensor size estimation and memory constraints.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..config import get_config
from ..dryrun import DryRunGuard
from ..logging import get_logger
from ..utils.timers import time_operation

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class MemoryEstimate:
    """Memory usage estimate for a batch."""

    input_size_mb: float
    intermediate_size_mb: float
    output_size_mb: float
    peak_size_mb: float

    @property
    def total_size_mb(self) -> float:
        """Total memory usage."""
        return self.input_size_mb + self.intermediate_size_mb + self.output_size_mb


class TensorSizeEstimator(ABC):
    """Abstract base class for estimating tensor memory usage."""

    @abstractmethod
    def estimate_input_size(self, data: Any) -> float:
        """Estimate input tensor size in MB."""
        pass

    @abstractmethod
    def estimate_output_size(self, data: Any) -> float:
        """Estimate output tensor size in MB."""
        pass

    @abstractmethod
    def estimate_intermediate_size(self, data: Any) -> float:
        """Estimate intermediate computation size in MB."""
        pass

    def estimate_batch_memory(self, batch: List[T]) -> MemoryEstimate:
        """Estimate total memory usage for a batch."""
        input_size = sum(self.estimate_input_size(item) for item in batch)
        output_size = sum(self.estimate_output_size(item) for item in batch)
        intermediate_size = sum(self.estimate_intermediate_size(item) for item in batch)

        # Peak memory is typically input + intermediate + output
        peak_size = input_size + intermediate_size + output_size

        return MemoryEstimate(
            input_size_mb=input_size,
            intermediate_size_mb=intermediate_size,
            output_size_mb=output_size,
            peak_size_mb=peak_size,
        )


class TextTensorEstimator(TensorSizeEstimator):
    """Tensor size estimator for text processing."""

    def __init__(self, model_name: str = "gpt2", precision: str = "fp16"):
        self.model_name = model_name
        self.precision = precision

        # Bytes per parameter based on precision
        self.bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(precision, 2)

        # Model-specific parameters (rough estimates)
        self.model_params = {
            "gpt2": {"hidden_size": 768, "vocab_size": 50257, "layers": 12},
            "gpt2-medium": {"hidden_size": 1024, "vocab_size": 50257, "layers": 24},
            "gpt2-large": {"hidden_size": 1280, "vocab_size": 50257, "layers": 36},
            "gpt2-xl": {"hidden_size": 1600, "vocab_size": 50257, "layers": 48},
        }.get(model_name, {"hidden_size": 768, "vocab_size": 50257, "layers": 12})

    def estimate_input_size(self, text: str) -> float:
        """Estimate input tensor size for text."""
        # Rough tokenization estimate (4 chars per token average)
        token_count = len(text) // 4 + 1

        # Input embeddings: token_count * hidden_size * bytes_per_param
        embedding_size = token_count * self.model_params["hidden_size"] * self.bytes_per_param

        return embedding_size / (1024 * 1024)  # Convert to MB

    def estimate_output_size(self, text: str) -> float:
        """Estimate output tensor size for text."""
        token_count = len(text) // 4 + 1

        # Output logits: token_count * vocab_size * bytes_per_param
        logits_size = token_count * self.model_params["vocab_size"] * self.bytes_per_param

        return logits_size / (1024 * 1024)  # Convert to MB

    def estimate_intermediate_size(self, text: str) -> float:
        """Estimate intermediate computation size for text."""
        token_count = len(text) // 4 + 1
        hidden_size = self.model_params["hidden_size"]
        layers = self.model_params["layers"]

        # Rough estimate: attention matrices + feed-forward activations
        # Attention: token_count^2 * num_heads * head_dim
        attention_size = token_count * token_count * hidden_size * self.bytes_per_param

        # Feed-forward: token_count * hidden_size * 4 (typical FF expansion)
        ff_size = token_count * hidden_size * 4 * self.bytes_per_param

        # Total intermediate per layer
        per_layer_size = attention_size + ff_size

        # Assume we need memory for a few layers simultaneously
        total_intermediate = per_layer_size * min(layers, 4)

        return total_intermediate / (1024 * 1024)  # Convert to MB


class EmbeddingTensorEstimator(TensorSizeEstimator):
    """Tensor size estimator for embedding models."""

    def __init__(self, embedding_dim: int = 768, precision: str = "fp16"):
        self.embedding_dim = embedding_dim
        self.bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(precision, 2)

    def estimate_input_size(self, text: str) -> float:
        """Estimate input size for embedding."""
        token_count = len(text) // 4 + 1
        input_size = token_count * self.embedding_dim * self.bytes_per_param
        return input_size / (1024 * 1024)

    def estimate_output_size(self, text: str) -> float:
        """Estimate output embedding size."""
        # Single embedding vector per input
        output_size = self.embedding_dim * self.bytes_per_param
        return output_size / (1024 * 1024)

    def estimate_intermediate_size(self, text: str) -> float:
        """Estimate intermediate computation size."""
        token_count = len(text) // 4 + 1
        # Rough estimate for transformer intermediate activations
        intermediate_size = token_count * self.embedding_dim * 2 * self.bytes_per_param
        return intermediate_size / (1024 * 1024)


@dataclass
class MicrobatchResult(Generic[R]):
    """Result of microbatch processing."""

    microbatch_id: str
    results: List[R]
    processing_time_ms: float
    memory_used_mb: float
    success: bool
    error: Optional[str] = None


class MicrobatchProcessor(Generic[T, R]):
    """
    Processor that intelligently fragments batches into microbatches
    based on memory constraints and tensor size estimation.
    """

    def __init__(
        self,
        processor: Callable[[List[T]], List[R]],
        size_estimator: TensorSizeEstimator,
        max_memory_mb: float = None,
        min_microbatch_size: int = 1,
        max_microbatch_size: int = None,
    ):
        """
        Initialize microbatch processor.

        Args:
            processor: Function to process microbatches
            size_estimator: Estimator for tensor memory usage
            max_memory_mb: Maximum memory per microbatch (from config if None)
            min_microbatch_size: Minimum microbatch size
            max_microbatch_size: Maximum microbatch size (from config if None)
        """
        self.config = get_config()
        self.processor = processor
        self.size_estimator = size_estimator

        # Memory configuration
        self.max_memory_mb = max_memory_mb or self._estimate_available_memory()
        self.min_microbatch_size = min_microbatch_size
        self.max_microbatch_size = max_microbatch_size or getattr(
            getattr(self.config, "batching", None), "microbatch_size", 8
        )

        # Statistics
        self.total_batches = 0
        self.total_microbatches = 0
        self.total_processing_time = 0.0
        self.memory_savings = 0.0

    def _estimate_available_memory(self) -> float:
        """Estimate available memory for microbatching."""
        try:
            import psutil

            # Use 25% of available system memory as a conservative estimate
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            return float(available_mb * 0.25)
        except ImportError:
            # Fallback to conservative estimate
            return 1024.0  # 1GB default

    @DryRunGuard.protect
    def process_batch(self, batch: List[T]) -> List[R]:
        """
        Process a batch by fragmenting into optimal microbatches.

        Args:
            batch: Batch to process

        Returns:
            Combined results from all microbatches
        """
        if not batch:
            return []

        with time_operation(f"microbatch_processing_{len(batch)}"):
            # Fragment batch into microbatches
            microbatches = self._fragment_batch(batch)

            logger.info(
                "Processing batch with microbatches",
                batch_size=len(batch),
                num_microbatches=len(microbatches),
                max_memory_mb=self.max_memory_mb,
            )

            # Process each microbatch
            all_results = []
            total_processing_time = 0.0

            for i, microbatch in enumerate(microbatches):
                try:
                    result = self._process_microbatch(f"mb_{i}", microbatch)

                    if result.success:
                        all_results.extend(result.results)
                        total_processing_time += result.processing_time_ms
                    else:
                        logger.error(
                            "Microbatch processing failed",
                            microbatch_id=result.microbatch_id,
                            error=result.error,
                        )
                        raise RuntimeError(
                            f"Microbatch {result.microbatch_id} failed: {result.error}"
                        )

                except Exception as e:
                    logger.error(
                        "Error processing microbatch",
                        microbatch_index=i,
                        microbatch_size=len(microbatch),
                        error=str(e),
                    )
                    raise

            # Update statistics
            self.total_batches += 1
            self.total_microbatches += len(microbatches)
            self.total_processing_time += total_processing_time

            logger.info(
                "Batch processing completed",
                batch_size=len(batch),
                num_microbatches=len(microbatches),
                total_processing_time_ms=total_processing_time,
                results_count=len(all_results),
            )

            return all_results

    def _fragment_batch(self, batch: List[T]) -> List[List[T]]:
        """Fragment batch into memory-efficient microbatches."""
        if len(batch) <= self.min_microbatch_size:
            return [batch]

        microbatches = []
        current_microbatch: List[T] = []
        current_memory = 0.0

        for item in batch:
            # Estimate memory for this item
            item_memory = self.size_estimator.estimate_batch_memory([item]).peak_size_mb

            # Check if adding this item would exceed memory limit
            if (
                current_memory + item_memory > self.max_memory_mb
                and len(current_microbatch) >= self.min_microbatch_size
            ):

                # Start new microbatch
                microbatches.append(current_microbatch)
                current_microbatch = [item]
                current_memory = item_memory

            elif (
                self.max_microbatch_size is not None
                and len(current_microbatch) >= self.max_microbatch_size
            ):
                # Microbatch size limit reached
                microbatches.append(current_microbatch)
                current_microbatch = [item]
                current_memory = item_memory

            else:
                # Add to current microbatch
                current_microbatch.append(item)
                current_memory += item_memory

        # Add final microbatch
        if current_microbatch:
            microbatches.append(current_microbatch)

        # Log fragmentation details
        sizes = [len(mb) for mb in microbatches]
        memory_estimates = [
            self.size_estimator.estimate_batch_memory(mb).peak_size_mb for mb in microbatches
        ]

        logger.debug(
            "Batch fragmented into microbatches",
            original_size=len(batch),
            num_microbatches=len(microbatches),
            microbatch_sizes=sizes,
            memory_estimates_mb=memory_estimates,
        )

        return microbatches

    def _process_microbatch(self, microbatch_id: str, microbatch: List[T]) -> MicrobatchResult[R]:
        """Process a single microbatch."""
        start_time = time.time()

        try:
            # Estimate memory usage
            memory_estimate = self.size_estimator.estimate_batch_memory(microbatch)

            logger.debug(
                "Processing microbatch",
                microbatch_id=microbatch_id,
                size=len(microbatch),
                estimated_memory_mb=memory_estimate.peak_size_mb,
            )

            # Process the microbatch
            results = self.processor(microbatch)

            processing_time_ms = (time.time() - start_time) * 1000

            return MicrobatchResult(
                microbatch_id=microbatch_id,
                results=results,
                processing_time_ms=processing_time_ms,
                memory_used_mb=memory_estimate.peak_size_mb,
                success=True,
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return MicrobatchResult(
                microbatch_id=microbatch_id,
                results=[],
                processing_time_ms=processing_time_ms,
                memory_used_mb=0.0,
                success=False,
                error=str(e),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get microbatch processor statistics."""
        avg_processing_time = (
            self.total_processing_time / self.total_microbatches
            if self.total_microbatches > 0
            else 0
        )

        avg_microbatches_per_batch = (
            self.total_microbatches / self.total_batches if self.total_batches > 0 else 0
        )

        return {
            "total_batches": self.total_batches,
            "total_microbatches": self.total_microbatches,
            "avg_microbatches_per_batch": avg_microbatches_per_batch,
            "avg_processing_time_ms": avg_processing_time,
            "total_processing_time_ms": self.total_processing_time,
            "config": {
                "max_memory_mb": self.max_memory_mb,
                "min_microbatch_size": self.min_microbatch_size,
                "max_microbatch_size": self.max_microbatch_size,
            },
        }

    def reset_stats(self):
        """Reset processor statistics."""
        self.total_batches = 0
        self.total_microbatches = 0
        self.total_processing_time = 0.0
        self.memory_savings = 0.0


def create_text_microbatch_processor(
    processor: Callable[[List[str]], List[str]],
    model_name: str = "gpt2",
    precision: str = "fp16",
    max_memory_mb: float = None,
) -> MicrobatchProcessor[str, str]:
    """Create a microbatch processor optimized for text processing."""

    estimator = TextTensorEstimator(model_name=model_name, precision=precision)

    return MicrobatchProcessor(
        processor=processor, size_estimator=estimator, max_memory_mb=max_memory_mb
    )


def create_embedding_microbatch_processor(
    processor: Callable[[List[str]], List[List[float]]],
    embedding_dim: int = 768,
    precision: str = "fp16",
    max_memory_mb: float = None,
) -> MicrobatchProcessor[str, List[float]]:
    """Create a microbatch processor optimized for embedding generation."""

    estimator = EmbeddingTensorEstimator(embedding_dim=embedding_dim, precision=precision)

    return MicrobatchProcessor(
        processor=processor, size_estimator=estimator, max_memory_mb=max_memory_mb
    )
