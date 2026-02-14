"""
Dynamic Batching with Latency SLA Awareness

This module implements intelligent batching that adapts to workload patterns
while respecting latency SLAs and memory constraints.
"""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..config import get_config
from ..dryrun import DryRunGuard
from ..logging import get_logger
from ..utils.randfail import seeded_failure
from ..utils.timers import time_operation

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchStatus(Enum):
    """Status of a batch request."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class BatchRequest(Generic[T]):
    """A request to be batched."""

    request_id: str
    data: T
    arrival_time: float
    max_latency_ms: Optional[int] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_ms(self) -> float:
        """Age of request in milliseconds."""
        return (time.time() - self.arrival_time) * 1000

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded max latency."""
        if self.max_latency_ms is None:
            return False
        return self.age_ms > self.max_latency_ms


@dataclass
class BatchResult(Generic[R]):
    """Result of batch processing."""

    batch_id: str
    results: List[R]
    request_ids: List[str]
    processing_time_ms: float
    batch_size: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicBatcher(Generic[T, R]):
    """
    Dynamic batcher that adapts to workload patterns while respecting SLAs.

    Features:
    - Latency-aware batching with configurable SLAs
    - Adaptive batch sizing based on throughput
    - Memory-aware microbatching
    - Priority-based request ordering
    - Comprehensive metrics and monitoring
    """

    def __init__(
        self,
        processor: Callable[[List[T]], List[R]],
        max_batch_size: int = None,
        max_latency_ms: int = None,
        min_batch_size: int = 1,
        adaptive_sizing: bool = True,
    ):
        """
        Initialize dynamic batcher.

        Args:
            processor: Function to process batches
            max_batch_size: Maximum batch size (from config if None)
            max_latency_ms: Maximum latency in ms (from config if None)
            min_batch_size: Minimum batch size before processing
            adaptive_sizing: Enable adaptive batch sizing
        """
        self.config = get_config()
        self.processor = processor

        # Batch configuration
        self.max_batch_size = max_batch_size or getattr(
            self.config.batching, "default_batch_size", 32
        )
        self.max_latency_ms = max_latency_ms or 5000  # 5 second default
        self.min_batch_size = min_batch_size
        self.adaptive_sizing = adaptive_sizing

        # Request queues
        self.pending_requests: deque = deque()
        self.priority_queues: Dict[int, deque] = {}

        # Processing state
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=self.config.execution.max_parallelism)

        # Adaptive sizing state
        self.recent_throughputs: deque = deque(maxlen=100)
        self.recent_latencies: deque = deque(maxlen=100)
        self.optimal_batch_size = (self.max_batch_size or 32) // 2

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_processing_time = 0.0
        self.sla_violations = 0

        # Thread safety
        self.lock = threading.RLock()

    def start(self):
        """Start the batcher processing loop."""
        with self.lock:
            if self.is_running:
                return

            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

            logger.info(
                "Dynamic batcher started",
                max_batch_size=self.max_batch_size,
                max_latency_ms=self.max_latency_ms,
                adaptive_sizing=self.adaptive_sizing,
            )

    def stop(self):
        """Stop the batcher processing loop."""
        with self.lock:
            if not self.is_running:
                return

            self.is_running = False

            if self.processing_thread:
                self.processing_thread.join(timeout=5.0)

            self.executor.shutdown(wait=True)

            logger.info("Dynamic batcher stopped")

    def submit_request(self, request: BatchRequest[T]) -> str:
        """
        Submit a request for batching.

        Args:
            request: Request to batch

        Returns:
            Request ID for tracking
        """
        with self.lock:
            # Add to appropriate priority queue
            priority = request.priority
            if priority not in self.priority_queues:
                self.priority_queues[priority] = deque()

            self.priority_queues[priority].append(request)
            self.total_requests += 1

            logger.debug(
                "Request submitted for batching",
                request_id=request.request_id,
                priority=priority,
                queue_size=len(self.priority_queues[priority]),
            )

            return request.request_id

    def _processing_loop(self):
        """Main processing loop for batching."""
        logger.debug("Batcher processing loop started")

        while self.is_running:
            try:
                # Check for failure injection
                seeded_failure("batch_processing")

                batch = self._collect_batch()
                if batch:
                    self._process_batch_async(batch)
                else:
                    # No batch ready, sleep briefly
                    time.sleep(0.001)  # 1ms

            except Exception as e:
                logger.error("Error in batcher processing loop", error=str(e))
                time.sleep(0.1)  # Back off on errors

        logger.debug("Batcher processing loop stopped")

    def _collect_batch(self) -> Optional[List[BatchRequest[T]]]:
        """Collect requests into a batch based on current strategy."""
        with self.lock:
            if not self.priority_queues:
                return None

            batch: List[BatchRequest[T]] = []
            target_batch_size = self._get_target_batch_size()

            # Process queues by priority (highest first)
            for priority in sorted(self.priority_queues.keys(), reverse=True):
                queue = self.priority_queues[priority]

                while queue and len(batch) < target_batch_size:
                    request = queue[0]  # Peek at oldest request

                    # Check if request should be processed now
                    should_process = (
                        len(batch) >= self.min_batch_size
                        or request.age_ms >= self.max_latency_ms
                        or request.is_expired
                    )

                    if should_process:
                        batch.append(queue.popleft())
                    else:
                        break  # Wait for more requests or timeout

                # Clean up empty queues
                if not queue:
                    del self.priority_queues[priority]

            # Check SLA violations
            for request in batch:
                if request.is_expired:
                    self.sla_violations += 1
                    logger.warning(
                        "SLA violation detected",
                        request_id=request.request_id,
                        age_ms=request.age_ms,
                        max_latency_ms=request.max_latency_ms,
                    )

            return batch if batch else None

    def _get_target_batch_size(self) -> int:
        """Calculate target batch size based on adaptive strategy."""
        if not self.adaptive_sizing:
            return self.max_batch_size or 32

        # Use recent performance to adjust batch size
        if len(self.recent_throughputs) < 10:
            return self.optimal_batch_size or 16

        # Calculate recent average latency
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)

        # Adjust batch size based on performance
        if avg_latency > self.max_latency_ms * 0.8:  # Approaching SLA limit
            self.optimal_batch_size = max(self.min_batch_size, int(self.optimal_batch_size * 0.9))
        elif avg_latency < self.max_latency_ms * 0.5:  # Well under SLA
            self.optimal_batch_size = min(
                self.max_batch_size or 32, int(self.optimal_batch_size * 1.1)
            )

        return self.optimal_batch_size

    def _process_batch_async(self, batch: List[BatchRequest[T]]):
        """Process batch asynchronously."""
        batch_id = f"batch_{int(time.time() * 1000)}"

        # Submit to thread pool for processing
        self.executor.submit(self._process_batch, batch_id, batch)

        # Don't wait for completion in the main loop
        logger.debug("Batch submitted for processing", batch_id=batch_id, batch_size=len(batch))

    @DryRunGuard.protect
    def _process_batch(self, batch_id: str, batch: List[BatchRequest[T]]) -> BatchResult[R]:
        """Process a batch of requests."""
        start_time = time.time()

        try:
            with time_operation(f"batch_processing_{len(batch)}"):
                # Extract data from requests
                batch_data = [req.data for req in batch]
                request_ids = [req.request_id for req in batch]

                logger.info(
                    "Processing batch",
                    batch_id=batch_id,
                    batch_size=len(batch),
                    request_ids=request_ids[:5],
                )  # Log first 5 IDs

                # Process the batch
                results = self.processor(batch_data)

                processing_time_ms = (time.time() - start_time) * 1000

                # Update metrics
                with self.lock:
                    self.total_batches += 1
                    self.total_processing_time += processing_time_ms

                    # Update adaptive sizing metrics
                    throughput = len(batch) / (processing_time_ms / 1000)
                    self.recent_throughputs.append(throughput)
                    self.recent_latencies.append(processing_time_ms)

                result = BatchResult(
                    batch_id=batch_id,
                    results=results,
                    request_ids=request_ids,
                    processing_time_ms=processing_time_ms,
                    batch_size=len(batch),
                    success=True,
                    metadata={
                        "throughput_items_per_sec": throughput,
                        "avg_latency_ms": processing_time_ms / len(batch),
                    },
                )

                logger.info(
                    "Batch processed successfully",
                    batch_id=batch_id,
                    processing_time_ms=processing_time_ms,
                    throughput=throughput,
                )

                return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            logger.error(
                "Batch processing failed",
                batch_id=batch_id,
                error=str(e),
                processing_time_ms=processing_time_ms,
            )

            return BatchResult(
                batch_id=batch_id,
                results=[],
                request_ids=[req.request_id for req in batch],
                processing_time_ms=processing_time_ms,
                batch_size=len(batch),
                success=False,
                error=str(e),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        with self.lock:
            queue_sizes = {priority: len(queue) for priority, queue in self.priority_queues.items()}

            avg_processing_time = (
                self.total_processing_time / self.total_batches if self.total_batches > 0 else 0
            )

            avg_throughput = (
                sum(self.recent_throughputs) / len(self.recent_throughputs)
                if self.recent_throughputs
                else 0
            )

            return {
                "is_running": self.is_running,
                "total_requests": self.total_requests,
                "total_batches": self.total_batches,
                "sla_violations": self.sla_violations,
                "sla_violation_rate": self.sla_violations / max(self.total_requests, 1),
                "avg_processing_time_ms": avg_processing_time,
                "avg_throughput_items_per_sec": avg_throughput,
                "current_batch_size_target": self.optimal_batch_size,
                "queue_sizes": queue_sizes,
                "total_queued": sum(queue_sizes.values()),
                "config": {
                    "max_batch_size": self.max_batch_size,
                    "max_latency_ms": self.max_latency_ms,
                    "min_batch_size": self.min_batch_size,
                    "adaptive_sizing": self.adaptive_sizing,
                },
            }

    def reset_stats(self):
        """Reset batcher statistics."""
        with self.lock:
            self.total_requests = 0
            self.total_batches = 0
            self.total_processing_time = 0.0
            self.sla_violations = 0
            self.recent_throughputs.clear()
            self.recent_latencies.clear()

        logger.info("Batcher statistics reset")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class BatchingStrategy(Enum):
    """Different batching strategies."""

    LATENCY_FIRST = "latency_first"  # Prioritize meeting latency SLAs
    THROUGHPUT_FIRST = "throughput_first"  # Prioritize maximum throughput
    BALANCED = "balanced"  # Balance latency and throughput
    ADAPTIVE = "adaptive"  # Adapt based on workload patterns


def create_text_batcher(
    processor: Callable[[List[str]], List[str]],
    strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE,
) -> DynamicBatcher[str, str]:
    """Create a batcher optimized for text processing."""
    config = get_config()

    batch_latency_ms = getattr(config.batching, "batch_latency_ms", 100)
    max_batch = getattr(config.batching, "max_batch_size", config.batching.default_batch_size)

    # Strategy-specific configurations
    if strategy == BatchingStrategy.LATENCY_FIRST:
        max_batch_size = min(16, max_batch)
        max_latency_ms = batch_latency_ms // 2
        adaptive_sizing = False
    elif strategy == BatchingStrategy.THROUGHPUT_FIRST:
        max_batch_size = max_batch
        max_latency_ms = batch_latency_ms * 2
        adaptive_sizing = False
    else:  # BALANCED or ADAPTIVE
        max_batch_size = max_batch
        max_latency_ms = batch_latency_ms
        adaptive_sizing = strategy == BatchingStrategy.ADAPTIVE

    return DynamicBatcher(
        processor=processor,
        max_batch_size=max_batch_size,
        max_latency_ms=max_latency_ms,
        adaptive_sizing=adaptive_sizing,
    )
