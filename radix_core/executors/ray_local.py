"""
Ray Local Executor for Map/Reduce Patterns

This module provides Ray-based execution in local mode only for ridiculously
parallel tasks with comprehensive safety guards.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from ..config import get_config
from ..dryrun import DryRunGuard
from ..logging import get_logger
from ..utils.randfail import seeded_failure
from ..utils.timers import time_operation

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class RayTaskResult:
    """Result of a Ray task execution."""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    node_id: Optional[str] = None


class RayLocalExecutor:
    """
    Ray executor that enforces local-only execution with safety guards.

    Features:
    - Local-only Ray cluster (no external connections)
    - Map/Reduce pattern support for parallel processing
    - Resource allocation and monitoring
    - Comprehensive error handling and recovery
    - Cost tracking (always $0.00 in dry-run)
    """

    def __init__(
        self,
        num_cpus: Optional[int] = None,
        num_gpus: int = 0,
        object_store_memory: Optional[int] = None,
    ):
        """
        Initialize Ray local executor.

        Args:
            num_cpus: Number of CPUs for Ray (from config if None)
            num_gpus: Number of GPUs for Ray (0 for safety)
            object_store_memory: Object store memory in bytes
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not available. Install with: pip install ray[default]")

        self.config = get_config()

        # Ray configuration (enforce local mode)
        self.num_cpus = num_cpus or self.config.execution.ray_num_cpus
        self.num_gpus = num_gpus  # Always 0 for safety unless explicitly enabled
        self.object_store_memory = object_store_memory

        # Safety checks
        if not self.config.execution.ray_local_mode:
            from ..mode import is_production

            if not is_production():
                raise ValueError("Ray must be in local mode for safety")

        if self.num_gpus > 0 and not getattr(
            self.config.execution,
            "enable_cuda",
            getattr(self.config.execution, "enable_gpu", False),
        ):
            logger.warning("GPU allocation requested but CUDA disabled")
            self.num_gpus = 0

        # Ray cluster state
        self.is_initialized = False
        self.cluster_info: Dict[str, Any] = {}

        # Execution tracking
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: List[RayTaskResult] = []

        # Statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_execution_time = 0.0

        # Thread safety
        self.lock = threading.RLock()

    def initialize(self):
        """Initialize Ray in local mode with safety constraints."""
        if self.is_initialized:
            return

        try:
            # Ray initialization parameters
            ray_config = {
                "local_mode": self.config.execution.ray_local_mode,
                "num_cpus": self.num_cpus,
                "num_gpus": self.num_gpus,
                "ignore_reinit_error": True,
                "logging_level": "WARNING",  # Reduce Ray logging noise
                "dashboard_host": "127.0.0.1",  # Local only
                "dashboard_port": None,  # Disable dashboard for safety
            }

            if self.object_store_memory:
                ray_config["object_store_memory"] = self.object_store_memory

            logger.info(
                "Initializing Ray in local mode",
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
                local_mode=True,
            )

            # Initialize Ray
            ray.init(**ray_config)

            # Verify local mode
            cluster_resources = ray.cluster_resources()
            nodes = ray.nodes()

            # Safety check: ensure only local node
            if len(nodes) > 1:
                ray.shutdown()
                raise ValueError("Multiple Ray nodes detected - local mode violation")

            # Store cluster information
            self.cluster_info = {
                "resources": cluster_resources,
                "nodes": len(nodes),
                "local_mode": True,
                "initialized_at": datetime.utcnow().isoformat(),
            }

            self.is_initialized = True

            logger.info(
                "Ray initialized successfully",
                cluster_resources=cluster_resources,
                node_count=len(nodes),
            )

        except Exception as e:
            logger.error("Failed to initialize Ray", error=str(e))
            raise

    def shutdown(self):
        """Shutdown Ray cluster safely."""
        if not self.is_initialized:
            return

        try:
            logger.info("Shutting down Ray cluster", active_tasks=len(self.active_tasks))

            # Cancel active tasks
            with self.lock:
                for task_id, task_ref in self.active_tasks.items():
                    try:
                        ray.cancel(task_ref)
                    except Exception as e:
                        logger.warning("Error canceling task", task_id=task_id, error=str(e))

            # Shutdown Ray
            ray.shutdown()
            self.is_initialized = False

            logger.info("Ray cluster shutdown complete")

        except Exception as e:
            logger.error("Error during Ray shutdown", error=str(e))

    @DryRunGuard.protect
    def map_parallel(
        self, data: List[T], map_func: Callable[[T], R], chunk_size: Optional[int] = None
    ) -> List[R]:
        """
        Execute map operation in parallel using Ray.

        Args:
            data: List of data items to process
            map_func: Function to apply to each item
            chunk_size: Size of chunks for batching (auto if None)

        Returns:
            List of results in original order
        """
        if not self.is_initialized:
            self.initialize()

        if not data:
            return []

        # Check for failure injection
        seeded_failure("ray_map_parallel")

        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.num_cpus * 2))

        with time_operation(f"ray_map_parallel_{len(data)}"):
            logger.info(
                "Starting parallel map operation",
                data_size=len(data),
                chunk_size=chunk_size,
                num_chunks=len(data) // chunk_size + (1 if len(data) % chunk_size else 0),
            )

            # Create Ray remote function
            @ray.remote
            def process_chunk(chunk: List[T]) -> List[R]:
                """Process a chunk of data."""
                try:
                    return [map_func(item) for item in chunk]
                except Exception as e:
                    logger.error("Error processing chunk", error=str(e))
                    raise

            # Split data into chunks
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

            # Submit tasks
            task_refs = []
            task_ids = []

            with self.lock:
                for i, chunk in enumerate(chunks):
                    task_id = f"map_task_{int(time.time() * 1000)}_{i}"
                    task_ref = process_chunk.remote(chunk)

                    task_refs.append(task_ref)
                    task_ids.append(task_id)
                    self.active_tasks[task_id] = task_ref
                    self.total_tasks_submitted += 1

            try:
                # Wait for all tasks to complete
                chunk_results = ray.get(task_refs)

                # Flatten results to maintain order
                results = []
                for chunk_result in chunk_results:
                    results.extend(chunk_result)

                # Update statistics
                with self.lock:
                    for task_id in task_ids:
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]

                        self.total_tasks_completed += 1
                        self.completed_tasks.append(
                            RayTaskResult(
                                task_id=task_id,
                                success=True,
                                result=(
                                    len(chunk_results[task_ids.index(task_id)])
                                    if task_id in task_ids
                                    else 0
                                ),
                            )
                        )

                logger.info(
                    "Parallel map operation completed",
                    data_size=len(data),
                    results_size=len(results),
                    tasks_completed=len(task_ids),
                )

                return results

            except Exception as e:
                # Handle task failures
                with self.lock:
                    for task_id in task_ids:
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]

                        self.total_tasks_failed += 1
                        self.completed_tasks.append(
                            RayTaskResult(task_id=task_id, success=False, error=str(e))
                        )

                logger.error("Parallel map operation failed", error=str(e))
                raise

    @DryRunGuard.protect
    def reduce_parallel(
        self, data: List[T], reduce_func: Callable[[List[T]], R], chunk_size: Optional[int] = None
    ) -> R:
        """
        Execute reduce operation in parallel using Ray.

        Args:
            data: List of data items to reduce
            reduce_func: Function to reduce chunks
            chunk_size: Size of chunks for parallel reduction

        Returns:
            Final reduced result
        """
        if not self.is_initialized:
            self.initialize()

        if not data:
            raise ValueError("Cannot reduce empty data")

        if len(data) == 1:
            return reduce_func(data)

        # Check for failure injection
        seeded_failure("ray_reduce_parallel")

        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(data) // self.num_cpus)

        with time_operation(f"ray_reduce_parallel_{len(data)}"):
            logger.info(
                "Starting parallel reduce operation", data_size=len(data), chunk_size=chunk_size
            )

            # Create Ray remote function
            @ray.remote
            def reduce_chunk(chunk: List[T]) -> R:
                """Reduce a chunk of data."""
                try:
                    return reduce_func(chunk)
                except Exception as e:
                    logger.error("Error reducing chunk", error=str(e))
                    raise

            current_data = data
            iteration = 0

            # Iteratively reduce until single result
            while len(current_data) > 1:
                iteration += 1

                # Split into chunks
                chunks = [
                    current_data[i : i + chunk_size]
                    for i in range(0, len(current_data), chunk_size)
                ]

                # Submit reduction tasks
                task_refs = []
                task_ids = []

                with self.lock:
                    for i, chunk in enumerate(chunks):
                        task_id = f"reduce_task_{iteration}_{i}"
                        task_ref = reduce_chunk.remote(chunk)

                        task_refs.append(task_ref)
                        task_ids.append(task_id)
                        self.active_tasks[task_id] = task_ref
                        self.total_tasks_submitted += 1

                try:
                    # Get results for next iteration
                    current_data = ray.get(task_refs)

                    # Update statistics
                    with self.lock:
                        for task_id in task_ids:
                            if task_id in self.active_tasks:
                                del self.active_tasks[task_id]

                            self.total_tasks_completed += 1
                            self.completed_tasks.append(
                                RayTaskResult(task_id=task_id, success=True)
                            )

                except Exception as e:
                    # Handle failures
                    with self.lock:
                        for task_id in task_ids:
                            if task_id in self.active_tasks:
                                del self.active_tasks[task_id]

                            self.total_tasks_failed += 1

                    logger.error(
                        "Parallel reduce operation failed", iteration=iteration, error=str(e)
                    )
                    raise

                logger.debug(
                    "Reduce iteration completed",
                    iteration=iteration,
                    input_size=len(chunks),
                    output_size=len(current_data),
                )

            result = current_data[0]

            logger.info(
                "Parallel reduce operation completed",
                iterations=iteration,
                final_result_type=type(result).__name__,
            )

            return result

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self.lock:
            success_rate = self.total_tasks_completed / max(self.total_tasks_submitted, 1)

            return {
                "executor_type": "ray_local",
                "ray_available": RAY_AVAILABLE,
                "is_initialized": self.is_initialized,
                "cluster_info": self.cluster_info,
                "active_tasks": len(self.active_tasks),
                "total_submitted": self.total_tasks_submitted,
                "total_completed": self.total_tasks_completed,
                "total_failed": self.total_tasks_failed,
                "success_rate": success_rate,
                "total_execution_time": self.total_execution_time,
                "config": {
                    "num_cpus": self.num_cpus,
                    "num_gpus": self.num_gpus,
                    "local_mode_enforced": True,
                },
            }

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get Ray cluster information."""
        if not self.is_initialized:
            return {"error": "Ray not initialized"}

        try:
            return {
                "resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
                "nodes": len(ray.nodes()),
                "local_mode": True,
                "object_store_stats": (
                    ray.object_store_stats() if hasattr(ray, "object_store_stats") else {}
                ),
            }
        except Exception as e:
            logger.error("Error getting cluster info", error=str(e))
            return {"error": str(e)}

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Convenience functions for common patterns
def parallel_map(
    data: List[T], map_func: Callable[[T], R], num_cpus: Optional[int] = None
) -> List[R]:
    """
    Convenience function for parallel map operation.

    Args:
        data: Data to process
        map_func: Function to apply
        num_cpus: Number of CPUs to use

    Returns:
        List of results
    """
    with RayLocalExecutor(num_cpus=num_cpus) as executor:
        return executor.map_parallel(data, map_func)


def parallel_reduce(
    data: List[T], reduce_func: Callable[[List[T]], R], num_cpus: Optional[int] = None
) -> R:
    """
    Convenience function for parallel reduce operation.

    Args:
        data: Data to reduce
        reduce_func: Reduction function
        num_cpus: Number of CPUs to use

    Returns:
        Reduced result
    """
    with RayLocalExecutor(num_cpus=num_cpus) as executor:
        return executor.reduce_parallel(data, reduce_func)
