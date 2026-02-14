"""
ThreadPool Executor for Local CPU/GPU Work

This module provides a thread pool executor that respects parallelism limits,
cost guards, and safety constraints while providing comprehensive monitoring.
"""

import queue
import threading
import time
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as StdThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..config import get_config
from ..dryrun import DryRunGuard
from ..logging import get_logger
from ..types import Job, JobResult, JobStatus
from ..utils.randfail import seeded_failure
from ..utils.timers import time_operation

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ExecutionContext:
    """Context information for job execution."""

    job_id: str
    thread_id: int
    start_time: datetime
    estimated_duration: float
    resource_allocation: Dict[str, Any]
    metadata: Dict[str, Any]


class ThreadPoolExecutor(Generic[T, R]):
    """
    Enhanced ThreadPool executor with safety guards and monitoring.

    Features:
    - Respects MAX_PARALLELISM configuration
    - Cost tracking and caps enforcement
    - Comprehensive execution monitoring
    - Failure injection support
    - Resource usage tracking
    - Graceful shutdown handling
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "RadixWorker",
        enable_monitoring: bool = True,
    ):
        """
        Initialize thread pool executor.

        Args:
            max_workers: Maximum number of worker threads (from config if None)
            thread_name_prefix: Prefix for worker thread names
            enable_monitoring: Enable detailed execution monitoring
        """
        self.config = get_config()

        # Thread pool configuration
        self.max_workers = max_workers or self.config.execution.max_parallelism
        self.thread_name_prefix = thread_name_prefix
        self.enable_monitoring = enable_monitoring

        # Create underlying thread pool
        self.executor = StdThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix=thread_name_prefix
        )

        # Execution tracking
        self.active_jobs: Dict[str, ExecutionContext] = {}
        self.completed_jobs: List[JobResult] = []
        self.job_queue = queue.Queue()

        # Statistics
        self.total_jobs_submitted = 0
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0
        self.total_execution_time = 0.0
        self.peak_concurrent_jobs = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            "ThreadPool executor initialized",
            max_workers=self.max_workers,
            monitoring_enabled=enable_monitoring,
        )

    @DryRunGuard.protect
    def submit_job(self, job: Job, processor: Callable[[T], R]) -> Future[JobResult]:
        """
        Submit a job for execution.

        Args:
            job: Job to execute
            processor: Function to process the job

        Returns:
            Future for the job result
        """
        with self.lock:
            self.total_jobs_submitted += 1

            # Check cost caps before submission
            estimated_cost = self._estimate_job_cost(job)
            if estimated_cost > self.config.safety.max_job_cost_usd:
                raise ValueError(
                    f"Job cost ${estimated_cost:.2f} exceeds cap ${self.config.safety.max_job_cost_usd:.2f}"
                )

            # Create execution context
            context = ExecutionContext(
                job_id=job.job_id,
                thread_id=-1,  # Will be set by worker
                start_time=datetime.utcnow(),
                estimated_duration=job.estimated_duration(),
                resource_allocation=self._allocate_resources(job),
                metadata={"estimated_cost_usd": estimated_cost},
            )

            # Submit to thread pool
            future = self.executor.submit(self._execute_job, job, processor, context)

            logger.info(
                "Job submitted to thread pool",
                job_id=job.job_id,
                estimated_duration=context.estimated_duration,
                estimated_cost=estimated_cost,
            )

            return future

    def submit_batch(self, jobs: List[Job], processor: Callable[[T], R]) -> List[Future[JobResult]]:
        """
        Submit multiple jobs for parallel execution.

        Args:
            jobs: List of jobs to execute
            processor: Function to process each job

        Returns:
            List of futures for job results
        """
        futures = []

        for job in jobs:
            future = self.submit_job(job, processor)
            futures.append(future)

        logger.info(
            "Batch of jobs submitted",
            batch_size=len(jobs),
            total_submitted=self.total_jobs_submitted,
        )

        return futures

    def wait_for_completion(
        self, futures: List[Future[JobResult]], timeout: Optional[float] = None
    ) -> List[JobResult]:
        """
        Wait for all futures to complete and return results.

        Args:
            futures: List of futures to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            List of job results
        """
        results = []

        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)

                    if result.succeeded:
                        logger.debug(
                            "Job completed successfully",
                            job_id=result.job_id,
                            duration=result.duration_seconds,
                        )
                    else:
                        logger.warning(
                            "Job failed", job_id=result.job_id, error=result.error_message
                        )

                except Exception as e:
                    logger.error("Error retrieving job result", error=str(e))
                    # Create failed result
                    failed_result = JobResult(
                        job_id="unknown", status=JobStatus.FAILED, error_message=str(e)
                    )
                    results.append(failed_result)

        except TimeoutError:
            logger.warning("Timeout waiting for job completion", timeout=timeout)

            # Cancel remaining futures
            for future in futures:
                if not future.done():
                    future.cancel()

        return results

    @DryRunGuard.protect
    def _execute_job(
        self, job: Job, processor: Callable[[T], R], context: ExecutionContext
    ) -> JobResult:
        """Execute a single job with monitoring and safety guards."""

        # Update context with actual thread ID
        context.thread_id = threading.get_ident()

        with self.lock:
            self.active_jobs[job.job_id] = context
            self.peak_concurrent_jobs = max(self.peak_concurrent_jobs, len(self.active_jobs))

        start_time = time.time()

        try:
            with time_operation(f"job_execution_{job.job_id}"):
                # Check for failure injection
                seeded_failure(f"job_execution_{job.job_id}")

                logger.info(
                    "Starting job execution",
                    job_id=job.job_id,
                    thread_id=context.thread_id,
                    estimated_duration=context.estimated_duration,
                )

                # Execute the job
                if job.function:
                    # Function-based execution
                    result_data = job.function(*job.args, **job.kwargs)
                elif job.command:
                    # Command-based execution (simulated in dry-run)
                    result_data = self._execute_command(job.command)
                else:
                    # Use provided processor
                    result_data = processor(job)

                end_time = time.time()
                duration = end_time - start_time

                # Create successful result
                result = JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    start_time=datetime.fromtimestamp(start_time),
                    end_time=datetime.fromtimestamp(end_time),
                    executor_type="threadpool",
                    result_data=result_data,
                    cpu_time_seconds=duration,  # Approximation
                    metadata={
                        "thread_id": context.thread_id,
                        "estimated_cost_usd": context.metadata.get("estimated_cost_usd", 0.0),
                        "actual_cost_usd": 0.0,  # Always $0.00 in dry-run
                    },
                )

                logger.info(
                    "Job execution completed", job_id=job.job_id, duration=duration, success=True
                )

                return result

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            logger.error("Job execution failed", job_id=job.job_id, error=str(e), duration=duration)

            # Create failed result
            result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.fromtimestamp(end_time),
                executor_type="threadpool",
                error_message=str(e),
                exception=e,
                cpu_time_seconds=duration,
                metadata={
                    "thread_id": context.thread_id,
                    "estimated_cost_usd": context.metadata.get("estimated_cost_usd", 0.0),
                    "actual_cost_usd": 0.0,
                },
            )

            return result

        finally:
            # Clean up execution context
            with self.lock:
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]

                self.completed_jobs.append(result)
                self.total_execution_time += duration

                if result.succeeded:
                    self.total_jobs_completed += 1
                else:
                    self.total_jobs_failed += 1

    def _execute_command(self, command: str) -> str:
        """Execute a command (simulated in dry-run mode)."""
        logger.info("Simulating command execution", command=command)

        # Simulate some processing time
        time.sleep(0.1)

        return f"Simulated output for command: {command}"

    def _estimate_job_cost(self, job: Job) -> float:
        """Estimate the cost of executing a job."""
        # In dry-run mode, all costs are $0.00
        if self.config.safety.dry_run:
            return 0.0

        # Estimate based on duration and resource requirements
        duration_hours = job.estimated_duration() / 3600.0
        cpu_rate = getattr(self.config.execution, "cpu_cost_per_sec_usd", 0.0)
        gpu_rate = getattr(self.config.execution, "gpu_cost_per_sec_usd", 0.0)
        cpu_cost = job.requirements.cpu_cores * duration_hours * cpu_rate * 3600
        gpu_cost = job.requirements.gpu_count * duration_hours * gpu_rate * 3600

        return cpu_cost + gpu_cost

    def _allocate_resources(self, job: Job) -> Dict[str, Any]:
        """Allocate resources for job execution."""
        return {
            "cpu_cores": job.requirements.cpu_cores,
            "memory_mb": job.requirements.memory_mb,
            "gpu_count": job.requirements.gpu_count,
            "thread_pool": self.thread_name_prefix,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self.lock:
            active_job_count = len(self.active_jobs)
            avg_execution_time = self.total_execution_time / max(self.total_jobs_completed, 1)

            success_rate = self.total_jobs_completed / max(self.total_jobs_submitted, 1)

            return {
                "executor_type": "threadpool",
                "max_workers": self.max_workers,
                "active_jobs": active_job_count,
                "total_submitted": self.total_jobs_submitted,
                "total_completed": self.total_jobs_completed,
                "total_failed": self.total_jobs_failed,
                "success_rate": success_rate,
                "peak_concurrent_jobs": self.peak_concurrent_jobs,
                "avg_execution_time_seconds": avg_execution_time,
                "total_execution_time_seconds": self.total_execution_time,
                "monitoring_enabled": self.enable_monitoring,
            }

    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get information about currently active jobs."""
        with self.lock:
            active_jobs = []
            current_time = datetime.utcnow()

            for job_id, context in self.active_jobs.items():
                runtime = (current_time - context.start_time).total_seconds()

                active_jobs.append(
                    {
                        "job_id": job_id,
                        "thread_id": context.thread_id,
                        "start_time": context.start_time.isoformat(),
                        "runtime_seconds": runtime,
                        "estimated_duration": context.estimated_duration,
                        "progress": (
                            min(runtime / context.estimated_duration, 1.0)
                            if context.estimated_duration > 0
                            else 0.0
                        ),
                        "resource_allocation": context.resource_allocation,
                    }
                )

            return active_jobs

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the executor gracefully."""
        logger.info(
            "Shutting down ThreadPool executor",
            active_jobs=len(self.active_jobs),
            wait=wait,
            timeout=timeout,
        )

        try:
            self.executor.shutdown(wait=wait, timeout=timeout)

            with self.lock:
                if self.active_jobs:
                    logger.warning(
                        "Executor shutdown with active jobs", active_job_count=len(self.active_jobs)
                    )

        except Exception as e:
            logger.error("Error during executor shutdown", error=str(e))

        logger.info("ThreadPool executor shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True, timeout=30.0)
