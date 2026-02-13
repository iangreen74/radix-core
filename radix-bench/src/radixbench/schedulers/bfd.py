"""Best Fit Decreasing scheduler implementation."""

from typing import List, Dict, Tuple
from .base import Scheduler, Job, GPU, Assignment, JobStatus
import logging

logger = logging.getLogger(__name__)


class BFDScheduler(Scheduler):
    """
    Best Fit Decreasing scheduler - minimal working implementation.

    Sorts jobs by memory requirement (decreasing) and assigns each job
    to the GPU with the least remaining memory that can still fit the job.
    Classic bin packing heuristic adapted for GPU scheduling.
    """

    def __init__(self):
        super().__init__()
        self.gpu_utilization: Dict[str, float] = {}  # gpu_id -> used_memory_gb
        logger.info("BFD scheduler: bin packing with memory optimization")

    def submit(self, job: Job):
        """Submit job to BFD scheduler."""
        self.submitted_jobs.append(job)
        logger.debug(f"BFD job {job.job_id} submitted (memory: {job.memory_gb} GB)")

    def _update_gpu_utilization(self, cluster: List[GPU]):
        """Update GPU memory utilization tracking."""
        for gpu in cluster:
            if gpu.gpu_id not in self.gpu_utilization:
                self.gpu_utilization[gpu.gpu_id] = 0.0

            # In real implementation, would track actual memory usage
            # For now, assume single job per GPU
            if not gpu.is_available:
                self.gpu_utilization[gpu.gpu_id] = gpu.memory_gb
            else:
                self.gpu_utilization[gpu.gpu_id] = 0.0

    def _find_best_fit_gpu(self, job: Job, available_gpus: List[GPU]) -> GPU:
        """Find GPU with least remaining memory that can fit the job."""
        best_gpu = None
        min_remaining_memory = float('inf')

        for gpu in available_gpus:
            if gpu.memory_gb >= job.memory_gb:
                used_memory = self.gpu_utilization.get(gpu.gpu_id, 0.0)
                remaining_memory = gpu.memory_gb - used_memory

                # Best fit: smallest remaining memory that still fits
                if remaining_memory >= job.memory_gb and remaining_memory < min_remaining_memory:
                    min_remaining_memory = remaining_memory
                    best_gpu = gpu

        return best_gpu

    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs using Best Fit Decreasing algorithm."""
        assignments = []
        available_gpus = [gpu for gpu in cluster if gpu.is_available]
        pending_jobs = [job for job in self.submitted_jobs if job.status == JobStatus.PENDING]

        if not available_gpus or not pending_jobs:
            return assignments

        # Update GPU utilization tracking
        self._update_gpu_utilization(cluster)

        # Sort jobs by memory requirement (decreasing order)
        # Secondary sort by job_id for deterministic tie-breaking
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (-j.memory_gb, j.job_id)
        )

        # Assign jobs using best fit heuristic
        for job in sorted_jobs:
            if not available_gpus:
                break

            best_gpu = self._find_best_fit_gpu(job, available_gpus)

            if best_gpu:
                assignment = Assignment(
                    job=job,
                    gpu=best_gpu,
                    start_time=current_time
                )
                assignments.append(assignment)
                available_gpus.remove(best_gpu)

                # Update job status
                job.status = JobStatus.RUNNING
                job.start_time = current_time
                job.assigned_gpu = best_gpu.gpu_id

                # Update utilization tracking
                self.gpu_utilization[best_gpu.gpu_id] += job.memory_gb

                logger.debug(f"BFD assigned job {job.job_id} ({job.memory_gb} GB) to GPU {best_gpu.gpu_id}")

        return assignments

    def job_completed(self, job: Job):
        """Update utilization when job completes."""
        if job.assigned_gpu and job.assigned_gpu in self.gpu_utilization:
            self.gpu_utilization[job.assigned_gpu] = max(
                0.0,
                self.gpu_utilization[job.assigned_gpu] - job.memory_gb
            )
