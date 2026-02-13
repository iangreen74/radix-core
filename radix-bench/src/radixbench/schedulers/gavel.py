"""Gavel scheduler implementation."""

from typing import List, Dict, Optional
from .base import Scheduler, Job, GPU, Assignment, JobStatus
import logging
import math

logger = logging.getLogger(__name__)


class GavelScheduler(Scheduler):
    """
    Gavel scheduler - minimal working implementation.

    Utility-aware scheduling that considers job speedup and fairness.
    Approximates Gavel's utility maximization with simplified speedup model.
    """

    def __init__(self):
        super().__init__()
        self.user_allocations: Dict[str, float] = {}  # user -> total allocated time
        self.speedup_history: Dict[str, List[float]] = {}  # job_type -> observed speedups
        logger.info("Gavel scheduler: utility-aware with speedup optimization")

    def submit(self, job: Job):
        """Submit job to Gavel scheduler."""
        self.submitted_jobs.append(job)

        # Initialize user allocation tracking
        user = self._get_user_from_job(job)
        if user not in self.user_allocations:
            self.user_allocations[user] = 0.0

        logger.debug(f"Gavel job {job.job_id} submitted for user {user}")

    def _get_user_from_job(self, job: Job) -> str:
        """Extract user from job_id."""
        return job.job_id.split('_')[0] if '_' in job.job_id else 'default'

    def _estimate_speedup(self, job: Job, gpu: GPU) -> float:
        """Estimate speedup for job on given GPU."""
        job_type = job.job_id.split('_')[1] if '_' in job.job_id else 'default'

        # Use historical speedup data if available
        if job_type in self.speedup_history and self.speedup_history[job_type]:
            return sum(self.speedup_history[job_type]) / len(self.speedup_history[job_type])

        # Simple speedup model based on memory ratio
        # Assume jobs with higher memory requirements get better speedup on larger GPUs
        memory_ratio = min(job.memory_gb / gpu.memory_gb, 1.0)

        # Logarithmic speedup model: speedup = 1 + log(1 + memory_ratio)
        speedup = 1.0 + math.log(1.0 + memory_ratio)

        return speedup

    def _calculate_utility(self, job: Job, gpu: GPU, current_time: float) -> float:
        """Calculate utility for assigning job to GPU."""
        user = self._get_user_from_job(job)

        # Speedup component
        speedup = self._estimate_speedup(job, gpu)

        # Fairness component (inverse of user's current allocation)
        user_alloc = self.user_allocations.get(user, 0.0)
        fairness_weight = 1.0 / (1.0 + user_alloc)

        # Age component (jobs waiting longer get higher priority)
        wait_time = current_time - job.submit_time
        age_weight = 1.0 + (wait_time / 3600.0)  # Increase priority by 1 per hour

        # Combined utility: speedup * fairness * age
        utility = speedup * fairness_weight * age_weight

        return utility

    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs using Gavel utility maximization."""
        assignments = []
        available_gpus = [gpu for gpu in cluster if gpu.is_available]
        pending_jobs = [job for job in self.submitted_jobs if job.status == JobStatus.PENDING]

        if not available_gpus or not pending_jobs:
            return assignments

        # Calculate utility for all job-GPU pairs
        job_gpu_utilities = []
        for job in pending_jobs:
            for gpu in available_gpus:
                if gpu.memory_gb >= job.memory_gb:
                    utility = self._calculate_utility(job, gpu, current_time)
                    job_gpu_utilities.append((utility, job.job_id, job, gpu))

        # Sort by utility (highest first), then job_id for determinism
        job_gpu_utilities.sort(key=lambda x: (-x[0], x[1]))

        # Greedily assign highest utility pairs
        assigned_jobs = set()
        assigned_gpus = set()

        for utility, job_id, job, gpu in job_gpu_utilities:
            if job.job_id in assigned_jobs or gpu.gpu_id in assigned_gpus:
                continue

            assignment = Assignment(
                job=job,
                gpu=gpu,
                start_time=current_time
            )
            assignments.append(assignment)

            assigned_jobs.add(job.job_id)
            assigned_gpus.add(gpu.gpu_id)

            # Update job status
            job.status = JobStatus.RUNNING
            job.start_time = current_time
            job.assigned_gpu = gpu.gpu_id

            # Update user allocation
            user = self._get_user_from_job(job)
            self.user_allocations[user] += job.runtime_estimate

            logger.debug(f"Gavel assigned job {job.job_id} to GPU {gpu.gpu_id} (utility: {utility:.3f})")

        return assignments

    def job_completed(self, job: Job):
        """Update speedup history when job completes."""
        if job.actual_runtime is not None and job.runtime_estimate > 0:
            # Calculate observed speedup (estimate / actual)
            observed_speedup = job.runtime_estimate / job.actual_runtime

            job_type = job.job_id.split('_')[1] if '_' in job.job_id else 'default'
            if job_type not in self.speedup_history:
                self.speedup_history[job_type] = []

            # Keep last 10 observations for speedup estimation
            self.speedup_history[job_type].append(observed_speedup)
            if len(self.speedup_history[job_type]) > 10:
                self.speedup_history[job_type].pop(0)

            logger.debug(f"Gavel recorded speedup {observed_speedup:.2f} for job type {job_type}")
