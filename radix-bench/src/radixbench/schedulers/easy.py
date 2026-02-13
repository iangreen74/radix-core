"""EASY Backfilling scheduler implementation."""

from typing import List, Optional, Dict
from .base import Scheduler, Job, GPU, Assignment, JobStatus
import logging

logger = logging.getLogger(__name__)


class EASYScheduler(Scheduler):
    """
    EASY Backfilling scheduler - minimal working implementation.

    Maintains reservation for head-of-queue job and packs smaller jobs
    that do not delay the reservation time.
    """

    def __init__(self):
        super().__init__()
        self.queue = []
        self.reservations: Dict[str, float] = {}  # gpu_id -> reserved_until_time
        self.runtime_history: Dict[str, List[float]] = {}  # job_type -> runtimes
        logger.info("EASY backfilling scheduler with runtime prediction")

    def submit(self, job: Job):
        """Submit job to queue."""
        self.submitted_jobs.append(job)
        self.queue.append(job)
        logger.debug(f"Job {job.job_id} added to EASY queue")

    def _predict_runtime(self, job: Job) -> float:
        """Predict job runtime based on history or use estimate."""
        job_type = job.job_id.split('_')[0] if '_' in job.job_id else 'default'

        if job_type in self.runtime_history and self.runtime_history[job_type]:
            # Use average of historical runtimes
            return sum(self.runtime_history[job_type]) / len(self.runtime_history[job_type])

        # Fallback to user estimate
        return job.runtime_estimate

    def _find_earliest_slot(self, job: Job, cluster: List[GPU], current_time: float) -> Optional[tuple]:
        """Find earliest time slot for job considering reservations."""
        predicted_runtime = self._predict_runtime(job)

        for gpu in cluster:
            if gpu.memory_gb < job.memory_gb:
                continue

            # Check if GPU is available now
            if gpu.is_available:
                return (gpu, current_time)

            # Check reservation time
            reserved_until = self.reservations.get(gpu.gpu_id, current_time)
            if reserved_until <= current_time:
                return (gpu, current_time)

            # Could start after reservation expires
            return (gpu, reserved_until)

        return None

    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs using EASY backfilling algorithm."""
        assignments = []
        available_gpus = [gpu for gpu in cluster if gpu.is_available]
        pending_jobs = [job for job in self.submitted_jobs if job.status == JobStatus.PENDING]

        if not pending_jobs:
            return assignments

        # Sort queue by submit time (FIFO order)
        pending_jobs.sort(key=lambda j: j.submit_time)

        # Try to schedule head-of-queue job first (make reservation if needed)
        if pending_jobs and available_gpus:
            head_job = pending_jobs[0]
            slot = self._find_earliest_slot(head_job, cluster, current_time)

            if slot:
                gpu, start_time = slot
                if start_time == current_time and gpu.is_available:
                    # Can start immediately
                    assignment = Assignment(
                        job=head_job,
                        gpu=gpu,
                        start_time=current_time
                    )
                    assignments.append(assignment)
                    available_gpus.remove(gpu)

                    # Update job status
                    head_job.status = JobStatus.RUNNING
                    head_job.start_time = current_time
                    head_job.assigned_gpu = gpu.gpu_id

                    logger.debug(f"EASY scheduled head job {head_job.job_id} on GPU {gpu.gpu_id}")
                else:
                    # Make reservation for future start
                    predicted_runtime = self._predict_runtime(head_job)
                    self.reservations[gpu.gpu_id] = start_time + predicted_runtime
                    logger.debug(f"EASY reserved GPU {gpu.gpu_id} for job {head_job.job_id} at {start_time}")

        # Backfill: try to schedule smaller jobs that won't delay reservations
        for job in pending_jobs[1:]:  # Skip head job
            if not available_gpus:
                break

            predicted_runtime = self._predict_runtime(job)

            # Find GPU that can fit this job without delaying reservations
            suitable_gpu = None
            for gpu in available_gpus:
                if gpu.memory_gb >= job.memory_gb:
                    # Check if job would finish before any reservation
                    job_end_time = current_time + predicted_runtime
                    reserved_time = self.reservations.get(gpu.gpu_id, float('inf'))

                    if job_end_time <= reserved_time:
                        suitable_gpu = gpu
                        break

            if suitable_gpu:
                assignment = Assignment(
                    job=job,
                    gpu=suitable_gpu,
                    start_time=current_time
                )
                assignments.append(assignment)
                available_gpus.remove(suitable_gpu)

                # Update job status
                job.status = JobStatus.RUNNING
                job.start_time = current_time
                job.assigned_gpu = suitable_gpu.gpu_id

                logger.debug(f"EASY backfilled job {job.job_id} on GPU {suitable_gpu.gpu_id}")

        return assignments

    def job_completed(self, job: Job):
        """Update runtime history when job completes."""
        if job.actual_runtime is not None:
            job_type = job.job_id.split('_')[0] if '_' in job.job_id else 'default'
            if job_type not in self.runtime_history:
                self.runtime_history[job_type] = []

            # Keep last 10 runtimes for prediction
            self.runtime_history[job_type].append(job.actual_runtime)
            if len(self.runtime_history[job_type]) > 10:
                self.runtime_history[job_type].pop(0)

        # Clear reservation if this GPU was reserved
        if job.assigned_gpu and job.assigned_gpu in self.reservations:
            del self.reservations[job.assigned_gpu]
