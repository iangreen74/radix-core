"""First-In-First-Out scheduler implementation."""

from typing import List
from .base import Scheduler, Job, GPU, Assignment
import logging

logger = logging.getLogger(__name__)


class FIFOScheduler(Scheduler):
    """
    First-In-First-Out scheduler.

    Jobs are scheduled in the order they arrive (submit_time).
    Simple baseline scheduler for comparison.
    """

    def __init__(self):
        super().__init__()
        logger.info("FIFO scheduler initialized")

    def submit(self, job: Job):
        """Submit job to pending queue."""
        self.submitted_jobs.append(job)
        logger.debug(f"Job {job.job_id} submitted at {job.submit_time}")

    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs in FIFO order on available GPUs."""
        assignments = []
        available_gpus = [gpu for gpu in cluster if gpu.is_available]

        # Sort pending jobs by submit time (FIFO)
        pending_jobs = [job for job in self.submitted_jobs if job.status.value == "pending"]
        pending_jobs.sort(key=lambda j: j.submit_time)

        # Assign jobs to available GPUs
        for job in pending_jobs:
            if not available_gpus:
                break

            # Find GPU with sufficient memory
            suitable_gpu = None
            for gpu in available_gpus:
                if gpu.memory_gb >= job.memory_gb:
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
                job.status = job.status.RUNNING
                job.start_time = current_time
                job.assigned_gpu = suitable_gpu.gpu_id

                logger.debug(f"Assigned job {job.job_id} to GPU {suitable_gpu.gpu_id}")

        return assignments
