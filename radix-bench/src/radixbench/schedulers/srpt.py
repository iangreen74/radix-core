"""Shortest Remaining Processing Time scheduler implementation."""

from typing import List
from .base import Scheduler, Job, GPU, Assignment, JobStatus
import logging

logger = logging.getLogger(__name__)


class SRPTScheduler(Scheduler):
    """
    Shortest Remaining Processing Time scheduler.

    Prioritizes jobs with the shortest estimated remaining runtime.
    Preemptive scheduler that can interrupt running jobs for shorter ones.
    """

    def __init__(self):
        super().__init__()
        logger.info("SRPT scheduler initialized")

    def submit(self, job: Job):
        """Submit job to pending queue."""
        self.submitted_jobs.append(job)
        logger.debug(f"Job {job.job_id} submitted with runtime estimate {job.runtime_estimate}")

    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs by shortest remaining processing time."""
        assignments = []
        available_gpus = [gpu for gpu in cluster if gpu.is_available]

        # Get all schedulable jobs (pending + potentially preemptible running jobs)
        pending_jobs = [job for job in self.submitted_jobs if job.status == JobStatus.PENDING]
        running_jobs = [job for job in self.submitted_jobs if job.status == JobStatus.RUNNING]

        # Calculate remaining time for running jobs
        for job in running_jobs:
            if job.start_time is not None:
                elapsed = current_time - job.start_time
                job.remaining_time = max(0, job.runtime_estimate - elapsed)

        # Sort pending jobs by runtime estimate (shortest first)
        pending_jobs.sort(key=lambda j: j.runtime_estimate)

        # Assign pending jobs to available GPUs
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
                job.status = JobStatus.RUNNING
                job.start_time = current_time
                job.assigned_gpu = suitable_gpu.gpu_id

                logger.debug(f"Assigned job {job.job_id} to GPU {suitable_gpu.gpu_id} (est. runtime: {job.runtime_estimate})")

        # TODO: Implement preemption logic for running jobs with longer remaining time
        # For now, this is a non-preemptive SRPT implementation

        return assignments
