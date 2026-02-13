"""Dominant Resource Fairness scheduler implementation."""

from typing import List, Dict, Tuple
from .base import Scheduler, Job, GPU, Assignment, JobStatus
import logging

logger = logging.getLogger(__name__)


class DRFScheduler(Scheduler):
    """
    Dominant Resource Fairness scheduler - minimal working implementation.

    Models each job's demand as (gpu=1, vram=memory_gb).
    Computes dominant share over cluster capacity and allocates greedily
    by lowest dominant share first.
    """

    def __init__(self):
        super().__init__()
        self.user_allocations: Dict[str, Dict[str, float]] = {}  # user -> {gpu: count, vram: GB}
        self.cluster_capacity = {"gpu": 0, "vram": 0}  # Total cluster resources
        logger.info("DRF scheduler: multi-resource fairness for single-GPU jobs")

    def submit(self, job: Job):
        """Submit job and initialize user allocation if needed."""
        self.submitted_jobs.append(job)
        user = job.job_id.split('_')[0] if '_' in job.job_id else 'default'
        if user not in self.user_allocations:
            self.user_allocations[user] = {"gpu": 0.0, "vram": 0.0}

    def _update_cluster_capacity(self, cluster: List[GPU]):
        """Update cluster capacity based on available GPUs."""
        self.cluster_capacity["gpu"] = len(cluster)
        self.cluster_capacity["vram"] = sum(gpu.memory_gb for gpu in cluster)

    def _calculate_dominant_share(self, user: str, job: Job) -> float:
        """Calculate user's dominant share if this job were allocated."""
        if self.cluster_capacity["gpu"] == 0 or self.cluster_capacity["vram"] == 0:
            return 0.0

        current_alloc = self.user_allocations.get(user, {"gpu": 0.0, "vram": 0.0})

        # Job demand: 1 GPU + memory_gb VRAM
        new_gpu_share = (current_alloc["gpu"] + 1) / self.cluster_capacity["gpu"]
        new_vram_share = (current_alloc["vram"] + job.memory_gb) / self.cluster_capacity["vram"]

        # Dominant share is the maximum resource share
        return max(new_gpu_share, new_vram_share)

    def _get_user_from_job(self, job: Job) -> str:
        """Extract user from job_id."""
        return job.job_id.split('_')[0] if '_' in job.job_id else 'default'

    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs using DRF algorithm."""
        assignments = []
        available_gpus = [gpu for gpu in cluster if gpu.is_available]
        pending_jobs = [job for job in self.submitted_jobs if job.status == JobStatus.PENDING]

        if not available_gpus or not pending_jobs:
            return assignments

        # Update cluster capacity
        self._update_cluster_capacity(cluster)

        # Calculate dominant share for each job and sort by fairness
        job_priorities = []
        for job in pending_jobs:
            user = self._get_user_from_job(job)
            dominant_share = self._calculate_dominant_share(user, job)
            job_priorities.append((dominant_share, job.job_id, job))

        # Sort by dominant share (lowest first), then job_id for deterministic tie-breaking
        job_priorities.sort(key=lambda x: (x[0], x[1]))

        # Allocate jobs greedily
        for _, _, job in job_priorities:
            if not available_gpus:
                break

            # Find suitable GPU
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

                # Update user allocation tracking
                user = self._get_user_from_job(job)
                if user not in self.user_allocations:
                    self.user_allocations[user] = {"gpu": 0.0, "vram": 0.0}
                self.user_allocations[user]["gpu"] += 1
                self.user_allocations[user]["vram"] += job.memory_gb

                logger.debug(f"DRF assigned job {job.job_id} to GPU {suitable_gpu.gpu_id} (user: {user})")

        return assignments

    def job_completed(self, job: Job, user: str = None):
        """Update allocations when job completes."""
        if user is None:
            user = self._get_user_from_job(job)

        if user in self.user_allocations:
            self.user_allocations[user]["gpu"] = max(0, self.user_allocations[user]["gpu"] - 1)
            self.user_allocations[user]["vram"] = max(0, self.user_allocations[user]["vram"] - job.memory_gb)
