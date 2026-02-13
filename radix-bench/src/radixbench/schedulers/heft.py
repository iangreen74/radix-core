"""HEFT (Heterogeneous Earliest Finish Time) scheduler implementation."""

from typing import List, Dict, Optional, Set
from .base import Scheduler, Job, GPU, Assignment, JobStatus
import logging

logger = logging.getLogger(__name__)


class HEFTScheduler(Scheduler):
    """
    HEFT scheduler - minimal working implementation.

    For single-GPU jobs without DAG dependencies, falls back to SRPT-like
    scheduling with earliest finish time heuristic.
    """

    def __init__(self):
        super().__init__()
        self.queue = []
        self.dag_edges: Dict[str, List[str]] = {}  # job_id -> list of successor job_ids
        self.computation_costs: Dict[str, float] = {}  # job_id -> mean computation cost
        logger.info("HEFT scheduler with DAG support and SRPT fallback")

    def submit(self, job: Job):
        """Submit job to HEFT scheduler."""
        self.submitted_jobs.append(job)
        self.queue.append(job)

        # Store computation cost estimate
        self.computation_costs[job.job_id] = job.runtime_estimate

        logger.debug(f"HEFT job {job.job_id} submitted with cost {job.runtime_estimate}")

    def add_dependency(self, parent_job_id: str, child_job_id: str):
        """Add DAG dependency between jobs."""
        if parent_job_id not in self.dag_edges:
            self.dag_edges[parent_job_id] = []
        self.dag_edges[parent_job_id].append(child_job_id)
        logger.debug(f"HEFT dependency: {parent_job_id} -> {child_job_id}")

    def _calculate_upward_rank(self, job_id: str, memo: Dict[str, float] = None) -> float:
        """Calculate upward rank (critical path length from job to exit)."""
        if memo is None:
            memo = {}

        if job_id in memo:
            return memo[job_id]

        # Base computation cost
        comp_cost = self.computation_costs.get(job_id, 0.0)

        # Find maximum path through successors
        max_successor_rank = 0.0
        successors = self.dag_edges.get(job_id, [])

        for successor_id in successors:
            successor_rank = self._calculate_upward_rank(successor_id, memo)
            max_successor_rank = max(max_successor_rank, successor_rank)

        rank = comp_cost + max_successor_rank
        memo[job_id] = rank
        return rank

    def _get_ready_jobs(self, pending_jobs: List[Job]) -> List[Job]:
        """Get jobs that have no pending dependencies."""
        ready_jobs = []

        for job in pending_jobs:
            # Check if all dependencies are completed
            is_ready = True
            for parent_id, children in self.dag_edges.items():
                if job.job_id in children:
                    # This job has a parent - check if parent is completed
                    parent_completed = any(
                        j.job_id == parent_id and j.status == JobStatus.COMPLETED
                        for j in self.submitted_jobs
                    )
                    if not parent_completed:
                        is_ready = False
                        break

            if is_ready:
                ready_jobs.append(job)

        return ready_jobs

    def _calculate_earliest_finish_time(self, job: Job, gpu: GPU, current_time: float) -> float:
        """Calculate earliest finish time for job on given GPU."""
        # Simple model: start time + runtime estimate
        start_time = current_time

        # If GPU is busy, wait until it's free
        if not gpu.is_available:
            # In real implementation, would track GPU availability times
            start_time = current_time  # Simplified

        return start_time + job.runtime_estimate

    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs using HEFT algorithm."""
        assignments = []
        available_gpus = [gpu for gpu in cluster if gpu.is_available]
        pending_jobs = [job for job in self.submitted_jobs if job.status == JobStatus.PENDING]

        if not available_gpus or not pending_jobs:
            return assignments

        # Get jobs ready for execution (no pending dependencies)
        ready_jobs = self._get_ready_jobs(pending_jobs)

        if not ready_jobs:
            return assignments

        # Calculate upward ranks for prioritization
        job_ranks = []
        for job in ready_jobs:
            rank = self._calculate_upward_rank(job.job_id)
            job_ranks.append((rank, job.job_id, job))

        # Sort by upward rank (highest first), then job_id for determinism
        job_ranks.sort(key=lambda x: (-x[0], x[1]))

        # Schedule jobs greedily by rank
        for _, _, job in job_ranks:
            if not available_gpus:
                break

            # Find GPU that gives earliest finish time
            best_gpu = None
            best_finish_time = float('inf')

            for gpu in available_gpus:
                if gpu.memory_gb >= job.memory_gb:
                    finish_time = self._calculate_earliest_finish_time(job, gpu, current_time)
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_gpu = gpu

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

                logger.debug(f"HEFT assigned job {job.job_id} to GPU {best_gpu.gpu_id} (rank: {job_ranks[0][0]:.2f})")

        return assignments
