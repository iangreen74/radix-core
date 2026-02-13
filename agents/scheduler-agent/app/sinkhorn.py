"""Optional Sinkhorn assignment for batched GPU allocation.

This module provides entropy-regularized optimal transport for soft assignment
of multiple jobs to GPUs when batching is enabled. Feature-flagged and
unit-tested but not required for the MVP.
"""

import math
from typing import List, Dict, Tuple
import numpy as np


class SinkhornSolver:
    """Entropy-regularized optimal transport solver for GPU assignment."""

    def __init__(self, epsilon: float = 0.1, max_iterations: int = 100):
        """Initialize Sinkhorn solver.

        Args:
            epsilon: Entropy regularization parameter
            max_iterations: Maximum Sinkhorn iterations
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def solve(self, costs: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Solve optimal transport problem using Sinkhorn algorithm.

        Args:
            costs: Cost matrix [n_jobs, n_gpus]
            supply: Job demands [n_jobs] (typically all 1.0)
            demand: GPU capacities [n_gpus]

        Returns:
            Assignment matrix [n_jobs, n_gpus] with soft assignments
        """
        n_jobs, n_gpus = costs.shape

        # Convert to transport kernel
        K = np.exp(-costs / self.epsilon)

        # Initialize dual variables
        u = np.ones(n_jobs) / n_jobs
        v = np.ones(n_gpus) / n_gpus

        # Sinkhorn iterations
        for _ in range(self.max_iterations):
            u_prev = u.copy()

            # Update v
            v = demand / (K.T @ u + 1e-10)

            # Update u
            u = supply / (K @ v + 1e-10)

            # Check convergence
            if np.allclose(u, u_prev, rtol=1e-6):
                break

        # Compute assignment matrix
        assignment = np.diag(u) @ K @ np.diag(v)

        return assignment

    def assign_jobs_to_gpus(self, job_costs: List[List[float]],
                           gpu_capacities: List[int]) -> List[Dict[str, float]]:
        """Assign jobs to GPUs using Sinkhorn algorithm.

        Args:
            job_costs: List of cost vectors for each job [job][gpu]
            gpu_capacities: Capacity of each GPU

        Returns:
            List of assignment probabilities for each job {gpu_idx: probability}
        """
        if not job_costs or not gpu_capacities:
            return []

        n_jobs = len(job_costs)
        n_gpus = len(gpu_capacities)

        # Build cost matrix
        costs = np.array(job_costs)

        # Job supplies (all jobs need 1 GPU)
        supply = np.ones(n_jobs)

        # GPU demands (normalized capacities)
        demand = np.array(gpu_capacities, dtype=float)
        demand = demand / demand.sum() * n_jobs  # Scale to match total supply

        # Solve assignment
        assignment = self.solve(costs, supply, demand)

        # Convert to list of dictionaries
        result = []
        for job_idx in range(n_jobs):
            job_assignment = {}
            for gpu_idx in range(n_gpus):
                prob = assignment[job_idx, gpu_idx]
                if prob > 1e-6:  # Only include non-zero assignments
                    job_assignment[gpu_idx] = prob
            result.append(job_assignment)

        return result


def test_sinkhorn_basic():
    """Basic test for Sinkhorn assignment."""
    solver = SinkhornSolver(epsilon=0.1)

    # Simple 2x2 problem
    costs = np.array([[1.0, 2.0], [2.0, 1.0]])
    supply = np.array([1.0, 1.0])
    demand = np.array([1.0, 1.0])

    assignment = solver.solve(costs, supply, demand)

    # Check marginal constraints (approximately)
    row_sums = assignment.sum(axis=1)
    col_sums = assignment.sum(axis=0)

    assert np.allclose(row_sums, supply, atol=1e-3), f"Row sums: {row_sums} != {supply}"
    assert np.allclose(col_sums, demand, atol=1e-3), f"Col sums: {col_sums} != {demand}"

    print("Basic Sinkhorn test passed")


def test_job_assignment():
    """Test job-to-GPU assignment interface."""
    solver = SinkhornSolver()

    # 3 jobs, 2 GPUs
    job_costs = [
        [1.0, 2.0],  # Job 0 prefers GPU 0
        [2.0, 1.0],  # Job 1 prefers GPU 1
        [1.5, 1.5]   # Job 2 indifferent
    ]
    gpu_capacities = [2, 2]  # Each GPU can handle 2 jobs

    assignments = solver.assign_jobs_to_gpus(job_costs, gpu_capacities)

    assert len(assignments) == 3, "Should have 3 job assignments"

    # Each job should have non-empty assignment
    for i, assignment in enumerate(assignments):
        assert len(assignment) > 0, f"Job {i} has empty assignment"
        assert abs(sum(assignment.values()) - 1.0) < 0.1, f"Job {i} probabilities don't sum to ~1"

    print("Job assignment test passed")


if __name__ == "__main__":
    test_sinkhorn_basic()
    test_job_assignment()
    print("All Sinkhorn tests passed")
