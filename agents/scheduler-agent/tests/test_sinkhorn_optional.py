"""Test optional Sinkhorn assignment functionality."""

import pytest
import numpy as np
from app.sinkhorn import SinkhornSolver, test_sinkhorn_basic, test_job_assignment


class TestSinkhornOptional:
    """Test the optional Sinkhorn assignment feature."""

    def setup_method(self):
        """Set up test environment."""
        self.solver = SinkhornSolver(epsilon=0.1, max_iterations=100)

    def test_sinkhorn_basic_functionality(self):
        """Test basic Sinkhorn algorithm functionality."""
        # Run the built-in tests
        test_sinkhorn_basic()
        test_job_assignment()

    def test_sinkhorn_marginal_constraints(self):
        """Test that Sinkhorn respects marginal constraints."""
        # 3x3 problem
        costs = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [3.0, 2.0, 1.0]
        ])
        supply = np.array([1.0, 1.0, 1.0])
        demand = np.array([1.0, 1.0, 1.0])

        assignment = self.solver.solve(costs, supply, demand)

        # Check marginal constraints
        row_sums = assignment.sum(axis=1)
        col_sums = assignment.sum(axis=0)

        np.testing.assert_allclose(row_sums, supply, atol=1e-3)
        np.testing.assert_allclose(col_sums, demand, atol=1e-3)

        # Assignment should be non-negative
        assert np.all(assignment >= 0), "Assignment matrix should be non-negative"

    def test_sinkhorn_unbalanced_problem(self):
        """Test Sinkhorn with unbalanced supply and demand."""
        costs = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 1.5]
        ])
        supply = np.array([1.0, 1.0, 1.0])  # 3 jobs
        demand = np.array([2.0, 1.0])       # 3 GPU capacity total

        assignment = self.solver.solve(costs, supply, demand)

        # Check dimensions
        assert assignment.shape == (3, 2), "Assignment should have correct dimensions"

        # Check that total assignment doesn't exceed total capacity
        total_assigned = assignment.sum()
        total_capacity = demand.sum()
        assert total_assigned <= total_capacity + 1e-3, \
            f"Total assignment {total_assigned:.3f} should not exceed capacity {total_capacity:.3f}"

    def test_sinkhorn_convergence(self):
        """Test that Sinkhorn algorithm converges."""
        costs = np.random.rand(5, 4) * 10  # Random cost matrix
        supply = np.ones(5)
        demand = np.ones(4) * 1.25  # Slightly unbalanced

        assignment = self.solver.solve(costs, supply, demand)

        # Should converge to a valid assignment
        assert not np.any(np.isnan(assignment)), "Assignment should not contain NaN"
        assert not np.any(np.isinf(assignment)), "Assignment should not contain Inf"
        assert np.all(assignment >= 0), "Assignment should be non-negative"

    def test_job_gpu_assignment_interface(self):
        """Test the job-to-GPU assignment interface."""
        # 4 jobs, 3 GPUs
        job_costs = [
            [1.0, 2.0, 3.0],  # Job 0 prefers GPU 0
            [3.0, 1.0, 2.0],  # Job 1 prefers GPU 1
            [2.0, 3.0, 1.0],  # Job 2 prefers GPU 2
            [2.0, 2.0, 2.0]   # Job 3 is indifferent
        ]
        gpu_capacities = [2, 1, 2]  # Total capacity: 5 slots for 4 jobs

        assignments = self.solver.assign_jobs_to_gpus(job_costs, gpu_capacities)

        # Check results
        assert len(assignments) == 4, "Should have assignment for each job"

        for i, assignment in enumerate(assignments):
            # Each job should have some assignment
            assert len(assignment) > 0, f"Job {i} should have some GPU assignment"

            # Probabilities should be reasonable
            total_prob = sum(assignment.values())
            assert 0.8 <= total_prob <= 1.2, \
                f"Job {i} total assignment probability should be ~1.0, got {total_prob:.3f}"

            # All probabilities should be positive
            for gpu_idx, prob in assignment.items():
                assert prob > 0, f"Job {i} assignment to GPU {gpu_idx} should be positive"
                assert gpu_idx < len(gpu_capacities), f"Invalid GPU index {gpu_idx}"

    def test_sinkhorn_preference_respecting(self):
        """Test that Sinkhorn respects cost preferences."""
        # Simple 2x2 problem with clear preferences
        job_costs = [
            [1.0, 10.0],  # Job 0 strongly prefers GPU 0
            [10.0, 1.0]   # Job 1 strongly prefers GPU 1
        ]
        gpu_capacities = [1, 1]

        assignments = self.solver.assign_jobs_to_gpus(job_costs, gpu_capacities)

        # Job 0 should be assigned primarily to GPU 0
        job_0_assignment = assignments[0]
        assert 0 in job_0_assignment, "Job 0 should be assigned to GPU 0"
        if 1 in job_0_assignment:
            assert job_0_assignment[0] > job_0_assignment[1], \
                "Job 0 should prefer GPU 0 over GPU 1"

        # Job 1 should be assigned primarily to GPU 1
        job_1_assignment = assignments[1]
        assert 1 in job_1_assignment, "Job 1 should be assigned to GPU 1"
        if 0 in job_1_assignment:
            assert job_1_assignment[1] > job_1_assignment[0], \
                "Job 1 should prefer GPU 1 over GPU 0"

    def test_sinkhorn_empty_inputs(self):
        """Test Sinkhorn behavior with empty inputs."""
        # Empty job list
        empty_assignments = self.solver.assign_jobs_to_gpus([], [1, 2, 3])
        assert empty_assignments == [], "Empty job list should return empty assignments"

        # Empty GPU list
        empty_assignments = self.solver.assign_jobs_to_gpus([[1.0, 2.0]], [])
        assert empty_assignments == [], "Empty GPU list should return empty assignments"

    def test_sinkhorn_epsilon_effect(self):
        """Test the effect of entropy regularization parameter."""
        costs = np.array([[1.0, 5.0], [5.0, 1.0]])
        supply = np.array([1.0, 1.0])
        demand = np.array([1.0, 1.0])

        # High entropy (more exploration)
        solver_high_entropy = SinkhornSolver(epsilon=1.0)
        assignment_high = solver_high_entropy.solve(costs, supply, demand)

        # Low entropy (more exploitation)
        solver_low_entropy = SinkhornSolver(epsilon=0.01)
        assignment_low = solver_low_entropy.solve(costs, supply, demand)

        # High entropy should be more uniform
        entropy_high = -np.sum(assignment_high * np.log(assignment_high + 1e-10))
        entropy_low = -np.sum(assignment_low * np.log(assignment_low + 1e-10))

        assert entropy_high > entropy_low, \
            f"High epsilon should lead to higher entropy: {entropy_high:.3f} vs {entropy_low:.3f}"

    def test_sinkhorn_deterministic(self):
        """Test that Sinkhorn gives deterministic results."""
        costs = np.random.rand(3, 3)
        supply = np.ones(3)
        demand = np.ones(3)

        # Run twice with same inputs
        assignment_1 = self.solver.solve(costs, supply, demand)
        assignment_2 = self.solver.solve(costs, supply, demand)

        # Should be identical
        np.testing.assert_allclose(assignment_1, assignment_2, atol=1e-10)

    def test_sinkhorn_feature_flag_integration(self):
        """Test integration with feature flag system."""
        from app.config import SchedulerConfig

        # Test with Sinkhorn enabled
        config_enabled = SchedulerConfig(enable_sinkhorn=True)
        assert config_enabled.enable_sinkhorn, "Sinkhorn should be enabled"

        # Test with Sinkhorn disabled (default)
        config_disabled = SchedulerConfig(enable_sinkhorn=False)
        assert not config_disabled.enable_sinkhorn, "Sinkhorn should be disabled by default"

        # The actual integration would be in the main scoring logic
        # This test just ensures the configuration is available
