"""Test that uncertainty decreases with observations."""

import pytest
import math
from app.config import SchedulerConfig
from app.model import SchedulerModel, JobFeatures


class TestUncertaintyShrinks:
    """Test that model uncertainty decreases as observations are added."""

    def setup_method(self):
        """Set up test environment."""
        self.config = SchedulerConfig()
        self.model = SchedulerModel(self.config, ":memory:")

    def test_uncertainty_decreases_with_observations(self):
        """Test basic uncertainty reduction with observations."""
        job_type = "shrink-test"
        gpu_type = "A100-80GB"

        # Get initial uncertainty (should be high)
        initial_mean, initial_variance = self.model.get_stats(job_type, gpu_type)
        initial_uncertainty = math.sqrt(initial_variance)

        assert initial_uncertainty > 2.0, "Initial uncertainty should be high for unknown job-GPU pair"

        # Add observations
        observations = [5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 5.1, 4.8]

        for runtime in observations:
            self.model.observe(job_type, gpu_type, runtime)

        # Check that uncertainty decreased
        final_mean, final_variance = self.model.get_stats(job_type, gpu_type)
        final_uncertainty = math.sqrt(final_variance)

        assert final_uncertainty < initial_uncertainty, \
            f"Uncertainty should decrease: {initial_uncertainty:.3f} -> {final_uncertainty:.3f}"

        # Mean should converge to observed values
        expected_mean = sum(observations) / len(observations)
        assert abs(final_mean - expected_mean) < 1.0, \
            f"Mean should converge to observations: expected ~{expected_mean:.2f}, got {final_mean:.2f}"

    def test_uncertainty_monotonic_decrease(self):
        """Test that uncertainty decreases monotonically with more observations."""
        job_type = "monotonic-test"
        gpu_type = "A100-80GB"

        uncertainties = []
        observations = [10.0] * 20  # Consistent observations

        for i, runtime in enumerate(observations):
            if i > 0:  # Skip first measurement (before any observations)
                _, variance = self.model.get_stats(job_type, gpu_type)
                uncertainties.append(math.sqrt(variance))

            self.model.observe(job_type, gpu_type, runtime)

        # Check that uncertainty generally decreases
        # (Allow for some noise due to EMA updates)
        for i in range(1, len(uncertainties)):
            # Uncertainty should be decreasing or staying roughly the same
            assert uncertainties[i] <= uncertainties[i-1] + 0.1, \
                f"Uncertainty increased significantly at step {i}: " \
                f"{uncertainties[i-1]:.3f} -> {uncertainties[i]:.3f}"

        # Overall trend should be downward
        assert uncertainties[-1] < uncertainties[0], \
            f"Overall uncertainty should decrease: {uncertainties[0]:.3f} -> {uncertainties[-1]:.3f}"

    def test_uncertainty_convergence_limit(self):
        """Test that uncertainty converges to a reasonable minimum."""
        job_type = "convergence-test"
        gpu_type = "A100-80GB"

        # Add many consistent observations
        consistent_runtime = 7.5
        for _ in range(100):
            self.model.observe(job_type, gpu_type, consistent_runtime)

        _, final_variance = self.model.get_stats(job_type, gpu_type)
        final_uncertainty = math.sqrt(final_variance)

        # Should converge to a small but non-zero uncertainty
        assert final_uncertainty > 0.01, "Uncertainty should not go to zero"
        assert final_uncertainty < 1.0, "Uncertainty should be small with many observations"

    def test_uncertainty_with_noisy_observations(self):
        """Test uncertainty behavior with noisy observations."""
        job_type = "noisy-test"
        gpu_type = "A100-80GB"

        # Get initial state
        _, initial_variance = self.model.get_stats(job_type, gpu_type)
        initial_uncertainty = math.sqrt(initial_variance)

        # Add very noisy observations
        import random
        random.seed(42)
        noisy_observations = [random.uniform(1.0, 20.0) for _ in range(50)]

        for runtime in noisy_observations:
            self.model.observe(job_type, gpu_type, runtime)

        _, final_variance = self.model.get_stats(job_type, gpu_type)
        final_uncertainty = math.sqrt(final_variance)

        # Uncertainty should still decrease from initial high value
        # but remain higher than with consistent observations
        assert final_uncertainty < initial_uncertainty, \
            "Even with noise, uncertainty should decrease from initial value"
        assert final_uncertainty > 1.0, \
            "With noisy observations, uncertainty should remain reasonably high"

    def test_different_jobs_independent_uncertainty(self):
        """Test that uncertainty for different jobs is tracked independently."""
        gpu_type = "A100-80GB"

        # Observe one job type extensively
        job_type_1 = "well-observed-job"
        for _ in range(20):
            self.model.observe(job_type_1, gpu_type, 5.0)

        # Keep another job type unobserved
        job_type_2 = "unobserved-job"

        # Check that uncertainties are different
        _, variance_1 = self.model.get_stats(job_type_1, gpu_type)
        _, variance_2 = self.model.get_stats(job_type_2, gpu_type)

        uncertainty_1 = math.sqrt(variance_1)
        uncertainty_2 = math.sqrt(variance_2)

        assert uncertainty_1 < uncertainty_2, \
            f"Well-observed job should have lower uncertainty: " \
            f"{uncertainty_1:.3f} vs {uncertainty_2:.3f}"

    def test_information_gain_calculation(self):
        """Test that information gain correlates with uncertainty."""
        job_type = "ig-test"
        gpu_type = "A100-80GB"

        # Calculate information gain at different uncertainty levels
        variances = [0.1, 1.0, 4.0, 16.0, 25.0]
        information_gains = []

        for variance in variances:
            ig = self.model.information_gain(variance)
            information_gains.append(ig)

        # Information gain should increase with variance
        for i in range(1, len(information_gains)):
            assert information_gains[i] > information_gains[i-1], \
                f"Information gain should increase with variance: " \
                f"IG({variances[i-1]}) = {information_gains[i-1]:.3f}, " \
                f"IG({variances[i]}) = {information_gains[i]:.3f}"

        # Should be bounded and reasonable
        for ig in information_gains:
            assert ig >= 0, "Information gain should be non-negative"
            assert ig < 10, "Information gain should not be extremely large"

    def test_scoring_reflects_uncertainty_changes(self):
        """Test that scoring decisions reflect uncertainty changes."""
        features = JobFeatures(job_type="scoring-uncertainty-test")
        candidates = ["A100-80GB", "L4-24GB"]

        # Initial scoring with high uncertainty
        initial_result = self.model.score_job(features, candidates)
        initial_uncertainty = initial_result.terms["sigma"]
        initial_ig = initial_result.terms["ig"]

        # Add observations to reduce uncertainty
        for runtime in [8.0, 7.8, 8.2, 7.9, 8.1]:
            self.model.observe(features.job_type, initial_result.chosen_gpu, runtime)

        # Score again
        final_result = self.model.score_job(features, candidates)
        final_uncertainty = final_result.terms["sigma"]
        final_ig = final_result.terms["ig"]

        # Uncertainty and information gain should decrease
        assert final_uncertainty < initial_uncertainty, \
            "Uncertainty in scoring should decrease with observations"
        assert final_ig < initial_ig, \
            "Information gain should decrease with reduced uncertainty"
