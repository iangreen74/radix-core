"""Test exploration cap enforcement."""

import pytest
import random
from unittest.mock import MagicMock

from app.config import SchedulerConfig
from app.model import SchedulerModel, JobFeatures


class TestExplorationCap:
    """Test that exploration decisions respect the configured cap."""

    def setup_method(self):
        """Set up test environment."""
        self.config = SchedulerConfig(exploration_cap=0.25)
        self.model = SchedulerModel(self.config, ":memory:")
        random.seed(42)  # Deterministic tests

    def test_exploration_cap_enforcement(self):
        """Test that exploration ratio stays below cap over many trials."""
        features = JobFeatures(job_type="test-job")
        candidates = ["A100-80GB", "L4-24GB"]

        # Run many scoring decisions
        exploration_decisions = 0
        total_decisions = 1000

        for i in range(total_decisions):
            # Vary features slightly to trigger different uncertainty levels
            varied_features = JobFeatures(
                job_type=f"test-job-{i % 10}",
                batch_size=random.randint(1, 64)
            )

            result = self.model.score_job(varied_features, candidates)

            # High uncertainty (sigma > 1.0) indicates exploration
            if result.terms["sigma"] > 1.0:
                exploration_decisions += 1

        exploration_ratio = exploration_decisions / total_decisions

        # Should be at or below the cap (with some tolerance for randomness)
        assert exploration_ratio <= self.config.exploration_cap + 0.05, \
            f"Exploration ratio {exploration_ratio:.3f} exceeds cap {self.config.exploration_cap}"

    def test_exploration_cap_with_observations(self):
        """Test exploration cap with learning from observations."""
        features = JobFeatures(job_type="learned-job")
        candidates = ["A100-80GB"]

        # Initial high uncertainty should trigger exploration
        result1 = self.model.score_job(features, candidates)
        initial_uncertainty = result1.terms["sigma"]

        # Add several observations to reduce uncertainty
        for runtime in [5.0, 5.2, 4.8, 5.1, 4.9]:
            self.model.observe("learned-job", "A100-80GB", runtime)

        # Uncertainty should decrease
        result2 = self.model.score_job(features, candidates)
        final_uncertainty = result2.terms["sigma"]

        assert final_uncertainty < initial_uncertainty, \
            "Uncertainty should decrease with observations"

        # With low uncertainty, should prefer exploitation
        assert final_uncertainty < 1.0, \
            "After observations, uncertainty should be low enough for exploitation"

    def test_exploration_cap_zero(self):
        """Test that zero exploration cap forces pure exploitation."""
        config = SchedulerConfig(exploration_cap=0.0)
        model = SchedulerModel(config, ":memory:")

        features = JobFeatures(job_type="exploit-only")
        candidates = ["A100-80GB", "L4-24GB"]

        # Even with high uncertainty, should not explore
        for _ in range(100):
            result = model.score_job(features, candidates)
            # With zero exploration cap, should always choose lowest mean
            # (which for uninformed prior is the first candidate)
            assert result.chosen_gpu == candidates[0]

    def test_exploration_cap_one(self):
        """Test that exploration cap of 1.0 allows full exploration."""
        config = SchedulerConfig(exploration_cap=1.0)
        model = SchedulerModel(config, ":memory:")

        features = JobFeatures(job_type="explore-all")
        candidates = ["A100-80GB", "L4-24GB"]

        # Should allow exploration without constraint
        exploration_count = 0
        total_decisions = 100

        for _ in range(total_decisions):
            result = model.score_job(features, candidates)
            if result.terms["sigma"] > 1.0:
                exploration_count += 1

        # Should see significant exploration with uninformed priors
        exploration_ratio = exploration_count / total_decisions
        assert exploration_ratio > 0.5, \
            f"With cap=1.0, should see substantial exploration, got {exploration_ratio:.3f}"

    def test_exploration_metrics(self):
        """Test that exploration metrics are tracked correctly."""
        features = JobFeatures(job_type="metrics-test")
        candidates = ["A100-80GB"]

        # Initial state
        initial_metrics = self.model.get_metrics()
        assert initial_metrics["scheduler_decisions_total"] == 0
        assert initial_metrics["exploration_ratio"] == 0.0

        # Make some decisions
        for _ in range(10):
            self.model.score_job(features, candidates)

        # Check updated metrics
        updated_metrics = self.model.get_metrics()
        assert updated_metrics["scheduler_decisions_total"] == 10
        assert 0.0 <= updated_metrics["exploration_ratio"] <= 1.0

    def test_exploration_cap_boundary_conditions(self):
        """Test exploration cap at boundary values."""
        # Test very small cap
        config_small = SchedulerConfig(exploration_cap=0.01)
        model_small = SchedulerModel(config_small, ":memory:")

        features = JobFeatures(job_type="boundary-test")
        candidates = ["A100-80GB", "L4-24GB"]

        # Run decisions and check that exploration is very limited
        for _ in range(100):
            model_small.score_job(features, candidates)

        metrics = model_small.get_metrics()
        assert metrics["exploration_ratio"] <= 0.05  # Should be very low

        # Test cap very close to 1.0
        config_high = SchedulerConfig(exploration_cap=0.99)
        model_high = SchedulerModel(config_high, ":memory:")

        for _ in range(100):
            model_high.score_job(features, candidates)

        metrics_high = model_high.get_metrics()
        assert metrics_high["exploration_ratio"] <= 1.0  # Should not exceed 1.0
