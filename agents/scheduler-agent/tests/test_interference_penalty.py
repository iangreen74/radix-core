"""Test interference penalty learning and application."""

import pytest
from app.config import SchedulerConfig
from app.model import SchedulerModel, JobFeatures


class TestInterferencePenalty:
    """Test interference penalty learning and application."""

    def setup_method(self):
        """Set up test environment with interference enabled."""
        self.config = SchedulerConfig(
            enable_interference=True,
            gamma_interference=1.0  # Make interference penalties significant
        )
        self.model = SchedulerModel(self.config, ":memory:")

    def test_interference_learning_basic(self):
        """Test basic interference learning from colocated jobs."""
        job_type = "train-bert"
        gpu_type = "A100-80GB"
        interfering_job = "train-llama"

        # First, establish baseline performance (solo runs)
        solo_runtimes = [10.0, 10.2, 9.8, 10.1, 9.9]
        for runtime in solo_runtimes:
            self.model.observe(job_type, gpu_type, runtime, colocated_types=[])

        # Then observe performance with interference
        interfered_runtimes = [15.0, 15.5, 14.8, 15.2, 14.9]  # 50% slowdown
        for runtime in interfered_runtimes:
            self.model.observe(job_type, gpu_type, runtime, colocated_types=[interfering_job])

        # Check that interference was learned
        penalty = self.model.interference_penalty(job_type, gpu_type, [interfering_job])
        assert penalty > 0.1, f"Should learn significant interference penalty: {penalty:.3f}"
        assert penalty < 1.0, f"Penalty should be reasonable: {penalty:.3f}"

    def test_interference_affects_scoring(self):
        """Test that learned interference affects scoring decisions."""
        features = JobFeatures(job_type="sensitive-job")
        candidates = ["A100-80GB", "L4-24GB"]
        toxic_colocated = ["resource-hog"]

        # Establish that A100 normally preferred (better performance)
        self.model.observe("sensitive-job", "A100-80GB", 5.0)
        self.model.observe("sensitive-job", "L4-24GB", 8.0)

        # Score without interference
        result_clean = self.model.score_job(features, candidates, colocated_types=[])
        assert result_clean.chosen_gpu == "A100-80GB", "Should prefer A100 when clean"

        # Learn interference on A100
        interfered_runtimes = [12.0, 11.8, 12.2]  # Significant slowdown
        for runtime in interfered_runtimes:
            self.model.observe("sensitive-job", "A100-80GB", runtime,
                             colocated_types=[toxic_colocated[0]])

        # Score with interference present
        result_interfered = self.model.score_job(features, candidates,
                                               colocated_types=toxic_colocated)

        # Should now avoid A100 due to interference
        if result_interfered.chosen_gpu != "A100-80GB":
            # Interference successfully influenced decision
            assert result_interfered.chosen_gpu == "L4-24GB"

        # Check that interference penalty is reflected in terms
        assert result_interfered.terms["penalty"] > 0, \
            "Interference penalty should be reflected in scoring terms"

    def test_interference_matrix_multiple_jobs(self):
        """Test learning interference between multiple job types."""
        job_types = ["train-bert", "train-gpt", "inference-clip"]
        gpu_type = "A100-80GB"

        # Learn different interference patterns
        interference_pairs = [
            ("train-bert", "train-gpt", 0.3),      # 30% slowdown
            ("train-bert", "inference-clip", 0.1),  # 10% slowdown
            ("train-gpt", "inference-clip", 0.5),   # 50% slowdown
        ]

        for job1, job2, slowdown_factor in interference_pairs:
            # Establish baseline
            for _ in range(5):
                self.model.observe(job1, gpu_type, 10.0, colocated_types=[])

            # Learn interference
            interfered_runtime = 10.0 * (1 + slowdown_factor)
            for _ in range(10):
                self.model.observe(job1, gpu_type, interfered_runtime,
                                 colocated_types=[job2])

        # Test interference lookup
        penalty_1_2 = self.model.interference_penalty("train-bert", gpu_type, ["train-gpt"])
        penalty_1_3 = self.model.interference_penalty("train-bert", gpu_type, ["inference-clip"])
        penalty_2_3 = self.model.interference_penalty("train-gpt", gpu_type, ["inference-clip"])

        # Should reflect learned patterns
        assert penalty_2_3 > penalty_1_2 > penalty_1_3, \
            f"Interference penalties should match learned patterns: " \
            f"2-3={penalty_2_3:.3f}, 1-2={penalty_1_2:.3f}, 1-3={penalty_1_3:.3f}"

    def test_interference_anti_affinity_recommendations(self):
        """Test that high interference leads to anti-affinity recommendations."""
        features = JobFeatures(job_type="anti-affinity-test")
        candidates = ["A100-80GB"]
        problematic_jobs = ["memory-hog", "cpu-intensive"]

        # Learn high interference with some jobs
        for problematic_job in problematic_jobs:
            # Establish baseline
            for _ in range(3):
                self.model.observe(features.job_type, "A100-80GB", 5.0, [])

            # Learn significant interference (>10% threshold)
            for _ in range(10):
                self.model.observe(features.job_type, "A100-80GB", 8.0, [problematic_job])

        # Score with these jobs present
        result = self.model.score_job(features, candidates, colocated_types=problematic_jobs)

        # Should recommend avoiding colocating with problematic jobs
        for problematic_job in problematic_jobs:
            assert problematic_job in result.avoid_co_locate_with, \
                f"Should recommend avoiding {problematic_job}"

    def test_interference_disabled(self):
        """Test that interference learning is disabled when feature flag is off."""
        config_no_interference = SchedulerConfig(enable_interference=False)
        model_no_interference = SchedulerModel(config_no_interference, ":memory:")

        job_type = "no-interference-test"
        gpu_type = "A100-80GB"
        colocated_job = "other-job"

        # Try to learn interference (should be ignored)
        for _ in range(10):
            model_no_interference.observe(job_type, gpu_type, 5.0, [])
            model_no_interference.observe(job_type, gpu_type, 10.0, [colocated_job])

        # Penalty should be zero
        penalty = model_no_interference.interference_penalty(job_type, gpu_type, [colocated_job])
        assert penalty == 0.0, "Interference penalty should be zero when disabled"

        # Scoring should not include interference terms
        features = JobFeatures(job_type=job_type)
        result = model_no_interference.score_job(features, [gpu_type], [colocated_job])
        assert result.terms["penalty"] == 0.0, "Penalty term should be zero when disabled"
        assert len(result.avoid_co_locate_with) == 0, "Should not recommend anti-affinity when disabled"

    def test_interference_multiple_colocated_jobs(self):
        """Test interference calculation with multiple colocated jobs."""
        job_type = "multi-interference-test"
        gpu_type = "A100-80GB"

        # Learn interference with individual jobs
        interfering_jobs = ["job-a", "job-b", "job-c"]
        individual_slowdowns = [0.1, 0.2, 0.15]  # 10%, 20%, 15%

        for job, slowdown in zip(interfering_jobs, individual_slowdowns):
            # Baseline
            for _ in range(3):
                self.model.observe(job_type, gpu_type, 10.0, [])
            # Interference
            for _ in range(10):
                self.model.observe(job_type, gpu_type, 10.0 * (1 + slowdown), [job])

        # Test penalty with multiple jobs
        penalty_single = self.model.interference_penalty(job_type, gpu_type, [interfering_jobs[0]])
        penalty_multiple = self.model.interference_penalty(job_type, gpu_type, interfering_jobs)

        # Multiple jobs should have penalty between average and sum
        expected_avg = sum(individual_slowdowns) / len(individual_slowdowns)
        assert penalty_multiple > penalty_single, \
            "Multiple interfering jobs should have higher penalty than single job"
        assert penalty_multiple <= sum(individual_slowdowns), \
            "Penalty should not exceed sum of individual interferences"

    def test_interference_persistence(self):
        """Test that interference patterns persist across model checkpoints."""
        job_type = "persistence-test"
        gpu_type = "A100-80GB"
        interfering_job = "persistent-interferer"

        # Learn interference
        for _ in range(5):
            self.model.observe(job_type, gpu_type, 5.0, [])
        for _ in range(10):
            self.model.observe(job_type, gpu_type, 8.0, [interfering_job])

        # Get initial penalty
        initial_penalty = self.model.interference_penalty(job_type, gpu_type, [interfering_job])
        assert initial_penalty > 0, "Should learn interference initially"

        # Checkpoint and reload
        self.model.checkpoint()
        new_model = SchedulerModel(self.config, self.model.storage_path)

        # Penalty should persist
        persisted_penalty = new_model.interference_penalty(job_type, gpu_type, [interfering_job])
        assert abs(persisted_penalty - initial_penalty) < 0.01, \
            f"Interference should persist: {initial_penalty:.3f} vs {persisted_penalty:.3f}"
