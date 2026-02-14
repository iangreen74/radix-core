"""Integration tests for end-to-end workflows."""

import pytest
from radix_core.config import get_config
from radix_core.types import Job, ResourceRequirements
from radix_core.scheduler.job_graph import JobGraph
from radix_core.scheduler.planner import GreedyPlanner
from radix_core.scheduler.placement import LocalPlacement
from radix_core.scheduler.policies import FIFOPolicy, PriorityPolicy
from radix_core.cost_simulator import CostSimulator
from radix_core.dryrun import DryRunGuard


@pytest.mark.integration
class TestSchedulingWorkflow:
    """Test the complete scheduling pipeline."""

    def test_plan_and_place(self):
        """Create jobs, plan them, and place them."""
        jobs = [
            Job(name="preprocess", command="echo preprocess", priority=2),
            Job(name="train", command="echo train", priority=1),
            Job(name="evaluate", command="echo eval", priority=0),
        ]

        planner = GreedyPlanner()
        plan = planner.create_execution_plan(jobs)
        assert len(plan.scheduled_jobs) == 3
        assert plan.dependencies_resolved is True

        strategy = LocalPlacement()
        placement_plan = strategy.place_jobs(plan.scheduled_jobs)
        assert len(placement_plan.placements) == 3

    def test_dag_scheduling(self):
        """Test DAG-based scheduling with dependencies."""
        a = Job(name="download", command="echo download")
        b = Job(name="preprocess", command="echo preprocess")
        c = Job(name="train", command="echo train")

        graph = JobGraph()
        graph.add_job(a)
        graph.add_job(b)
        graph.add_job(c)
        graph.add_dependency(a.job_id, b.job_id)
        graph.add_dependency(b.job_id, c.job_id)

        # Initially only a is ready
        ready = graph.get_ready_jobs()
        assert len(ready) == 1
        assert ready[0].job_id == a.job_id

        # Complete a -> b becomes ready
        graph.mark_job_completed(a.job_id)
        ready = graph.get_ready_jobs()
        assert len(ready) == 1
        assert ready[0].job_id == b.job_id

        # Complete b -> c becomes ready
        graph.mark_job_completed(b.job_id)
        ready = graph.get_ready_jobs()
        assert len(ready) == 1
        assert ready[0].job_id == c.job_id

    def test_policy_driven_scheduling(self):
        """Test scheduling with different policies."""
        available = ResourceRequirements(
            cpu_cores=16.0, memory_mb=32768,
            gpu_count=0, gpu_memory_mb=0,
            storage_mb=100000, network_mbps=1000.0,
        )

        jobs = [
            Job(name=f"job-{i}", command=f"echo {i}", priority=i)
            for i in range(5)
        ]

        # FIFO policy
        fifo = FIFOPolicy()
        fifo_decisions = fifo.select_jobs(jobs, available)
        assert len(fifo_decisions) == 5

        # Priority policy
        priority = PriorityPolicy()
        priority_decisions = priority.select_jobs(jobs, available)
        assert len(priority_decisions) == 5
        # Highest priority job should be scheduled first
        assert priority_decisions[0].job_id == jobs[-1].job_id


@pytest.mark.integration
class TestCostWorkflow:
    """Test cost estimation across the pipeline."""

    def test_job_cost_estimation(self):
        simulator = CostSimulator()
        job = Job(
            name="gpu-job",
            command="echo gpu",
            requirements=ResourceRequirements(cpu_cores=4.0, memory_mb=8192),
        )
        estimate = simulator.estimate_job_cost(job)
        assert estimate.estimated_cost_usd == 0.0
        assert estimate.cpu_core_hours > 0
        assert estimate.memory_gb_hours > 0

    def test_schedule_cost_estimation(self):
        simulator = CostSimulator()
        jobs = [
            Job(name=f"j{i}", command=f"echo {i}")
            for i in range(10)
        ]
        estimate = simulator.estimate_schedule_cost(jobs)
        assert estimate.estimated_cost_usd == 0.0
        assert estimate.duration_hours > 0


@pytest.mark.integration
class TestSafetyIntegration:
    """Test safety guards in realistic scenarios."""

    def test_all_safety_checks_pass(self):
        DryRunGuard.verify_safety()

    def test_config_is_safe_by_default(self):
        config = get_config()
        assert config.safety.dry_run is True
        assert config.safety.cost_cap_usd == 0.0
        assert config.safety.no_deploy_mode is True
        errors = config.validate()
        assert errors == []


@pytest.mark.integration
class TestProductionModeIntegration:
    """Test production mode configuration and execution."""

    def test_production_config_loads(self, monkeypatch):
        from radix_core.config import RadixConfig, reset_config
        monkeypatch.setenv("RADIX_MODE", "production")
        reset_config()
        cfg = RadixConfig.from_env()
        assert cfg.safety.dry_run is False
        assert cfg.safety.cost_cap_usd > 0
        assert cfg.safety.max_job_cost_usd > 0
        assert cfg.safety.no_deploy_mode is False

    def test_production_dryrun_guard_executes(self, monkeypatch):
        from radix_core.config import RadixConfig, reset_config, set_config
        monkeypatch.setenv("RADIX_MODE", "production")
        reset_config()
        cfg = RadixConfig.from_env()
        set_config(cfg)

        @DryRunGuard.protect
        def compute(x, y):
            return x + y

        assert compute(3, 4) == 7

    def test_production_cost_estimation(self, monkeypatch):
        from radix_core.config import RadixConfig, reset_config, set_config
        monkeypatch.setenv("RADIX_MODE", "production")
        reset_config()
        cfg = RadixConfig.from_env()
        set_config(cfg)

        simulator = CostSimulator()
        job = Job(
            name="prod-job",
            command="echo prod",
            requirements=ResourceRequirements(cpu_cores=4.0, memory_mb=8192),
        )
        estimate = simulator.estimate_job_cost(job)
        # In production mode with dry_run=False, real cost flows through
        assert estimate.dry_run_mode is False
