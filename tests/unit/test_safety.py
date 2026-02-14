"""Tests for safety guards, dry-run, and cost simulation."""

import pytest
from radix_core.dryrun import DryRunGuard, CostGuard, DeploymentGuard, NetworkGuard
from radix_core.cost_simulator import CostSimulator
from radix_core.errors import SafetyViolationError, CostCapExceededError


class TestDryRunGuard:
    def test_verify_safety_passes(self):
        DryRunGuard.verify_safety()  # Should not raise

    def test_protect_decorator_marks_function(self):
        @DryRunGuard.protect
        def my_func():
            return "real result"

        assert DryRunGuard.is_protected(my_func)

    def test_protect_simulates_operation(self):
        @DryRunGuard.protect
        def execute_something():
            return "real result"

        result = execute_something()
        # In dry-run mode the result comes from _simulate_operation, not the real function
        assert result is not None

    def test_operation_log(self):
        DryRunGuard.clear_operation_log()

        @DryRunGuard.protect
        def test_op():
            pass

        test_op()
        log = DryRunGuard.get_operation_log()
        assert len(log) >= 1

    def test_clear_operation_log(self):
        DryRunGuard.clear_operation_log()
        assert len(DryRunGuard.get_operation_log()) == 0


class TestCostGuard:
    def test_zero_cost_passes(self):
        CostGuard.check_cost(0.0, "test")

    def test_nonzero_cost_in_dry_run_raises(self):
        with pytest.raises(CostCapExceededError):
            CostGuard.check_cost(1.0, "expensive_op")


class TestDeploymentGuard:
    def test_safe_operation_allowed(self):
        DeploymentGuard.check_operation("process_batch")  # Should not raise

    def test_deploy_blocked(self):
        with pytest.raises(SafetyViolationError, match="forbidden"):
            DeploymentGuard.check_operation("deploy_model")

    def test_kubectl_blocked(self):
        with pytest.raises(SafetyViolationError, match="forbidden"):
            DeploymentGuard.check_operation("kubectl_apply")


class TestNetworkGuard:
    def test_localhost_allowed(self):
        NetworkGuard.check_network_operation("localhost")
        NetworkGuard.check_network_operation("127.0.0.1")

    def test_external_host_blocked(self):
        with pytest.raises(SafetyViolationError, match="forbidden"):
            NetworkGuard.check_network_operation("example.com")


class TestProductionModeSafety:
    """Tests for safety guards in production mode."""

    def test_dryrun_guard_executes_real_function_in_production(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "production")
        from radix_core.config import reset_config, RadixConfig, set_config
        reset_config()
        cfg = RadixConfig.from_env()
        set_config(cfg)

        @DryRunGuard.protect
        def real_function():
            return "real_result"

        result = real_function()
        assert result == "real_result"

    def test_verify_safety_production_requires_cost_caps(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "production")
        from radix_core.config import reset_config, RadixConfig, set_config
        reset_config()

        # Default production config has positive cost caps, should pass
        cfg = RadixConfig.from_env()
        set_config(cfg)
        DryRunGuard.verify_safety()  # Should not raise

    def test_cost_guard_allows_nonzero_in_production(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "production")
        from radix_core.config import reset_config, RadixConfig, set_config
        reset_config()
        cfg = RadixConfig.from_env()
        set_config(cfg)

        # Non-zero cost within cap should pass
        CostGuard.check_cost(5.0, "test_op")

    def test_cost_guard_enforces_cap_in_production(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "production")
        from radix_core.config import reset_config, RadixConfig, set_config
        reset_config()
        cfg = RadixConfig.from_env()
        set_config(cfg)

        # Cost exceeding cap should fail
        with pytest.raises(CostCapExceededError):
            CostGuard.check_cost(999.0, "expensive_op")

    def test_network_guard_allows_external_in_production(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "production")
        NetworkGuard.check_network_operation("example.com")  # Should not raise

    def test_deployment_guard_allows_deploy_in_production(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "production")
        from radix_core.config import reset_config, RadixConfig, set_config
        reset_config()
        cfg = RadixConfig.from_env()
        set_config(cfg)

        DeploymentGuard.check_operation("deploy_model")  # Should not raise


class TestCostSimulator:
    def test_estimate_job_cost_zero_in_dry_run(self, sample_job):
        sim = CostSimulator()
        estimate = sim.estimate_job_cost(sample_job)
        assert estimate.estimated_cost_usd == 0.0
        assert estimate.dry_run_mode is True

    def test_estimate_batch_cost_zero_in_dry_run(self):
        sim = CostSimulator()
        cost = sim.estimate_batch_cost(batch_size=100)
        assert cost == 0.0

    def test_estimate_schedule_cost(self, sample_jobs):
        sim = CostSimulator()
        estimate = sim.estimate_schedule_cost(sample_jobs)
        assert estimate.estimated_cost_usd == 0.0
        assert estimate.is_simulated is True

    def test_estimate_empty_schedule(self):
        sim = CostSimulator()
        estimate = sim.estimate_schedule_cost([])
        assert estimate.estimated_cost_usd == 0.0

    def test_estimate_swarm_cost(self):
        sim = CostSimulator()
        estimate = sim.estimate_swarm_cost(node_count=5, duration_hours=1.0)
        assert estimate.estimated_cost_usd == 0.0

    def test_check_cost_cap_zero_passes(self):
        sim = CostSimulator()
        sim.check_cost_cap(0.0)  # Should not raise

    def test_check_cost_cap_nonzero_raises(self):
        sim = CostSimulator()
        with pytest.raises(CostCapExceededError):
            sim.check_cost_cap(1.0, "expensive")

    def test_caching(self, sample_job):
        sim = CostSimulator()
        e1 = sim.estimate_job_cost(sample_job)
        e2 = sim.estimate_job_cost(sample_job)
        assert e1 is e2  # Same object from cache

    def test_clear_cache(self, sample_job):
        sim = CostSimulator()
        sim.estimate_job_cost(sample_job)
        assert len(sim.estimates_cache) > 0
        sim.clear_cache()
        assert len(sim.estimates_cache) == 0
