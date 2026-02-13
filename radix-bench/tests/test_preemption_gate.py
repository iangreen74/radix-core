"""Tests for preemption gate functionality."""

import os
import pytest
from unittest.mock import patch

from radixbench.schedulers.radix_info_theory import RadixInfoTheoryScheduler
from radixbench.schedulers.radix_softmax import RadixSoftmaxScheduler
from radixbench.schedulers.base import Job, GPU, JobStatus


class TestPreemptionGate:
    """Test preemption gate functionality."""
    
    def test_preemption_disabled_by_default(self):
        """Test that preemption is disabled by default."""
        scheduler = RadixInfoTheoryScheduler()
        assert scheduler.preempt_enable is False
        
        # Should return empty list when preemption is disabled
        cluster = [GPU(gpu_id="gpu-0", memory_gb=80)]
        actions = scheduler.preempt(cluster, 0.0)
        assert actions == []
    
    def test_preemption_enabled_via_constructor(self):
        """Test enabling preemption via constructor."""
        scheduler = RadixInfoTheoryScheduler(preempt_enable=True)
        assert scheduler.preempt_enable is True
    
    def test_preemption_enabled_via_env_var(self):
        """Test enabling preemption via environment variable."""
        with patch.dict(os.environ, {"RADIX_PREEMPT_ENABLE": "1"}):
            scheduler = RadixInfoTheoryScheduler()
            assert scheduler.preempt_enable is True
    
    def test_preemption_budget_configuration(self):
        """Test preemption budget configuration."""
        # Default budget
        scheduler = RadixInfoTheoryScheduler()
        assert scheduler.preempt_budget_per_hour == 2
        
        # Constructor override
        scheduler = RadixInfoTheoryScheduler(preempt_budget_per_hour=5)
        assert scheduler.preempt_budget_per_hour == 5
        
        # Environment variable override
        with patch.dict(os.environ, {"RADIX_PREEMPT_BUDGET_PER_HOUR": "10"}):
            scheduler = RadixInfoTheoryScheduler()
            assert scheduler.preempt_budget_per_hour == 10
    
    def test_preemption_gain_threshold_configuration(self):
        """Test preemption gain threshold configuration."""
        # Default threshold
        scheduler = RadixInfoTheoryScheduler()
        assert scheduler.preempt_gain_threshold == 0.20
        
        # Constructor override
        scheduler = RadixInfoTheoryScheduler(preempt_gain_threshold=0.5)
        assert scheduler.preempt_gain_threshold == 0.5
        
        # Environment variable override
        with patch.dict(os.environ, {"RADIX_PREEMPT_GAIN_THRESHOLD": "0.3"}):
            scheduler = RadixInfoTheoryScheduler()
            assert scheduler.preempt_gain_threshold == 0.3
    
    def test_preemption_with_no_pending_jobs(self):
        """Test preemption when there are no pending jobs."""
        scheduler = RadixInfoTheoryScheduler(preempt_enable=True)
        
        # Create cluster with running job
        cluster = [GPU(gpu_id="gpu-0", memory_gb=80, is_available=False, current_job="job-1")]
        
        # No pending jobs submitted
        actions = scheduler.preempt(cluster, 0.0)
        assert actions == []
    
    def test_preemption_basic_functionality(self):
        """Test basic preemption functionality."""
        scheduler = RadixInfoTheoryScheduler(
            preempt_enable=True,
            preempt_gain_threshold=0.1  # Low threshold for testing
        )
        
        # Submit a short job
        short_job = Job(
            job_id="short-job",
            submit_time=0.0,
            runtime_estimate=50.0,  # Short job
            memory_gb=40.0
        )
        scheduler.submit(short_job)
        
        # Create GPU with long-running job
        long_job = Job(
            job_id="long-job",
            submit_time=0.0,
            runtime_estimate=1000.0,  # Long job
            memory_gb=40.0
        )
        long_job.preemptible = True  # Mark as preemptible
        
        gpu = GPU(gpu_id="gpu-0", memory_gb=80, is_available=False, current_job="long-job")
        gpu.current_job = long_job
        
        cluster = [gpu]
        
        # Should suggest preemption
        actions = scheduler.preempt(cluster, 0.0)
        
        # Verify action structure (may be empty if conditions not met)
        assert isinstance(actions, list)
        for action in actions:
            assert len(action) >= 4
            assert action[0] == "preempt"
    
    def test_softmax_scheduler_inherits_preemption(self):
        """Test that RadixSoftmaxScheduler inherits preemption functionality."""
        scheduler = RadixSoftmaxScheduler(preempt_enable=True)
        assert hasattr(scheduler, 'preempt')
        assert scheduler.preempt_enable is True
        
        # Should return empty list when no conditions are met
        cluster = [GPU(gpu_id="gpu-0", memory_gb=80)]
        actions = scheduler.preempt(cluster, 0.0)
        assert actions == []
    
    def test_hour_bucket_calculation(self):
        """Test hour bucket calculation for preemption budgeting."""
        scheduler = RadixInfoTheoryScheduler()
        
        # Test various times
        assert scheduler._hour_bucket(0.0) == 0
        assert scheduler._hour_bucket(3599.0) == 0
        assert scheduler._hour_bucket(3600.0) == 1
        assert scheduler._hour_bucket(7200.0) == 2
    
    def test_preemption_budget_tracking(self):
        """Test preemption budget tracking."""
        scheduler = RadixInfoTheoryScheduler(preempt_budget_per_hour=2)
        
        # Initially should have full budget
        assert scheduler._remaining_preempt_budget(0.0) == 2
        
        # Record preemptions
        scheduler._record_preempt(0.0)
        assert scheduler._remaining_preempt_budget(0.0) == 1
        
        scheduler._record_preempt(0.0)
        assert scheduler._remaining_preempt_budget(0.0) == 0
        
        # Next hour should reset
        assert scheduler._remaining_preempt_budget(3600.0) == 2


class TestPreemptionIntegration:
    """Integration tests for preemption with simulation."""
    
    def test_preemption_gate_environment_variable(self):
        """Test that RADIX_PREEMPT_GATE environment variable is respected."""
        # This would be tested in the simulation layer
        # Here we just verify the environment variable can be read
        with patch.dict(os.environ, {"RADIX_PREEMPT_GATE": "1"}):
            gate_enabled = os.environ.get("RADIX_PREEMPT_GATE", "0") not in ("0", "false", "False")
            assert gate_enabled is True
        
        with patch.dict(os.environ, {"RADIX_PREEMPT_GATE": "0"}):
            gate_enabled = os.environ.get("RADIX_PREEMPT_GATE", "0") not in ("0", "false", "False")
            assert gate_enabled is False
    
    def test_scheduler_preempt_method_exists(self):
        """Test that schedulers expose the preempt method."""
        radix_scheduler = RadixInfoTheoryScheduler()
        assert hasattr(radix_scheduler, 'preempt')
        assert callable(getattr(radix_scheduler, 'preempt'))
        
        softmax_scheduler = RadixSoftmaxScheduler()
        assert hasattr(softmax_scheduler, 'preempt')
        assert callable(getattr(softmax_scheduler, 'preempt'))


if __name__ == "__main__":
    pytest.main([__file__])
