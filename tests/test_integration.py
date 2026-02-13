"""
Integration tests for complete job pipeline.

Tests the end-to-end flow from job submission through scheduling,
placement, execution, and result collection.
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch
from dataclasses import asdict

# Add engine to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from radix_core.types import JobSpec, JobGraph, ExecutionResult, ResourceRequirements
from radix_core.config import get_config
from radix_core.scheduler.planner import JobPlanner
from radix_core.scheduler.placement import ResourcePlacer
from radix_core.executors.threadpool import ThreadPoolExecutor
from radix_core.batching.dynamic_batcher import DynamicBatcher
from radix_core.cost_simulator import get_cost_simulator
from radix_core.dryrun import DryRunGuard


class TestJobPipeline:
    """Test complete job processing pipeline."""

    @pytest.fixture
    def config(self):
        """Get system configuration."""
        return get_config()

    @pytest.fixture
    def job_planner(self, config):
        """Create job planner instance."""
        return JobPlanner(config)

    @pytest.fixture
    def resource_placer(self, config):
        """Create resource placer instance."""
        return ResourcePlacer(config)

    @pytest.fixture
    def executor(self, config):
        """Create executor instance."""
        return ThreadPoolExecutor(config)

    @pytest.fixture
    def batcher(self, config):
        """Create dynamic batcher instance."""
        return DynamicBatcher(config)

    @pytest.fixture
    def sample_jobs(self):
        """Create a set of test jobs."""
        return [
            JobSpec(
                job_id=f"job-{i:03d}",
                name=f"test_job_{i}",
                command=f"python -c 'import time; time.sleep(0.1); print(\"Job {i} result: {{}}}\".format({i}*10))'",
                resources=ResourceRequirements(
                    cpu_cores=1,
                    memory_gb=0.5,
                    gpu_count=0
                ),
                priority=i % 3 + 1,  # Priority 1-3
                max_runtime_seconds=10,
                dependencies=[] if i == 0 else [f"job-{i-1:03d}"] if i % 5 == 0 else []
            )
            for i in range(10)
        ]

    @pytest.fixture
    def ml_pipeline_jobs(self):
        """Create ML pipeline jobs (data prep -> training -> evaluation)."""
        return [
            JobSpec(
                job_id="data-prep-001",
                name="data_preparation",
                command="python -c 'print(\"Data prepared: 1000 samples\")'",
                resources=ResourceRequirements(cpu_cores=2, memory_gb=2.0),
                priority=3,
                max_runtime_seconds=30,
                dependencies=[],
                job_type="data_prep"
            ),
            JobSpec(
                job_id="train-001",
                name="model_training",
                command="python -c 'import time; time.sleep(0.5); print(\"Model trained: accuracy=0.95\")'",
                resources=ResourceRequirements(cpu_cores=4, memory_gb=8.0),
                priority=2,
                max_runtime_seconds=60,
                dependencies=["data-prep-001"],
                job_type="training"
            ),
            JobSpec(
                job_id="eval-001",
                name="model_evaluation",
                command="python -c 'print(\"Evaluation complete: test_accuracy=0.92\")'",
                resources=ResourceRequirements(cpu_cores=2, memory_gb=4.0),
                priority=1,
                max_runtime_seconds=30,
                dependencies=["train-001"],
                job_type="evaluation"
            )
        ]

    def test_single_job_pipeline(self, job_planner, resource_placer, executor, sample_jobs):
        """Test complete pipeline for a single job."""
        job = sample_jobs[0]

        # Step 1: Planning
        plan = job_planner.create_execution_plan([job])
        assert plan is not None
        assert len(plan.scheduled_jobs) == 1

        # Step 2: Resource placement
        placement = resource_placer.place_jobs(plan.scheduled_jobs)
        assert placement is not None
        assert len(placement.placements) == 1

        # Step 3: Safety verification
        DryRunGuard.verify_safety()

        # Step 4: Execution
        result = executor.execute_job(job)
        assert result.success is True
        assert result.job_id == job.job_id
        assert "Job 0 result: 0" in result.stdout

    def test_batch_job_pipeline(self, job_planner, resource_placer, executor, batcher, sample_jobs):
        """Test pipeline with batch processing."""
        # Select subset for batching
        batch_jobs = sample_jobs[:5]

        # Step 1: Planning
        plan = job_planner.create_execution_plan(batch_jobs)
        assert len(plan.scheduled_jobs) == 5

        # Step 2: Resource placement
        placement = resource_placer.place_jobs(plan.scheduled_jobs)
        assert len(placement.placements) == 5

        # Step 3: Batching
        batches = batcher.create_batches(placement.placements)
        assert len(batches) > 0

        # Step 4: Execute batches
        all_results = []
        for batch in batches:
            batch_results = executor.execute_batch(batch.jobs)
            all_results.extend(batch_results)

        # Verify all jobs completed
        assert len(all_results) == 5
        assert all(r.success for r in all_results)

    def test_dependency_pipeline(self, job_planner, resource_placer, executor, ml_pipeline_jobs):
        """Test pipeline with job dependencies."""
        # Step 1: Planning with dependency resolution
        plan = job_planner.create_execution_plan(ml_pipeline_jobs)

        # Should order jobs according to dependencies
        ordered_jobs = plan.scheduled_jobs
        job_ids = [job.job_id for job in ordered_jobs]

        # Data prep should come first
        assert job_ids.index("data-prep-001") < job_ids.index("train-001")
        assert job_ids.index("train-001") < job_ids.index("eval-001")

        # Step 2: Execute in dependency order
        results = {}
        for job in ordered_jobs:
            # Check dependencies are satisfied
            for dep_id in job.dependencies:
                assert dep_id in results
                assert results[dep_id].success

            # Execute job
            result = executor.execute_job(job)
            results[job.job_id] = result
            assert result.success

        # Verify complete pipeline
        assert len(results) == 3
        assert all(r.success for r in results.values())

    def test_cost_estimation_pipeline(self, job_planner, sample_jobs, config):
        """Test cost estimation throughout pipeline."""
        cost_simulator = get_cost_simulator()

        # Step 1: Individual job cost estimation
        job_costs = []
        for job in sample_jobs[:3]:
            cost = cost_simulator.estimate_job_cost(job)
            job_costs.append(cost)
            assert cost.estimated_cost_usd == 0.0  # Dry-run mode

        # Step 2: Batch cost estimation
        total_cost = cost_simulator.estimate_batch_cost(sample_jobs[:3])
        assert total_cost == 0.0  # Dry-run mode

        # Step 3: Plan-level cost estimation
        plan = job_planner.create_execution_plan(sample_jobs[:3])
        plan_cost = cost_simulator.estimate_plan_cost(plan)
        assert plan_cost.total_cost_usd == 0.0  # Dry-run mode

    def test_error_handling_pipeline(self, job_planner, resource_placer, executor):
        """Test error handling throughout pipeline."""
        # Create jobs with various error scenarios
        error_jobs = [
            JobSpec(
                job_id="success-job",
                name="success",
                command="echo 'success'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=1,
                max_runtime_seconds=5
            ),
            JobSpec(
                job_id="error-job",
                name="error",
                command="python -c 'raise ValueError(\"test error\")'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=1,
                max_runtime_seconds=5
            ),
            JobSpec(
                job_id="timeout-job",
                name="timeout",
                command="python -c 'import time; time.sleep(10)'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=1,
                max_runtime_seconds=1  # Very short timeout
            )
        ]

        # Process through pipeline
        plan = job_planner.create_execution_plan(error_jobs)
        placement = resource_placer.place_jobs(plan.scheduled_jobs)

        # Execute and collect results
        results = {}
        for job in placement.placements:
            result = executor.execute_job(job.job)
            results[job.job.job_id] = result

        # Verify error handling
        assert results["success-job"].success is True
        assert results["error-job"].success is False
        assert "ValueError" in results["error-job"].stderr
        assert results["timeout-job"].success is False
        assert "timeout" in results["timeout-job"].error_message.lower()

    def test_performance_monitoring_pipeline(self, job_planner, executor, sample_jobs):
        """Test performance monitoring throughout pipeline."""
        # Create performance test jobs
        perf_jobs = [
            JobSpec(
                job_id=f"perf-{i}",
                name=f"performance_test_{i}",
                command=f"python -c 'import time; time.sleep(0.{i+1}); print(\"Task {i} complete\")'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=1,
                max_runtime_seconds=10
            )
            for i in range(3)
        ]

        # Execute with timing
        start_time = time.time()
        plan = job_planner.create_execution_plan(perf_jobs)
        planning_time = time.time() - start_time

        start_time = time.time()
        results = []
        for job in plan.scheduled_jobs:
            result = executor.execute_job(job)
            results.append(result)
        execution_time = time.time() - start_time

        # Verify performance metrics
        assert planning_time < 1.0  # Planning should be fast
        assert all(r.execution_time_seconds > 0 for r in results)
        assert all(r.resource_usage is not None for r in results)

        # Log performance metrics
        total_job_time = sum(r.execution_time_seconds for r in results)
        print(f"\nPipeline Performance:")
        print(f"  Planning time: {planning_time:.3f}s")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Total job time: {total_job_time:.3f}s")
        print(f"  Parallelization efficiency: {total_job_time/execution_time:.2f}x")


class TestAdvancedPipelines:
    """Test advanced pipeline scenarios."""

    @pytest.fixture
    def config(self):
        return get_config()

    def test_mixed_workload_pipeline(self, config):
        """Test pipeline with mixed CPU/GPU workloads."""
        mixed_jobs = [
            JobSpec(
                job_id="cpu-intensive",
                name="cpu_task",
                command="python -c 'sum(range(100000))'",
                resources=ResourceRequirements(cpu_cores=4, memory_gb=1.0, gpu_count=0),
                priority=2,
                max_runtime_seconds=10,
                job_type="cpu_intensive"
            ),
            JobSpec(
                job_id="memory-intensive",
                name="memory_task",
                command="python -c 'x = [0] * 1000000; print(len(x))'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=2.0, gpu_count=0),
                priority=2,
                max_runtime_seconds=10,
                job_type="memory_intensive"
            ),
            JobSpec(
                job_id="gpu-optional",
                name="gpu_task",
                command="python -c 'print(\"GPU task (simulated)\")'",
                resources=ResourceRequirements(cpu_cores=2, memory_gb=4.0, gpu_count=0),  # No GPU in safe mode
                priority=1,
                max_runtime_seconds=15,
                job_type="ml_inference"
            )
        ]

        # Process mixed workload
        planner = JobPlanner(config)
        placer = ResourcePlacer(config)
        executor = ThreadPoolExecutor(config)

        plan = planner.create_execution_plan(mixed_jobs)
        placement = placer.place_jobs(plan.scheduled_jobs)

        # Execute with resource-aware scheduling
        results = []
        for job_placement in placement.placements:
            result = executor.execute_job(job_placement.job)
            results.append(result)

        assert all(r.success for r in results)
        assert len(results) == 3

    def test_pipeline_scalability(self, config):
        """Test pipeline scalability with large job counts."""
        # Create large number of jobs
        large_job_set = [
            JobSpec(
                job_id=f"scale-{i:04d}",
                name=f"scale_test_{i}",
                command=f"python -c 'print(\"Job {i} of 100\")'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=(i % 3) + 1,
                max_runtime_seconds=5
            )
            for i in range(100)
        ]

        planner = JobPlanner(config)
        batcher = DynamicBatcher(config)
        executor = ThreadPoolExecutor(config)

        # Test planning scalability
        start_time = time.time()
        plan = planner.create_execution_plan(large_job_set)
        planning_time = time.time() - start_time

        assert planning_time < 5.0  # Should handle 100 jobs quickly
        assert len(plan.scheduled_jobs) == 100

        # Test batching scalability
        start_time = time.time()
        batches = batcher.create_batches(plan.scheduled_jobs)
        batching_time = time.time() - start_time

        assert batching_time < 2.0
        assert len(batches) > 0

        # Execute subset to verify functionality
        sample_batch = batches[0]
        results = executor.execute_batch(sample_batch.jobs[:5])  # Just test first 5
        assert all(r.success for r in results)

    def test_failure_recovery_pipeline(self, config):
        """Test pipeline failure recovery mechanisms."""
        # Create jobs with some failures
        recovery_jobs = [
            JobSpec(
                job_id="step-1",
                name="reliable_step",
                command="echo 'Step 1 complete'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=3,
                max_runtime_seconds=5
            ),
            JobSpec(
                job_id="step-2-fail",
                name="failing_step",
                command="python -c 'raise Exception(\"Simulated failure\")'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=2,
                max_runtime_seconds=5,
                dependencies=["step-1"],
                retry_count=2
            ),
            JobSpec(
                job_id="step-3",
                name="recovery_step",
                command="echo 'Recovery complete'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=1,
                max_runtime_seconds=5,
                dependencies=["step-1"]  # Should still run despite step-2 failure
            )
        ]

        planner = JobPlanner(config)
        executor = ThreadPoolExecutor(config)

        plan = planner.create_execution_plan(recovery_jobs)

        # Execute with failure handling
        results = {}
        for job in plan.scheduled_jobs:
            # Check if dependencies are satisfied (ignoring failed jobs)
            can_run = True
            for dep_id in job.dependencies:
                if dep_id in results and not results[dep_id].success:
                    # In a real system, this might trigger alternative paths
                    if dep_id == "step-2-fail":
                        continue  # Allow step-3 to run despite step-2 failure
                    can_run = False
                    break

            if can_run:
                result = executor.execute_job(job)
                results[job.job_id] = result

        # Verify recovery behavior
        assert results["step-1"].success is True
        assert results["step-2-fail"].success is False
        assert results["step-3"].success is True  # Should run despite step-2 failure


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
