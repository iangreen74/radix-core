"""Tests for core type definitions."""

import pytest
from radix_core.types import (
    Job, JobStatus, JobResult, ResourceRequirements,
    ExecutionResult,
)
from datetime import datetime


class TestResourceRequirements:
    def test_defaults(self):
        r = ResourceRequirements()
        assert r.cpu_cores == 1.0
        assert r.memory_mb == 512
        assert r.gpu_count == 0
        assert r.requires_local_only is True

    def test_invalid_cpu(self):
        with pytest.raises(ValueError, match="cpu_cores must be positive"):
            ResourceRequirements(cpu_cores=0)

    def test_invalid_memory(self):
        with pytest.raises(ValueError, match="memory_mb must be positive"):
            ResourceRequirements(memory_mb=-1)

    def test_invalid_gpu(self):
        with pytest.raises(ValueError, match="gpu_count cannot be negative"):
            ResourceRequirements(gpu_count=-1)

    def test_local_only_enforced(self):
        with pytest.raises(ValueError, match="requires_local_only must be True"):
            ResourceRequirements(requires_local_only=False)

    def test_is_satisfied_by(self):
        small = ResourceRequirements(cpu_cores=1.0, memory_mb=512)
        large = ResourceRequirements(cpu_cores=4.0, memory_mb=4096)
        assert small.is_satisfied_by(large)
        assert not large.is_satisfied_by(small)

    def test_total_cost_always_zero(self):
        r = ResourceRequirements(cpu_cores=8.0, memory_mb=16384, gpu_count=2)
        assert r.total_cost_estimate() == 0.0


class TestJob:
    def test_create_command_job(self):
        job = Job(name="test", command="echo hi")
        assert job.status == JobStatus.PENDING
        assert job.command == "echo hi"
        assert job.function is None
        assert job.job_id  # auto-generated

    def test_create_function_job(self):
        job = Job(name="test", function=lambda: 42)
        assert job.function is not None

    def test_must_have_command_or_function(self):
        with pytest.raises(ValueError, match="Either command or function"):
            Job(name="empty")

    def test_cannot_have_both(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            Job(name="both", command="echo", function=lambda: 1)

    def test_auto_name(self):
        job = Job(command="echo hi")
        assert job.name.startswith("job_")

    def test_is_ready_to_run(self):
        job = Job(command="echo", dependencies=["dep-1", "dep-2"])
        assert not job.is_ready_to_run({"dep-1"})
        assert job.is_ready_to_run({"dep-1", "dep-2"})

    def test_estimated_duration(self):
        job = Job(command="echo", requirements=ResourceRequirements())
        assert job.estimated_duration() > 0

    def test_to_dict(self):
        job = Job(name="test", command="echo hi")
        d = job.to_dict()
        assert d["name"] == "test"
        assert d["command"] == "echo hi"
        assert d["status"] == "PENDING"
        assert "job_id" in d


class TestJobResult:
    def test_succeeded(self):
        result = JobResult(job_id="abc", status=JobStatus.COMPLETED)
        assert result.succeeded is True

    def test_failed(self):
        result = JobResult(job_id="abc", status=JobStatus.FAILED, error_message="boom")
        assert result.succeeded is False

    def test_duration_none_when_no_times(self):
        result = JobResult(job_id="abc", status=JobStatus.COMPLETED)
        assert result.duration_seconds is None

    def test_duration_calculated(self):
        t1 = datetime(2024, 1, 1, 0, 0, 0)
        t2 = datetime(2024, 1, 1, 0, 0, 10)
        result = JobResult(job_id="abc", status=JobStatus.COMPLETED, start_time=t1, end_time=t2)
        assert result.duration_seconds == 10.0

    def test_to_dict(self):
        result = JobResult(job_id="abc", status=JobStatus.COMPLETED)
        d = result.to_dict()
        assert d["job_id"] == "abc"
        assert d["succeeded"] is True


class TestExecutionResult:
    def test_add_job_result(self):
        er = ExecutionResult(
            plan_id="p1", total_jobs=2, completed_jobs=0, failed_jobs=0,
            cancelled_jobs=0, start_time=datetime.utcnow(),
        )
        er.add_job_result(JobResult(job_id="j1", status=JobStatus.COMPLETED, cpu_time_seconds=1.5))
        assert er.completed_jobs == 1
        assert er.total_cpu_time_seconds == 1.5

        er.add_job_result(JobResult(job_id="j2", status=JobStatus.FAILED))
        assert er.failed_jobs == 1

    def test_success_rate(self):
        er = ExecutionResult(
            plan_id="p1", total_jobs=4, completed_jobs=3, failed_jobs=1,
            cancelled_jobs=0, start_time=datetime.utcnow(),
        )
        assert er.success_rate == 0.75
