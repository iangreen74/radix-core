"""Shared test fixtures for radix-core."""

import pytest
from radix_core.config import reset_config, get_config
from radix_core.types import Job, ResourceRequirements


@pytest.fixture(autouse=True)
def _reset_global_config(monkeypatch):
    """Reset global config and ensure development mode for all tests."""
    monkeypatch.delenv("RADIX_MODE", raising=False)
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config():
    """Return the default RadixConfig."""
    return get_config()


@pytest.fixture
def sample_job():
    """Return a simple sample job."""
    return Job(
        name="test-job",
        command="echo hello",
        requirements=ResourceRequirements(cpu_cores=1.0, memory_mb=512),
        priority=1,
    )


@pytest.fixture
def sample_jobs():
    """Return a list of sample jobs with varying priorities."""
    jobs = []
    for i in range(5):
        job = Job(
            name=f"test-job-{i}",
            command=f"echo job-{i}",
            requirements=ResourceRequirements(cpu_cores=1.0, memory_mb=256 * (i + 1)),
            priority=i,
        )
        jobs.append(job)
    return jobs


@pytest.fixture
def available_resources():
    """Return generous available resources for scheduling tests."""
    return ResourceRequirements(
        cpu_cores=16.0,
        memory_mb=32768,
        gpu_count=0,
        gpu_memory_mb=0,
        storage_mb=100000,
        network_mbps=1000.0,
    )
