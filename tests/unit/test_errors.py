"""Tests for error hierarchy."""

import pytest
from radix_core.errors import (
    RadixError, SafetyViolationError, CostCapExceededError,
    ConfigurationError, ExecutionError, SchedulingError,
    ResourceError, DependencyError, raise_safety_violation, raise_cost_exceeded,
)


class TestRadixError:
    def test_basic_error(self):
        err = RadixError("something broke")
        assert "something broke" in str(err)

    def test_error_with_details(self):
        err = RadixError("oops", details={"key": "val"})
        assert err.details == {"key": "val"}
        assert "key=val" in str(err)


class TestSafetyViolationError:
    def test_with_suggestion(self):
        err = SafetyViolationError("bad thing", suggestion="fix it")
        assert "Safety violation" in str(err)
        assert "fix it" in str(err)


class TestCostCapExceededError:
    def test_attributes(self):
        err = CostCapExceededError(10.0, 5.0, "big_job")
        assert err.estimated_cost == 10.0
        assert err.cost_cap == 5.0
        assert err.operation == "big_job"
        assert "$10.00" in str(err)
        assert "$5.00" in str(err)


class TestConfigurationError:
    def test_field_info(self):
        err = ConfigurationError("max_parallelism", 0, "positive integer")
        assert err.field == "max_parallelism"
        assert "0" in str(err)


class TestExecutionError:
    def test_job_id(self):
        err = ExecutionError("job-123", "timed out")
        assert err.job_id == "job-123"
        assert "timed out" in str(err)


class TestSchedulingError:
    def test_with_job_ids(self):
        err = SchedulingError("no resources", job_ids=["j1", "j2"])
        assert err.job_ids == ["j1", "j2"]


class TestResourceError:
    def test_resource_type(self):
        err = ResourceError("GPU", "not available")
        assert err.resource_type == "GPU"


class TestDependencyError:
    def test_missing_deps(self):
        err = DependencyError("j1", ["j2", "j3"])
        assert err.missing_dependencies == ["j2", "j3"]


class TestConvenienceFunctions:
    def test_raise_safety_violation(self):
        with pytest.raises(SafetyViolationError):
            raise_safety_violation("test violation")

    def test_raise_cost_exceeded(self):
        with pytest.raises(CostCapExceededError):
            raise_cost_exceeded(10.0, 5.0)
