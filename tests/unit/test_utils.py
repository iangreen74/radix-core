"""Tests for utility modules (timers, randfail)."""

import time

import pytest

from radix_core.utils.randfail import FailureType, RandomFailureInjector
from radix_core.utils.timers import (
    SLAMonitor,
    Timer,
    TimingResult,
    time_operation,
)


class TestTimer:
    def test_basic_timing(self):
        t = Timer("test_op")
        t.start()
        time.sleep(0.01)
        result = t.stop()
        assert isinstance(result, TimingResult)
        assert result.duration_seconds >= 0.01
        assert result.success is True

    def test_stop_without_start_raises(self):
        t = Timer("test_op")
        with pytest.raises(ValueError, match="Timer not started"):
            t.stop()

    def test_mark_failure(self):
        t = Timer("test_op")
        t.start()
        t.mark_failure()
        result = t.stop()
        assert result.success is False

    def test_context_manager(self):
        with Timer("test_op"):
            time.sleep(0.01)
        # Timer stopped after context exit

    def test_context_manager_on_exception(self):
        with pytest.raises(ValueError):
            with Timer("test_op"):
                raise ValueError("boom")
        # Timer should have recorded failure


class TestTimingResult:
    def test_duration_ms(self):
        from datetime import datetime

        result = TimingResult(
            operation="test",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=0.5,
            success=True,
            metadata={},
        )
        assert result.duration_ms == 500.0
        assert result.duration_us == 500_000.0


class TestTimeOperation:
    def test_basic_usage(self):
        with time_operation("test_op"):
            time.sleep(0.01)
        # Should not raise

    def test_failure_tracking(self):
        with pytest.raises(RuntimeError):
            with time_operation("test_op"):
                raise RuntimeError("fail")


class TestSLAMonitor:
    def test_meets_sla(self):
        from datetime import datetime

        monitor = SLAMonitor(sla_targets={"fast_op": 1.0})
        result = TimingResult(
            operation="fast_op",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=0.5,
            success=True,
            metadata={},
        )
        assert monitor.check_sla(result) is True
        assert monitor.violations.get("fast_op", 0) == 0

    def test_violates_sla(self):
        from datetime import datetime

        monitor = SLAMonitor(sla_targets={"fast_op": 0.1})
        result = TimingResult(
            operation="fast_op",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=0.5,
            success=True,
            metadata={},
        )
        assert monitor.check_sla(result) is False
        assert monitor.violations["fast_op"] == 1

    def test_unknown_operation_passes(self):
        from datetime import datetime

        monitor = SLAMonitor(sla_targets={"other_op": 1.0})
        result = TimingResult(
            operation="unknown_op",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=99.0,
            success=True,
            metadata={},
        )
        assert monitor.check_sla(result) is True

    def test_get_sla_stats(self):
        from datetime import datetime

        monitor = SLAMonitor(sla_targets={"op": 1.0})
        for dur in [0.1, 0.2, 0.3, 0.5, 2.0]:
            result = TimingResult(
                operation="op",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=dur,
                success=True,
                metadata={},
            )
            monitor.check_sla(result)

        stats = monitor.get_sla_stats("op")
        assert stats["total_measurements"] == 5
        assert stats["violations"] == 1
        assert stats["min_seconds"] == 0.1
        assert stats["max_seconds"] == 2.0


class TestRandomFailureInjector:
    def test_no_failure_when_disabled(self):
        injector = RandomFailureInjector(seed=42)
        injector.enabled = False
        assert injector.should_inject_failure("test_component") is False

    def test_no_failure_for_unconfigured_operation(self):
        injector = RandomFailureInjector(seed=42)
        assert injector.should_inject_failure("not_configured") is False

    def test_failure_types_defined(self):
        assert FailureType.PROCESS_CRASH is not None
        assert FailureType.NETWORK_TIMEOUT is not None
        assert FailureType.DATA_CORRUPTION is not None
