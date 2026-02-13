"""
High-precision timing utilities for performance measurement and SLA monitoring.
"""

import time
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingResult:
    """Result of a timing operation."""

    operation: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    metadata: Dict[str, Any]

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000

    @property
    def duration_us(self) -> float:
        """Duration in microseconds."""
        return self.duration_seconds * 1_000_000


class Timer:
    """High-precision timer for measuring operation durations."""

    def __init__(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.start_datetime: Optional[datetime] = None
        self.end_time: Optional[float] = None
        self.end_datetime: Optional[datetime] = None
        self.success = True

    def start(self) -> 'Timer':
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.start_datetime = datetime.utcnow()
        logger.debug("Timer started", operation=self.operation, **self.metadata)
        return self

    def stop(self) -> TimingResult:
        """Stop the timer and return results."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.perf_counter()
        self.end_datetime = datetime.utcnow()

        duration = self.end_time - self.start_time

        result = TimingResult(
            operation=self.operation,
            start_time=self.start_datetime,
            end_time=self.end_datetime,
            duration_seconds=duration,
            success=self.success,
            metadata=self.metadata
        )

        logger.debug("Timer stopped",
                    operation=self.operation,
                    duration_ms=result.duration_ms,
                    success=self.success,
                    **self.metadata)

        return result

    def mark_failure(self):
        """Mark the operation as failed."""
        self.success = False

    def __enter__(self) -> 'Timer':
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.mark_failure()
        self.stop()


class AsyncTimer:
    """Async version of Timer for async operations."""

    def __init__(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.start_datetime: Optional[datetime] = None
        self.end_time: Optional[float] = None
        self.end_datetime: Optional[datetime] = None
        self.success = True

    async def start(self) -> 'AsyncTimer':
        """Start the async timer."""
        self.start_time = time.perf_counter()
        self.start_datetime = datetime.utcnow()
        logger.debug("Async timer started", operation=self.operation, **self.metadata)
        return self

    async def stop(self) -> TimingResult:
        """Stop the async timer and return results."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.perf_counter()
        self.end_datetime = datetime.utcnow()

        duration = self.end_time - self.start_time

        result = TimingResult(
            operation=self.operation,
            start_time=self.start_datetime,
            end_time=self.end_datetime,
            duration_seconds=duration,
            success=self.success,
            metadata=self.metadata
        )

        logger.debug("Async timer stopped",
                    operation=self.operation,
                    duration_ms=result.duration_ms,
                    success=self.success,
                    **self.metadata)

        return result

    def mark_failure(self):
        """Mark the operation as failed."""
        self.success = False

    async def __aenter__(self) -> 'AsyncTimer':
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is not None:
            self.mark_failure()
        await self.stop()


@contextmanager
def time_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for timing operations."""
    timer = Timer(operation, metadata)
    try:
        timer.start()
        yield timer
    except Exception:
        timer.mark_failure()
        raise
    finally:
        result = timer.stop()
        # Log timing result for monitoring
        logger.info("Operation timed",
                   operation=operation,
                   duration_ms=result.duration_ms,
                   success=result.success,
                   **metadata or {})


@asynccontextmanager
async def async_time_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Async context manager for timing operations."""
    timer = AsyncTimer(operation, metadata)
    try:
        await timer.start()
        yield timer
    except Exception:
        timer.mark_failure()
        raise
    finally:
        result = await timer.stop()
        # Log timing result for monitoring
        logger.info("Async operation timed",
                   operation=operation,
                   duration_ms=result.duration_ms,
                   success=result.success,
                   **metadata or {})


class SLAMonitor:
    """Monitor operations against SLA targets."""

    def __init__(self, sla_targets: Dict[str, float]):
        """
        Initialize SLA monitor.

        Args:
            sla_targets: Dict mapping operation names to target duration in seconds
        """
        self.sla_targets = sla_targets
        self.violations: Dict[str, int] = {}
        self.measurements: Dict[str, list] = {}

    def check_sla(self, result: TimingResult) -> bool:
        """
        Check if timing result meets SLA.

        Args:
            result: Timing result to check

        Returns:
            True if SLA is met, False otherwise
        """
        if result.operation not in self.sla_targets:
            return True  # No SLA defined

        target = self.sla_targets[result.operation]
        meets_sla = result.duration_seconds <= target

        # Track measurements
        if result.operation not in self.measurements:
            self.measurements[result.operation] = []
        self.measurements[result.operation].append(result.duration_seconds)

        # Track violations
        if not meets_sla:
            if result.operation not in self.violations:
                self.violations[result.operation] = 0
            self.violations[result.operation] += 1

            logger.warning("SLA violation",
                          operation=result.operation,
                          duration_ms=result.duration_ms,
                          target_ms=target * 1000,
                          violation_count=self.violations[result.operation])

        return meets_sla

    def get_sla_stats(self, operation: str) -> Dict[str, Any]:
        """Get SLA statistics for an operation."""
        if operation not in self.measurements:
            return {"error": "No measurements found"}

        measurements = self.measurements[operation]
        target = self.sla_targets.get(operation, float('inf'))
        violations = self.violations.get(operation, 0)

        # Calculate percentiles
        sorted_measurements = sorted(measurements)
        count = len(sorted_measurements)

        p50_idx = int(0.5 * count)
        p95_idx = int(0.95 * count)
        p99_idx = int(0.99 * count)

        return {
            "operation": operation,
            "target_seconds": target,
            "total_measurements": count,
            "violations": violations,
            "violation_rate": violations / count if count > 0 else 0,
            "min_seconds": min(measurements),
            "max_seconds": max(measurements),
            "mean_seconds": sum(measurements) / count,
            "p50_seconds": sorted_measurements[min(p50_idx, count - 1)],
            "p95_seconds": sorted_measurements[min(p95_idx, count - 1)],
            "p99_seconds": sorted_measurements[min(p99_idx, count - 1)],
        }


# Global SLA monitor instance
_global_sla_monitor: Optional[SLAMonitor] = None


def get_sla_monitor() -> Optional[SLAMonitor]:
    """Get the global SLA monitor instance."""
    return _global_sla_monitor


def set_sla_targets(targets: Dict[str, float]):
    """Set SLA targets and initialize global monitor."""
    global _global_sla_monitor
    _global_sla_monitor = SLAMonitor(targets)


def check_operation_sla(result: TimingResult) -> bool:
    """Check if operation result meets SLA using global monitor."""
    monitor = get_sla_monitor()
    if monitor is None:
        return True  # No SLA monitoring configured
    return monitor.check_sla(result)


# Decorator for automatic timing and SLA checking
def timed_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for automatic operation timing."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with time_operation(operation, metadata) as timer:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    timer.mark_failure()
                    raise

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def async_timed_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for automatic async operation timing."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            async with async_time_operation(operation, metadata) as timer:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception:
                    timer.mark_failure()
                    raise

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
