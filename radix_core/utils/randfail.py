"""
Seeded failure injection for testing resilience and fault tolerance.

This module provides controlled, reproducible failure injection for testing
how the system handles various failure scenarios.
"""

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..logging import get_logger

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of failures that can be injected."""

    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_DISCONNECT = "network_disconnect"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PROCESS_CRASH = "process_crash"
    SLOW_RESPONSE = "slow_response"
    PARTIAL_FAILURE = "partial_failure"
    DATA_CORRUPTION = "data_corruption"
    AUTHENTICATION_ERROR = "auth_error"


@dataclass
class FailureConfig:
    """Configuration for a specific failure type."""

    failure_type: FailureType
    probability: float  # 0.0 to 1.0
    delay_range: tuple = (0.1, 2.0)  # Min/max delay in seconds
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Failure probability must be between 0.0 and 1.0")

        if self.metadata is None:
            self.metadata = {}


class FailureInjectionError(Exception):
    """Exception raised when failure injection is triggered."""

    def __init__(self, failure_type: FailureType, message: str, metadata: Dict[str, Any] = None):
        self.failure_type = failure_type
        self.metadata = metadata or {}
        super().__init__(message)


class RandomFailureInjector:
    """Seeded random failure injector for reproducible testing."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.failure_configs: Dict[str, FailureConfig] = {}
        self.injection_history: List[Dict[str, Any]] = []
        self.enabled = True

    def configure_failure(self, operation: str, config: FailureConfig):
        """Configure failure injection for a specific operation."""
        self.failure_configs[operation] = config
        logger.debug(
            "Failure injection configured",
            operation=operation,
            failure_type=config.failure_type.value,
            probability=config.probability,
        )

    def should_inject_failure(self, operation: str) -> bool:
        """Determine if failure should be injected for this operation."""
        if not self.enabled:
            return False

        if operation not in self.failure_configs:
            return False

        config = self.failure_configs[operation]
        return self.rng.random() < config.probability

    def inject_failure(self, operation: str) -> None:
        """Inject failure for the specified operation."""
        if not self.should_inject_failure(operation):
            return

        config = self.failure_configs[operation]

        # Add random delay if configured
        if config.delay_range:
            delay = self.rng.uniform(*config.delay_range)
            logger.warning(
                "Injecting failure delay",
                operation=operation,
                failure_type=config.failure_type.value,
                delay_seconds=delay,
            )
            time.sleep(delay)

        # Record injection in history
        injection_record = {
            "timestamp": time.time(),
            "operation": operation,
            "failure_type": config.failure_type.value,
            "metadata": config.metadata,
        }
        self.injection_history.append(injection_record)

        # Raise appropriate exception based on failure type
        error_messages = {
            FailureType.NETWORK_TIMEOUT: f"Network timeout in {operation}",
            FailureType.NETWORK_DISCONNECT: f"Network disconnected during {operation}",
            FailureType.RESOURCE_EXHAUSTION: f"Resource exhaustion in {operation}",
            FailureType.PROCESS_CRASH: f"Process crashed during {operation}",
            FailureType.SLOW_RESPONSE: f"Slow response in {operation}",
            FailureType.PARTIAL_FAILURE: f"Partial failure in {operation}",
            FailureType.DATA_CORRUPTION: f"Data corruption detected in {operation}",
            FailureType.AUTHENTICATION_ERROR: f"Authentication failed in {operation}",
        }

        message = error_messages.get(config.failure_type, f"Unknown failure in {operation}")

        logger.error(
            "Failure injected",
            operation=operation,
            failure_type=config.failure_type.value,
            message=message,
        )

        raise FailureInjectionError(config.failure_type, message, config.metadata)

    def enable(self):
        """Enable failure injection."""
        self.enabled = True
        logger.info("Failure injection enabled")

    def disable(self):
        """Disable failure injection."""
        self.enabled = False
        logger.info("Failure injection disabled")

    def reset_history(self):
        """Clear injection history."""
        self.injection_history.clear()
        logger.debug("Failure injection history cleared")

    def get_injection_stats(self) -> Dict[str, Any]:
        """Get statistics about failure injections."""
        if not self.injection_history:
            return {"total_injections": 0}

        # Count by failure type
        failure_counts = {}
        operation_counts = {}

        for record in self.injection_history:
            failure_type = record["failure_type"]
            operation = record["operation"]

            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
            operation_counts[operation] = operation_counts.get(operation, 0) + 1

        return {
            "total_injections": len(self.injection_history),
            "failure_type_counts": failure_counts,
            "operation_counts": operation_counts,
            "first_injection": self.injection_history[0]["timestamp"],
            "last_injection": self.injection_history[-1]["timestamp"],
        }


# Global failure injector instance
_global_injector: Optional[RandomFailureInjector] = None


def get_failure_injector() -> RandomFailureInjector:
    """Get the global failure injector instance."""
    global _global_injector
    if _global_injector is None:
        _global_injector = RandomFailureInjector()
    return _global_injector


def configure_failure_injection(
    operation: str,
    failure_type: FailureType,
    probability: float,
    delay_range: tuple = (0.1, 2.0),
    metadata: Dict[str, Any] = None,
):
    """Configure failure injection for an operation."""
    config = FailureConfig(
        failure_type=failure_type,
        probability=probability,
        delay_range=delay_range,
        metadata=metadata,
    )
    get_failure_injector().configure_failure(operation, config)


def seeded_failure(operation: str):
    """Inject failure for an operation if configured."""
    get_failure_injector().inject_failure(operation)


@contextmanager
def failure_injection_context(operation: str):
    """Context manager that injects failures at entry."""
    try:
        seeded_failure(operation)
        yield
    except FailureInjectionError:
        # Re-raise injection errors
        raise
    except Exception as e:
        # Log unexpected errors but don't suppress them
        logger.error(
            "Unexpected error in failure injection context", operation=operation, error=str(e)
        )
        raise


def failure_prone_operation(operation: str, metadata: Dict[str, Any] = None):
    """Decorator that adds failure injection to a function."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with failure_injection_context(operation):
                return func(*args, **kwargs)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# Predefined failure scenarios for common testing patterns
class FailureScenarios:
    """Predefined failure scenarios for common testing patterns."""

    @staticmethod
    def network_instability(operations: List[str], probability: float = 0.1):
        """Configure network instability failures."""
        for operation in operations:
            configure_failure_injection(
                operation=operation,
                failure_type=FailureType.NETWORK_TIMEOUT,
                probability=probability,
                delay_range=(0.5, 3.0),
            )

    @staticmethod
    def resource_pressure(operations: List[str], probability: float = 0.05):
        """Configure resource exhaustion failures."""
        for operation in operations:
            configure_failure_injection(
                operation=operation,
                failure_type=FailureType.RESOURCE_EXHAUSTION,
                probability=probability,
                delay_range=(1.0, 5.0),
            )

    @staticmethod
    def intermittent_failures(operations: List[str], probability: float = 0.02):
        """Configure intermittent random failures."""
        failure_types = [
            FailureType.SLOW_RESPONSE,
            FailureType.PARTIAL_FAILURE,
            FailureType.NETWORK_DISCONNECT,
        ]

        injector = get_failure_injector()
        for operation in operations:
            # Randomly assign failure type
            failure_type = injector.rng.choice(failure_types)
            configure_failure_injection(
                operation=operation,
                failure_type=failure_type,
                probability=probability,
                delay_range=(0.1, 1.0),
            )

    @staticmethod
    def chaos_monkey(operations: List[str], intensity: float = 0.1):
        """Configure chaos monkey style random failures."""
        all_failure_types = list(FailureType)
        injector = get_failure_injector()

        for operation in operations:
            # Random failure type and probability
            failure_type = injector.rng.choice(all_failure_types)
            probability = injector.rng.uniform(0.01, intensity)

            configure_failure_injection(
                operation=operation,
                failure_type=failure_type,
                probability=probability,
                delay_range=(0.1, 2.0),
                metadata={"chaos_monkey": True},
            )


def reset_failure_injection(seed: int = None):
    """Reset failure injection with optional new seed."""
    global _global_injector
    if seed is not None:
        _global_injector = RandomFailureInjector(seed)
    elif _global_injector is not None:
        _global_injector.reset_history()
        _global_injector.failure_configs.clear()


def disable_failure_injection():
    """Disable all failure injection."""
    injector = get_failure_injector()
    injector.disable()


def enable_failure_injection():
    """Enable failure injection."""
    injector = get_failure_injector()
    injector.enable()
