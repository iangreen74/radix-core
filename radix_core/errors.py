"""
Error Definitions for Radix

This module defines custom exception classes used throughout the Radix system
to provide clear, actionable error messages and proper error handling.
"""

from typing import Any, Dict, Optional


class RadixError(Exception):
    """Base exception class for all Radix-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class SafetyViolationError(RadixError):
    """Raised when safety guards are violated."""

    def __init__(self, violation: str, suggestion: Optional[str] = None, **details):
        message = f"Safety violation: {violation}"
        if suggestion:
            message += f". Suggestion: {suggestion}"

        super().__init__(message, details)
        self.violation = violation
        self.suggestion = suggestion


class CostCapExceededError(RadixError):
    """Raised when an operation would exceed the cost cap."""

    def __init__(
        self, estimated_cost: float, cost_cap: float, operation: str = "operation", **details
    ):
        message = (
            f"Cost cap exceeded: {operation} would cost ${estimated_cost:.2f} "
            f"but cap is ${cost_cap:.2f}"
        )

        super().__init__(
            message,
            {
                "estimated_cost": estimated_cost,
                "cost_cap": cost_cap,
                "operation": operation,
                **details,
            },
        )
        self.estimated_cost = estimated_cost
        self.cost_cap = cost_cap
        self.operation = operation


class ConfigurationError(RadixError):
    """Raised when configuration is invalid or inconsistent."""

    def __init__(self, field: str, value: Any, expected: str, **details):
        message = f"Invalid configuration for {field}: got {value}, expected {expected}"

        super().__init__(message, {"field": field, "value": value, "expected": expected, **details})
        self.field = field
        self.value = value
        self.expected = expected


class ExecutionError(RadixError):
    """Raised when job execution fails."""

    def __init__(self, job_id: str, reason: str, **details):
        message = f"Job execution failed for {job_id}: {reason}"

        super().__init__(message, {"job_id": job_id, "reason": reason, **details})
        self.job_id = job_id
        self.reason = reason


class SchedulingError(RadixError):
    """Raised when job scheduling fails."""

    def __init__(self, reason: str, job_ids: Optional[list] = None, **details):
        if job_ids:
            message = f"Scheduling failed for jobs {job_ids}: {reason}"
        else:
            message = f"Scheduling failed: {reason}"

        super().__init__(message, {"reason": reason, "job_ids": job_ids, **details})
        self.reason = reason
        self.job_ids = job_ids or []


class ResourceError(RadixError):
    """Raised when resource allocation or management fails."""

    def __init__(self, resource_type: str, reason: str, **details):
        message = f"Resource error ({resource_type}): {reason}"

        super().__init__(message, {"resource_type": resource_type, "reason": reason, **details})
        self.resource_type = resource_type
        self.reason = reason


class DependencyError(RadixError):
    """Raised when job dependencies cannot be resolved."""

    def __init__(self, job_id: str, missing_dependencies: list, **details):
        message = f"Unresolved dependencies for job {job_id}: {missing_dependencies}"

        super().__init__(
            message, {"job_id": job_id, "missing_dependencies": missing_dependencies, **details}
        )
        self.job_id = job_id
        self.missing_dependencies = missing_dependencies


class TimeoutError(RadixError):
    """Raised when an operation times out."""

    def __init__(self, operation: str, timeout_seconds: float, **details):
        message = f"Operation timed out: {operation} exceeded {timeout_seconds}s limit"

        super().__init__(
            message, {"operation": operation, "timeout_seconds": timeout_seconds, **details}
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class SwarmError(RadixError):
    """Raised when swarm operations fail."""

    def __init__(self, node_id: Optional[str], reason: str, **details):
        if node_id:
            message = f"Swarm error on node {node_id}: {reason}"
        else:
            message = f"Swarm error: {reason}"

        super().__init__(message, {"node_id": node_id, "reason": reason, **details})
        self.node_id = node_id
        self.reason = reason


class BatchingError(RadixError):
    """Raised when batching operations fail."""

    def __init__(self, batch_id: Optional[str], reason: str, **details):
        if batch_id:
            message = f"Batching error for batch {batch_id}: {reason}"
        else:
            message = f"Batching error: {reason}"

        super().__init__(message, {"batch_id": batch_id, "reason": reason, **details})
        self.batch_id = batch_id
        self.reason = reason


class ValidationError(RadixError):
    """Raised when data validation fails."""

    def __init__(self, field: str, value: Any, constraint: str, **details):
        message = f"Validation failed for {field}: {value} violates constraint '{constraint}'"

        super().__init__(
            message, {"field": field, "value": value, "constraint": constraint, **details}
        )
        self.field = field
        self.value = value
        self.constraint = constraint


class NetworkError(RadixError):
    """Raised when network operations fail (should not occur in local-only mode)."""

    def __init__(self, operation: str, reason: str, **details):
        message = f"Network error during {operation}: {reason}"

        # Add safety warning for network operations
        if "external" in operation.lower() or "remote" in operation.lower():
            message += " (WARNING: External network operations are not allowed in safe mode)"

        super().__init__(message, {"operation": operation, "reason": reason, **details})
        self.operation = operation
        self.reason = reason


class GPUError(RadixError):
    """Raised when GPU operations fail."""

    def __init__(self, gpu_id: Optional[int], reason: str, **details):
        if gpu_id is not None:
            message = f"GPU error on device {gpu_id}: {reason}"
        else:
            message = f"GPU error: {reason}"

        super().__init__(message, {"gpu_id": gpu_id, "reason": reason, **details})
        self.gpu_id = gpu_id
        self.reason = reason


class StorageError(RadixError):
    """Raised when storage operations fail."""

    def __init__(self, path: str, operation: str, reason: str, **details):
        message = f"Storage error during {operation} on {path}: {reason}"

        super().__init__(
            message, {"path": path, "operation": operation, "reason": reason, **details}
        )
        self.path = path
        self.operation = operation
        self.reason = reason


class MetricsError(RadixError):
    """Raised when metrics collection or processing fails."""

    def __init__(self, metric_name: str, reason: str, **details):
        message = f"Metrics error for {metric_name}: {reason}"

        super().__init__(message, {"metric_name": metric_name, "reason": reason, **details})
        self.metric_name = metric_name
        self.reason = reason


# Convenience functions for common error patterns


def raise_safety_violation(violation: str, suggestion: str = None, **details):
    """Raise a safety violation error with helpful context."""
    if suggestion is None:
        if "cost" in violation.lower():
            suggestion = "Check COST_CAP_USD setting and ensure dry-run mode is enabled"
        elif "deploy" in violation.lower():
            suggestion = "Ensure NO_DEPLOY_MODE=true in configuration"
        elif "network" in violation.lower() or "external" in violation.lower():
            suggestion = "All operations must be local-only for safety"
        else:
            suggestion = "Review safety configuration and documentation"

    raise SafetyViolationError(violation, suggestion, **details)


def raise_cost_exceeded(estimated_cost: float, cost_cap: float, operation: str = "operation"):
    """Raise a cost cap exceeded error with standard formatting."""
    raise CostCapExceededError(estimated_cost, cost_cap, operation)


def raise_configuration_error(field: str, value: Any, expected: str, **details):
    """Raise a configuration error with helpful context."""
    # Add common configuration suggestions
    suggestions = {
        "dry_run": "Set DRY_RUN=true in environment variables",
        "cost_cap_usd": "Set COST_CAP_USD=0.00 for safety",
        "no_deploy_mode": "Set NO_DEPLOY_MODE=true to prevent deployments",
        "max_parallelism": "Set to number of CPU cores or less",
        "enable_gpu": "Set to false if no GPU available",
    }

    suggestion = suggestions.get(field.lower())
    if suggestion:
        details["suggestion"] = suggestion

    raise ConfigurationError(field, value, expected, **details)


def raise_execution_error(job_id: str, reason: str, **details):
    """Raise an execution error with helpful context."""
    # Add common execution error suggestions
    if "timeout" in reason.lower():
        details["suggestion"] = "Increase max_runtime_seconds or optimize job"
    elif "memory" in reason.lower():
        details["suggestion"] = "Increase memory_mb requirement or optimize job"
    elif "gpu" in reason.lower():
        details["suggestion"] = "Check GPU availability and requirements"
    elif "permission" in reason.lower():
        details["suggestion"] = "Check file permissions and execution rights"

    raise ExecutionError(job_id, reason, **details)
