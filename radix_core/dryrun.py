"""
Dry Run Safety Guards for Radix

This module provides comprehensive safety guards to prevent actual resource
usage, costs, or deployments. All potentially dangerous operations are
protected by dry-run guards that simulate behavior without side effects.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, TypeVar
from datetime import datetime

from .config import get_config
from .errors import SafetyViolationError, CostCapExceededError

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])

# Global dry-run state tracking
_dry_run_operations: Dict[str, Dict[str, Any]] = {}
_operation_counter = 0

logger = logging.getLogger(__name__)


class DryRunGuard:
    """
    Decorator and context manager for protecting potentially dangerous operations.

    This guard ensures that operations are only simulated when in dry-run mode,
    preventing actual resource usage, costs, or external API calls.
    """

    @staticmethod
    def protect(func: F) -> F:
        """
        Decorator to protect a function with dry-run guards.

        Args:
            func: Function to protect

        Returns:
            Protected function that simulates behavior in dry-run mode
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()

            # Always enforce dry-run mode for safety
            if not config.safety.dry_run:
                raise SafetyViolationError(
                    "Dry-run mode is disabled",
                    "Set DRY_RUN=true in environment variables"
                )

            # Log the protected operation
            operation_id = _log_dry_run_operation(func.__name__, args, kwargs)

            try:
                # Simulate the operation instead of executing it
                result = _simulate_operation(func, args, kwargs)

                _log_operation_completion(operation_id, success=True, result=result)
                return result

            except Exception as e:
                _log_operation_completion(operation_id, success=False, error=str(e))
                raise

        # Mark function as protected
        wrapper._is_dry_run_protected = True
        return wrapper

    @staticmethod
    def is_protected(func: Callable) -> bool:
        """Check if a function is protected by dry-run guards."""
        return getattr(func, '_is_dry_run_protected', False)

    @staticmethod
    def verify_safety():
        """Verify that all safety settings are properly configured."""
        config = get_config()

        violations = []

        # Check dry-run mode
        if not config.safety.dry_run:
            violations.append("DRY_RUN must be True")

        # Check cost caps
        if config.safety.cost_cap_usd != 0.0:
            violations.append(f"COST_CAP_USD must be 0.00, got {config.safety.cost_cap_usd}")

        if config.safety.max_job_cost_usd != 0.0:
            violations.append(f"MAX_JOB_COST_USD must be 0.00, got {config.safety.max_job_cost_usd}")

        # Check deployment mode
        if not config.safety.no_deploy_mode:
            violations.append("NO_DEPLOY_MODE must be True")

        # Check for dangerous configurations
        if config.execution.ray_num_gpus > 0 and not config.execution.ray_local_mode:
            violations.append("Ray GPU usage requires local mode")

        if violations:
            raise SafetyViolationError(
                f"Safety violations detected: {'; '.join(violations)}",
                "Review and fix configuration settings"
            )

    @staticmethod
    def get_operation_log() -> Dict[str, Dict[str, Any]]:
        """Get log of all dry-run operations."""
        return _dry_run_operations.copy()

    @staticmethod
    def clear_operation_log():
        """Clear the dry-run operation log."""
        global _dry_run_operations, _operation_counter
        _dry_run_operations.clear()
        _operation_counter = 0


class CostGuard:
    """Guard against operations that would exceed cost caps."""

    @staticmethod
    def check_cost(estimated_cost: float, operation: str = "operation"):
        """
        Check if an operation would exceed cost caps.

        Args:
            estimated_cost: Estimated cost in USD
            operation: Description of the operation

        Raises:
            CostCapExceededError: If cost exceeds caps
        """
        config = get_config()

        # In dry-run mode, all costs should be $0.00
        if config.safety.dry_run and estimated_cost != 0.0:
            raise CostCapExceededError(
                estimated_cost, 0.0, operation,
                message="All costs must be $0.00 in dry-run mode"
            )

        # Check against cost caps
        if estimated_cost > config.safety.cost_cap_usd:
            raise CostCapExceededError(
                estimated_cost, config.safety.cost_cap_usd, operation
            )

        if estimated_cost > config.safety.max_job_cost_usd:
            raise CostCapExceededError(
                estimated_cost, config.safety.max_job_cost_usd, operation
            )

    @staticmethod
    def protect_with_cost_check(estimated_cost: float, operation: str = "operation"):
        """
        Decorator factory to protect operations with cost checking.

        Args:
            estimated_cost: Estimated cost in USD
            operation: Description of the operation
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                CostGuard.check_cost(estimated_cost, operation)
                return DryRunGuard.protect(func)(*args, **kwargs)
            return wrapper
        return decorator


class DeploymentGuard:
    """Guard against deployment operations."""

    FORBIDDEN_OPERATIONS = {
        'deploy', 'provision', 'create_instance', 'launch_cluster',
        'kubectl', 'helm', 'terraform', 'pulumi', 'aws', 'gcloud',
        'azure', 'docker_run', 'docker_build', 'push_image'
    }

    @staticmethod
    def check_operation(operation_name: str):
        """
        Check if an operation is a forbidden deployment operation.

        Args:
            operation_name: Name of the operation to check

        Raises:
            SafetyViolationError: If operation is forbidden
        """
        config = get_config()

        if not config.safety.no_deploy_mode:
            return  # Deployment mode is enabled (should never happen)

        operation_lower = operation_name.lower()

        for forbidden in DeploymentGuard.FORBIDDEN_OPERATIONS:
            if forbidden in operation_lower:
                raise SafetyViolationError(
                    f"Deployment operation '{operation_name}' is forbidden",
                    "All deployment operations are blocked in safe mode"
                )

    @staticmethod
    def protect_against_deployment(func: F) -> F:
        """Decorator to protect against deployment operations."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            DeploymentGuard.check_operation(func.__name__)
            return DryRunGuard.protect(func)(*args, **kwargs)
        return wrapper


class NetworkGuard:
    """Guard against external network operations."""

    ALLOWED_LOCALHOST = {'localhost', '127.0.0.1', '::1', '0.0.0.0'}

    @staticmethod
    def check_network_operation(host: str, operation: str = "network operation"):
        """
        Check if a network operation is safe (local only).

        Args:
            host: Target host for the operation
            operation: Description of the operation

        Raises:
            SafetyViolationError: If operation targets external hosts
        """
        if host not in NetworkGuard.ALLOWED_LOCALHOST:
            raise SafetyViolationError(
                f"External network operation to {host} is forbidden",
                "Only localhost operations are allowed in safe mode"
            )

    @staticmethod
    def protect_network_operation(host: str, operation: str = "network operation"):
        """Decorator factory to protect network operations."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                NetworkGuard.check_network_operation(host, operation)
                return DryRunGuard.protect(func)(*args, **kwargs)
            return wrapper
        return decorator


def _log_dry_run_operation(func_name: str, args: tuple, kwargs: dict) -> str:
    """Log a dry-run operation for audit purposes."""
    global _operation_counter
    _operation_counter += 1

    operation_id = f"op_{_operation_counter:06d}"

    # Sanitize arguments for logging (remove sensitive data)
    safe_args = _sanitize_for_logging(args)
    safe_kwargs = _sanitize_for_logging(kwargs)

    operation_info = {
        'operation_id': operation_id,
        'function_name': func_name,
        'timestamp': datetime.utcnow().isoformat(),
        'args': safe_args,
        'kwargs': safe_kwargs,
        'status': 'started',
        'simulated': True
    }

    _dry_run_operations[operation_id] = operation_info

    logger.info(f"DRY RUN: Starting {func_name} (ID: {operation_id})")

    return operation_id


def _log_operation_completion(operation_id: str, success: bool, result: Any = None, error: str = None):
    """Log completion of a dry-run operation."""
    if operation_id in _dry_run_operations:
        operation_info = _dry_run_operations[operation_id]
        operation_info.update({
            'status': 'completed' if success else 'failed',
            'end_timestamp': datetime.utcnow().isoformat(),
            'success': success
        })

        if success and result is not None:
            operation_info['result'] = _sanitize_for_logging(result)

        if not success and error:
            operation_info['error'] = error

        status = "SUCCESS" if success else "FAILED"
        func_name = operation_info.get('function_name', 'unknown')
        logger.info(f"DRY RUN: {status} {func_name} (ID: {operation_id})")


def _simulate_operation(func: Callable, args: tuple, kwargs: dict) -> Any:
    """
    Simulate an operation instead of executing it.

    This function provides realistic simulation behavior for different
    types of operations while ensuring no actual side effects occur.
    """
    func_name = func.__name__.lower()

    # Simulate timing for realistic behavior
    simulation_delay = _calculate_simulation_delay(func_name, args, kwargs)
    if simulation_delay > 0:
        time.sleep(simulation_delay)

    # Generate appropriate simulation results based on function type
    if 'execute' in func_name or 'run' in func_name:
        return _simulate_execution_result()
    elif 'create' in func_name or 'provision' in func_name:
        return _simulate_creation_result()
    elif 'deploy' in func_name:
        return _simulate_deployment_result()
    elif 'cost' in func_name or 'estimate' in func_name:
        return 0.0  # Always $0.00 in dry-run mode
    elif 'batch' in func_name or 'process' in func_name:
        return _simulate_batch_result(args, kwargs)
    else:
        return _simulate_generic_result()


def _calculate_simulation_delay(func_name: str, args: tuple, kwargs: dict) -> float:
    """Calculate realistic simulation delay based on operation type."""
    base_delay = 0.1  # Base 100ms delay

    # Scale delay based on operation complexity
    if 'batch' in func_name:
        # Simulate batch processing time
        batch_size = len(args[0]) if args and hasattr(args[0], '__len__') else 10
        return base_delay * min(batch_size / 10, 2.0)  # Max 2 seconds
    elif 'deploy' in func_name or 'provision' in func_name:
        return base_delay * 5  # Deployment operations are slower
    elif 'execute' in func_name:
        return base_delay * 2  # Execution operations are moderate
    else:
        return base_delay


def _simulate_execution_result() -> Dict[str, Any]:
    """Simulate job execution result."""
    return {
        'status': 'completed',
        'return_code': 0,
        'stdout': 'Simulated execution output',
        'stderr': '',
        'duration_seconds': 1.5,
        'simulated': True
    }


def _simulate_creation_result() -> Dict[str, Any]:
    """Simulate resource creation result."""
    return {
        'id': f'sim_{int(time.time())}',
        'status': 'created',
        'type': 'simulated_resource',
        'cost_usd': 0.0,
        'simulated': True
    }


def _simulate_deployment_result() -> Dict[str, Any]:
    """Simulate deployment result."""
    return {
        'deployment_id': f'sim_deploy_{int(time.time())}',
        'status': 'deployed',
        'url': 'http://localhost:8080',
        'cost_usd': 0.0,
        'simulated': True
    }


def _simulate_batch_result(args: tuple, kwargs: dict) -> Any:
    """Simulate batch processing result."""
    # Try to determine batch size from arguments
    batch_size = 1
    if args:
        first_arg = args[0]
        if hasattr(first_arg, '__len__'):
            batch_size = len(first_arg)

    # Return list of simulated results
    return [f'simulated_result_{i}' for i in range(batch_size)]


def _simulate_generic_result() -> Dict[str, Any]:
    """Simulate generic operation result."""
    return {
        'status': 'success',
        'timestamp': datetime.utcnow().isoformat(),
        'simulated': True
    }


def _sanitize_for_logging(obj: Any) -> Any:
    """Sanitize objects for safe logging (remove sensitive information)."""
    if isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in ['password', 'token', 'key', 'secret']):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = _sanitize_for_logging(value)
        return sanitized
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_logging(item) for item in obj]
    elif isinstance(obj, str) and len(obj) > 1000:
        return obj[:1000] + '...[TRUNCATED]'
    else:
        return obj


# Convenience decorators combining multiple guards
def safe_operation(func: F) -> F:
    """Decorator combining all safety guards."""
    return DryRunGuard.protect(DeploymentGuard.protect_against_deployment(func))


def safe_execution(estimated_cost: float = 0.0):
    """Decorator factory for safe job execution."""
    def decorator(func: F) -> F:
        return CostGuard.protect_with_cost_check(estimated_cost, f"execution of {func.__name__}")(func)
    return decorator


def safe_network_operation(host: str):
    """Decorator factory for safe network operations."""
    def decorator(func: F) -> F:
        return NetworkGuard.protect_network_operation(host, f"network operation {func.__name__}")(func)
    return decorator
