"""
Enhanced Structured Logging for Radix

This module provides comprehensive logging capabilities with correlation IDs,
distributed tracing, safety-aware configuration, and detailed audit trails
for research purposes.
"""

import json
import logging
import logging.handlers
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from .config import get_config

# Thread-local storage for correlation context
_correlation_context = threading.local()


class CorrelationContext:
    """Manages correlation IDs and tracing context across operations."""

    @staticmethod
    def get_correlation_id() -> str:
        """Get the current correlation ID, creating one if needed."""
        if not hasattr(_correlation_context, "correlation_id"):
            _correlation_context.correlation_id = str(uuid.uuid4())[:8]
        return _correlation_context.correlation_id

    @staticmethod
    def set_correlation_id(correlation_id: str):
        """Set the correlation ID for the current thread."""
        _correlation_context.correlation_id = correlation_id

    @staticmethod
    def clear_correlation_id():
        """Clear the correlation ID for the current thread."""
        if hasattr(_correlation_context, "correlation_id"):
            delattr(_correlation_context, "correlation_id")

    @staticmethod
    def get_trace_context() -> Dict[str, Any]:
        """Get full tracing context."""
        correlation_id = CorrelationContext.get_correlation_id()

        # Get operation stack if available
        operation_stack = getattr(_correlation_context, "operation_stack", [])

        return {
            "correlation_id": correlation_id,
            "thread_id": threading.get_ident(),
            "operation_stack": operation_stack,
            "depth": len(operation_stack),
        }

    @staticmethod
    def push_operation(operation_name: str):
        """Push an operation onto the trace stack."""
        if not hasattr(_correlation_context, "operation_stack"):
            _correlation_context.operation_stack = []
        _correlation_context.operation_stack.append(operation_name)

    @staticmethod
    def pop_operation():
        """Pop an operation from the trace stack."""
        if (
            hasattr(_correlation_context, "operation_stack")
            and _correlation_context.operation_stack
        ):
            return _correlation_context.operation_stack.pop()
        return None


def with_correlation_id(correlation_id: str = None):
    """Decorator to run function with specific correlation ID."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_correlation_id = getattr(_correlation_context, "correlation_id", None)

            try:
                if correlation_id:
                    CorrelationContext.set_correlation_id(correlation_id)
                else:
                    # Generate new correlation ID if none provided
                    CorrelationContext.set_correlation_id(str(uuid.uuid4())[:8])

                return func(*args, **kwargs)
            finally:
                # Restore previous correlation ID
                if old_correlation_id:
                    CorrelationContext.set_correlation_id(old_correlation_id)
                else:
                    CorrelationContext.clear_correlation_id()

        return wrapper

    return decorator


def trace_operation(operation_name: str):
    """Decorator to trace function execution with operation stack."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"radix.trace.{func.__module__}")
            trace_context = CorrelationContext.get_trace_context()

            # Push operation onto stack
            CorrelationContext.push_operation(operation_name)

            start_time = time.time()

            try:
                logger.info(
                    f"Starting operation: {operation_name}",
                    operation=operation_name,
                    function=func.__name__,
                    **trace_context,
                )

                result = func(*args, **kwargs)

                end_time = time.time()
                duration = end_time - start_time

                logger.info(
                    f"Completed operation: {operation_name}",
                    operation=operation_name,
                    function=func.__name__,
                    duration_seconds=duration,
                    success=True,
                    **trace_context,
                )

                return result

            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time

                logger.error(
                    f"Failed operation: {operation_name}",
                    operation=operation_name,
                    function=func.__name__,
                    duration_seconds=duration,
                    success=False,
                    error=str(e),
                    **trace_context,
                )
                raise
            finally:
                # Pop operation from stack
                CorrelationContext.pop_operation()

        return wrapper

    return decorator


class SafetyAwareFormatter(logging.Formatter):
    """Custom formatter that adds safety context and correlation IDs to log messages."""

    def format(self, record: logging.LogRecord) -> str:
        # Add safety context to all log records
        config = get_config()
        record.dry_run = config.safety.dry_run
        record.cost_cap = config.safety.cost_cap_usd
        record.no_deploy = config.safety.no_deploy_mode

        # Add correlation context
        trace_context = CorrelationContext.get_trace_context()
        record.correlation_id = trace_context["correlation_id"]
        record.thread_id = trace_context["thread_id"]
        record.operation_depth = trace_context["depth"]

        # Add current operation if available
        operation_stack = trace_context.get("operation_stack", [])
        record.current_operation = operation_stack[-1] if operation_stack else None

        # Add timestamp if not present
        if not hasattr(record, "timestamp"):
            record.timestamp = datetime.utcnow().isoformat()

        return super().format(record)


class JSONFormatter(SafetyAwareFormatter):
    """JSON formatter for structured logging with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation": {
                "correlation_id": getattr(record, "correlation_id", "unknown"),
                "thread_id": getattr(record, "thread_id", 0),
                "operation_depth": getattr(record, "operation_depth", 0),
                "current_operation": getattr(record, "current_operation", None),
            },
            "safety": {
                "dry_run": getattr(record, "dry_run", True),
                "cost_cap_usd": getattr(record, "cost_cap", 0.0),
                "no_deploy_mode": getattr(record, "no_deploy", True),
            },
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields (excluding internal fields)
        excluded_fields = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "dry_run",
            "cost_cap",
            "no_deploy",
            "correlation_id",
            "thread_id",
            "operation_depth",
            "current_operation",
            "timestamp",
        }

        for key, value in record.__dict__.items():
            if key not in excluded_fields:
                log_entry[key] = value

        return json.dumps(log_entry)


class ConsoleFormatter(SafetyAwareFormatter):
    """Console formatter with color support, correlation IDs, and safety indicators."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "GRAY": "\033[90m",  # Gray for correlation IDs
        "BOLD": "\033[1m",  # Bold
    }

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, "")
        reset_color = self.COLORS["RESET"]
        gray_color = self.COLORS["GRAY"]

        # Safety indicator
        safety_indicator = "🔒" if getattr(record, "dry_run", True) else "⚠️"

        # Format timestamp
        timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds

        # Correlation ID (short form for console)
        correlation_id = getattr(record, "correlation_id", "unknown")

        # Operation depth indicator
        depth = getattr(record, "operation_depth", 0)
        indent = "  " * depth

        # Current operation
        current_op = getattr(record, "current_operation", "")
        operation_info = f"[{current_op}]" if current_op else ""

        # Format message
        message = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        # Build formatted log line
        formatted_line = (
            f"{safety_indicator} {gray_color}{timestamp}{reset_color} "
            f"{gray_color}[{correlation_id}]{reset_color} "
            f"{level_color}{record.levelname:8}{reset_color} "
            f"{record.name:20} "
            f"{indent}{gray_color}{operation_info}{reset_color} "
            f"{message}"
        )

        return formatted_line


class AuditLogger:
    """Special logger for audit trail and safety events."""

    def __init__(self, name: str = "radix.audit"):
        self.logger = structlog.get_logger(name)
        self.audit_file: Optional[Path] = None
        self._setup_audit_file()

    def _setup_audit_file(self):
        """Setup audit log file."""
        config = get_config()
        if config.research.log_experiments:
            audit_dir = Path(config.research.results_dir) / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.audit_file = audit_dir / f"audit_{timestamp}.log"

    def log_safety_event(self, event_type: str, details: Dict[str, Any]):
        """Log safety-related events."""
        event = {
            "event_type": "safety",
            "safety_event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
        }

        self.logger.info("Safety event", **event)
        self._write_to_audit_file(event)

    def log_cost_event(self, operation: str, estimated_cost: float, actual_cost: float = 0.0):
        """Log cost-related events."""
        event = {
            "event_type": "cost",
            "operation": operation,
            "estimated_cost_usd": estimated_cost,
            "actual_cost_usd": actual_cost,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.logger.info("Cost event", **event)
        self._write_to_audit_file(event)

    def log_execution_event(self, job_id: str, event_type: str, details: Dict[str, Any]):
        """Log job execution events."""
        event = {
            "event_type": "execution",
            "job_id": job_id,
            "execution_event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
        }

        self.logger.info("Execution event", **event)
        self._write_to_audit_file(event)

    def log_configuration_change(self, field: str, old_value: Any, new_value: Any):
        """Log configuration changes."""
        event = {
            "event_type": "configuration",
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.logger.warning("Configuration change", **event)
        self._write_to_audit_file(event)

    def _write_to_audit_file(self, event: Dict[str, Any]):
        """Write event to audit file."""
        if self.audit_file:
            try:
                with open(self.audit_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                # Don't fail if audit logging fails
                self.logger.error("Failed to write audit log", error=str(e))


class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, name: str = "radix.performance"):
        self.logger = structlog.get_logger(name)

    @contextmanager
    def time_operation(self, operation: str, **context):
        """Context manager to time operations."""
        start_time = time.time()
        start_timestamp = datetime.utcnow()

        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time

            self.logger.info(
                "Operation completed",
                operation=operation,
                duration_seconds=duration,
                start_time=start_timestamp.isoformat(),
                success=success,
                error=error,
                **context,
            )

    def log_throughput(
        self, operation: str, items_processed: int, duration_seconds: float, **context
    ):
        """Log throughput metrics."""
        throughput = items_processed / duration_seconds if duration_seconds > 0 else 0

        self.logger.info(
            "Throughput measurement",
            operation=operation,
            items_processed=items_processed,
            duration_seconds=duration_seconds,
            throughput_items_per_second=throughput,
            **context,
        )

    def log_resource_usage(self, cpu_percent: float, memory_mb: float, **context):
        """Log resource usage metrics."""
        self.logger.info(
            "Resource usage",
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            timestamp=datetime.utcnow().isoformat(),
            **context,
        )


def setup_logging():
    """Setup structured logging for Radix."""
    config = get_config()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if config.logging.log_format == "json"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.logging.log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if config.logging.log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())

    root_logger.addHandler(console_handler)

    # Add file handler for persistent logging
    if config.research.log_experiments:
        log_dir = Path(config.research.results_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"radix_{timestamp}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Setup safety logging
    safety_logger = logging.getLogger("radix.safety")
    safety_logger.info(
        f"Logging initialized - dry_run={config.safety.dry_run}, "
        f"cost_cap=${config.safety.cost_cap_usd}, "
        f"log_level={config.logging.log_level}, "
        f"log_format={config.logging.log_format}"
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def get_audit_logger() -> AuditLogger:
    """Get the audit logger instance."""
    return AuditLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger instance."""
    return PerformanceLogger()


# Convenience functions for common logging patterns
def log_safety_violation(violation: str, details: Dict[str, Any] = None):
    """Log a safety violation."""
    audit_logger = get_audit_logger()
    audit_logger.log_safety_event("violation", {"violation": violation, "details": details or {}})


def log_cost_estimate(operation: str, estimated_cost: float):
    """Log a cost estimate."""
    audit_logger = get_audit_logger()
    audit_logger.log_cost_event(operation, estimated_cost)


def log_job_event(job_id: str, event_type: str, **details):
    """Log a job-related event."""
    audit_logger = get_audit_logger()
    audit_logger.log_execution_event(job_id, event_type, details)


# Initialize logging on module import
setup_logging()
