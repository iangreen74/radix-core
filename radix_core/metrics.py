"""
Enhanced Performance Metrics Collection System for Radix

Provides comprehensive metrics collection, aggregation, and analysis
for GPU orchestration research with real-time monitoring capabilities.
"""

import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """A single metric measurement point."""

    timestamp: datetime
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {"timestamp": self.timestamp.isoformat(), "value": self.value, "tags": self.tags}


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    name: str
    count: int
    min_value: float
    max_value: float
    mean: float
    median: float
    std_dev: float
    p95: float
    p99: float
    first_timestamp: datetime
    last_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "count": self.count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "p95": self.p95,
            "p99": self.p99,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
            "duration_seconds": (self.last_timestamp - self.first_timestamp).total_seconds(),
        }


class MetricsCollector:
    """
    Collects and manages metrics for performance monitoring and research analysis.

    This collector is designed for local research use with no external dependencies
    or network calls. All metrics are stored in memory and can be exported for analysis.
    """

    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)

        # Thread safety
        self._lock = threading.RLock()

        # Auto-collection thread
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_collection = threading.Event()
        self._collection_interval = 1.0  # seconds

        # Callbacks for custom metrics
        self._metric_callbacks: Dict[str, Callable[[], Union[int, float]]] = {}

    def start_collection(self, interval: float = 1.0):
        """Start automatic metrics collection."""
        self._collection_interval = interval

        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_collection.clear()
            self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self._collection_thread.start()
            logger.info("Metrics collection started", interval=interval)

    def stop_collection(self):
        """Stop automatic metrics collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            self._stop_collection.set()
            self._collection_thread.join(timeout=2.0)
            logger.info("Metrics collection stopped")

    def record_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric (monotonically increasing)."""
        with self._lock:
            self.counters[name] += value
            self._add_metric_point(name, self.counters[name], tags or {})

    def record_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Record a gauge metric (point-in-time value)."""
        with self._lock:
            self.gauges[name] = float(value)
            self._add_metric_point(name, value, tags or {})

    def record_histogram(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Record a histogram metric (distribution of values)."""
        with self._lock:
            self.histograms[name].append(float(value))
            # Keep histogram size manageable
            if len(self.histograms[name]) > self.max_points_per_metric:
                self.histograms[name] = self.histograms[name][-self.max_points_per_metric :]

            self._add_metric_point(name, value, tags or {})

    def record_timer(self, name: str, duration_seconds: float, tags: Dict[str, str] = None):
        """Record a timer metric (duration measurement)."""
        with self._lock:
            self.timers[name].append(duration_seconds)
            # Keep timer history manageable
            if len(self.timers[name]) > self.max_points_per_metric:
                self.timers[name] = self.timers[name][-self.max_points_per_metric :]

            self._add_metric_point(f"{name}_duration", duration_seconds, tags or {})

    def time_operation(self, name: str, tags: Dict[str, str] = None):
        """Context manager to time operations."""
        return TimerContext(self, name, tags or {})

    def register_callback(self, name: str, callback: Callable[[], Union[int, float]]):
        """Register a callback function to collect custom metrics."""
        self._metric_callbacks[name] = callback

    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return None

            points = list(self.metrics[name])
            values = [p.value for p in points]

            if not values:
                return None

            # Calculate statistics
            count = len(values)
            min_val = min(values)
            max_val = max(values)
            mean = statistics.mean(values)
            median = statistics.median(values)
            std_dev = statistics.stdev(values) if count > 1 else 0.0

            # Calculate percentiles
            sorted_values = sorted(values)
            p95_idx = int(0.95 * count)
            p99_idx = int(0.99 * count)
            p95 = sorted_values[min(p95_idx, count - 1)]
            p99 = sorted_values[min(p99_idx, count - 1)]

            return MetricSummary(
                name=name,
                count=count,
                min_value=min_val,
                max_value=max_val,
                mean=mean,
                median=median,
                std_dev=std_dev,
                p95=p95,
                p99=p99,
                first_timestamp=points[0].timestamp,
                last_timestamp=points[-1].timestamp,
            )

    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Get summary statistics for all metrics."""
        summaries = {}

        with self._lock:
            for name in self.metrics.keys():
                summary = self.get_metric_summary(name)
                if summary:
                    summaries[name] = summary

        return summaries

    def get_recent_values(self, name: str, duration: timedelta) -> List[MetricPoint]:
        """Get metric values from the last specified duration."""
        cutoff_time = datetime.utcnow() - duration

        with self._lock:
            if name not in self.metrics:
                return []

            return [p for p in self.metrics[name] if p.timestamp >= cutoff_time]

    def export_metrics(self, filepath: str, format: str = "json"):
        """Export all metrics to a file."""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "format_version": "1.0",
            "metrics": {},
            "summaries": {},
        }

        with self._lock:
            # Export raw metrics
            for name, points in self.metrics.items():
                export_data["metrics"][name] = [p.to_dict() for p in points]

            # Export summaries
            for name, summary in self.get_all_summaries().items():
                export_data["summaries"][name] = summary.to_dict()

        # Write to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info("Metrics exported", filepath=str(filepath), format=format)

    def clear_metrics(self):
        """Clear all collected metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()

        logger.info("All metrics cleared")

    def _add_metric_point(self, name: str, value: Union[int, float], tags: Dict[str, str]):
        """Add a metric point to the collection."""
        point = MetricPoint(timestamp=datetime.utcnow(), value=float(value), tags=tags)
        self.metrics[name].append(point)

    def _collection_loop(self):
        """Background thread loop for automatic metrics collection."""
        while not self._stop_collection.wait(self._collection_interval):
            try:
                self._collect_system_metrics()
                self._collect_callback_metrics()
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_gauge("system.cpu.percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system.memory.percent", memory.percent)
            self.record_gauge("system.memory.used_mb", memory.used / 1024 / 1024)
            self.record_gauge("system.memory.available_mb", memory.available / 1024 / 1024)

            # Disk metrics
            disk = psutil.disk_usage("/")
            self.record_gauge("system.disk.percent", disk.percent)
            self.record_gauge("system.disk.used_gb", disk.used / 1024 / 1024 / 1024)
            self.record_gauge("system.disk.free_gb", disk.free / 1024 / 1024 / 1024)

        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))

    def _collect_callback_metrics(self):
        """Collect metrics from registered callbacks."""
        for name, callback in self._metric_callbacks.items():
            try:
                value = callback()
                self.record_gauge(f"callback.{name}", value)
            except Exception as e:
                logger.error("Error in metric callback", callback=name, error=str(e))


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time

            # Add success/failure tag
            if exc_type is None:
                self.tags["status"] = "success"
            else:
                self.tags["status"] = "error"
                self.tags["error_type"] = exc_type.__name__

            self.collector.record_timer(self.name, duration, self.tags)


class JobMetricsCollector:
    """Specialized metrics collector for job execution."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def record_job_submitted(self, job_id: str, job_type: str = "unknown"):
        """Record job submission."""
        self.collector.record_counter("jobs.submitted", 1, {"job_id": job_id, "job_type": job_type})

    def record_job_started(self, job_id: str, executor_type: str = "unknown"):
        """Record job start."""
        self.collector.record_counter(
            "jobs.started", 1, {"job_id": job_id, "executor_type": executor_type}
        )

    def record_job_completed(self, job_id: str, duration_seconds: float, success: bool = True):
        """Record job completion."""
        status = "success" if success else "failure"

        self.collector.record_counter(f"jobs.{status}", 1, {"job_id": job_id})
        self.collector.record_timer(
            "jobs.duration", duration_seconds, {"job_id": job_id, "status": status}
        )

    def record_batch_processed(self, batch_size: int, duration_seconds: float):
        """Record batch processing metrics."""
        throughput = batch_size / duration_seconds if duration_seconds > 0 else 0

        self.collector.record_histogram("batch.size", batch_size)
        self.collector.record_timer("batch.duration", duration_seconds)
        self.collector.record_gauge("batch.throughput", throughput)

    def record_resource_usage(self, cpu_percent: float, memory_mb: float, gpu_percent: float = 0.0):
        """Record resource usage during job execution."""
        self.collector.record_gauge("job.cpu.percent", cpu_percent)
        self.collector.record_gauge("job.memory.mb", memory_mb)
        if gpu_percent > 0:
            self.collector.record_gauge("job.gpu.percent", gpu_percent)


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        config = get_config()
        _global_collector = MetricsCollector()

        if config.logging.enable_metrics:
            _global_collector.start_collection(config.logging.metrics_interval)

    return _global_collector


def get_job_metrics_collector() -> JobMetricsCollector:
    """Get a job-specific metrics collector."""
    return JobMetricsCollector(get_metrics_collector())


def reset_metrics_collector():
    """Reset the global metrics collector."""
    global _global_collector
    if _global_collector:
        _global_collector.stop_collection()
    _global_collector = None


# Convenience functions
def record_counter(name: str, value: int = 1, tags: Dict[str, str] = None):
    """Record a counter metric."""
    get_metrics_collector().record_counter(name, value, tags)


def record_gauge(name: str, value: Union[int, float], tags: Dict[str, str] = None):
    """Record a gauge metric."""
    get_metrics_collector().record_gauge(name, value, tags)


def record_histogram(name: str, value: Union[int, float], tags: Dict[str, str] = None):
    """Record a histogram metric."""
    get_metrics_collector().record_histogram(name, value, tags)


def record_timer(name: str, duration_seconds: float, tags: Dict[str, str] = None):
    """Record a timer metric."""
    get_metrics_collector().record_timer(name, duration_seconds, tags)


def time_operation(name: str, tags: Dict[str, str] = None):
    """Context manager to time operations."""
    return get_metrics_collector().time_operation(name, tags)
