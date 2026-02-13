"""
Enhanced Performance Metrics Collection System for Radix

Provides comprehensive metrics collection, aggregation, and analysis
for GPU orchestration research with real-time monitoring capabilities.
"""

import time
import threading
import queue
import statistics
import json
import psutil
import gc
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .config import get_config
from .logging import get_logger, CorrelationContext


@dataclass
class MetricPoint:
    """Individual metric measurement point."""
    timestamp: datetime
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class AggregatedMetric:
    """Aggregated metric with statistical information."""
    name: str
    count: int
    sum: float
    mean: float
    min: float
    max: float
    std_dev: float
    p50: float
    p95: float
    p99: float
    labels: Dict[str, str] = field(default_factory=dict)
    time_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    memory_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    process_count: int
    thread_count: int
    file_descriptors: int
    correlation_id: Optional[str] = None


@dataclass
class JobMetrics:
    """Job execution metrics."""
    job_id: str
    job_type: str
    executor_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    cpu_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    exit_code: Optional[int] = None
    success: Optional[bool] = None
    queue_time_seconds: float = 0.0
    execution_time_seconds: float = 0.0
    cleanup_time_seconds: float = 0.0
    correlation_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe metrics collector with real-time aggregation."""

    def __init__(self, collection_interval: float = 1.0, max_points: int = 10000):
        self.collection_interval = collection_interval
        self.max_points = max_points
        self.logger = get_logger("radix.metrics")

        # Thread-safe storage
        self._metrics_queue = queue.Queue()
        self._raw_metrics: deque = deque(maxlen=max_points)
        self._system_metrics: deque = deque(maxlen=1000)  # Keep last 1000 system snapshots
        self._job_metrics: Dict[str, JobMetrics] = {}
        self._aggregated_metrics: Dict[str, AggregatedMetric] = {}

        # Threading control
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        self._aggregation_thread: Optional[threading.Thread] = None
        self._system_monitor_thread: Optional[threading.Thread] = None

        # Lock for thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._operation_timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def start_collection(self):
        """Start metrics collection in background threads."""
        if self._collecting:
            return

        self._collecting = True

        # Start collection threads
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._system_monitor_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)

        self._collection_thread.start()
        self._aggregation_thread.start()
        self._system_monitor_thread.start()

        self.logger.info("Metrics collection started",
                        collection_interval=self.collection_interval,
                        max_points=self.max_points)

    def stop_collection(self):
        """Stop metrics collection."""
        self._collecting = False

        # Wait for threads to finish
        for thread in [self._collection_thread, self._aggregation_thread, self._system_monitor_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)

        self.logger.info("Metrics collection stopped")

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric measurement."""
        metric = MetricPoint(
            timestamp=datetime.utcnow(),
            name=name,
            value=value,
            labels=labels or {},
            correlation_id=CorrelationContext.get_correlation_id()
        )

        try:
            self._metrics_queue.put_nowait(metric)
        except queue.Full:
            self.logger.warning("Metrics queue full, dropping metric", metric_name=name)

    def record_counter(self, name: str, increment: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter increment."""
        self.record_metric(f"{name}.count", increment, labels)

    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge value."""
        self.record_metric(f"{name}.gauge", value, labels)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value."""
        self.record_metric(f"{name}.histogram", value, labels)

    @contextmanager
    def time_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """Context manager to time operations."""
        start_time = time.time()
        correlation_id = CorrelationContext.get_correlation_id()

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

            # Record timing metric
            timing_labels = (labels or {}).copy()
            timing_labels.update({
                'operation': operation_name,
                'success': str(success),
                'correlation_id': correlation_id
            })

            self.record_histogram("operation.duration_seconds", duration, timing_labels)

            # Track operation timings for analysis
            with self._lock:
                self._operation_timers[operation_name].append(duration)

    def start_job_metrics(self, job_id: str, job_type: str = "unknown", executor_type: str = "unknown") -> JobMetrics:
        """Start tracking metrics for a job."""
        job_metrics = JobMetrics(
            job_id=job_id,
            job_type=job_type,
            executor_type=executor_type,
            start_time=datetime.utcnow(),
            correlation_id=CorrelationContext.get_correlation_id()
        )

        with self._lock:
            self._job_metrics[job_id] = job_metrics

        self.logger.debug("Started job metrics tracking", job_id=job_id, job_type=job_type)
        return job_metrics

    def finish_job_metrics(self, job_id: str, success: bool, exit_code: int = 0, **metrics):
        """Finish tracking metrics for a job."""
        with self._lock:
            if job_id not in self._job_metrics:
                self.logger.warning("Job metrics not found", job_id=job_id)
                return

            job_metrics = self._job_metrics[job_id]
            job_metrics.end_time = datetime.utcnow()
            job_metrics.duration_seconds = (job_metrics.end_time - job_metrics.start_time).total_seconds()
            job_metrics.success = success
            job_metrics.exit_code = exit_code

            # Update with provided metrics
            for key, value in metrics.items():
                if hasattr(job_metrics, key):
                    setattr(job_metrics, key, value)

        # Record job completion metrics
        job_labels = {
            'job_id': job_id,
            'job_type': job_metrics.job_type,
            'executor_type': job_metrics.executor_type,
            'success': str(success)
        }

        self.record_histogram("job.duration_seconds", job_metrics.duration_seconds, job_labels)
        self.record_histogram("job.cpu_time_seconds", job_metrics.cpu_time_seconds, job_labels)
        self.record_histogram("job.peak_memory_mb", job_metrics.peak_memory_mb, job_labels)
        self.record_counter("job.completed", 1.0, job_labels)

    def get_aggregated_metrics(self, time_window: timedelta = None) -> Dict[str, AggregatedMetric]:
        """Get aggregated metrics for a time window."""
        if time_window is None:
            time_window = timedelta(minutes=5)

        cutoff_time = datetime.utcnow() - time_window
        aggregated = defaultdict(list)

        # Collect raw metrics within time window
        with self._lock:
            for metric in self._raw_metrics:
                if metric.timestamp >= cutoff_time:
                    key = f"{metric.name}:{json.dumps(metric.labels, sort_keys=True)}"
                    aggregated[key].append(metric.value)

        # Calculate aggregations
        result = {}
        for key, values in aggregated.items():
            if not values:
                continue

            name, labels_json = key.split(':', 1)
            labels = json.loads(labels_json)

            # Calculate statistics
            values_array = np.array(values)
            result[key] = AggregatedMetric(
                name=name,
                count=len(values),
                sum=float(np.sum(values_array)),
                mean=float(np.mean(values_array)),
                min=float(np.min(values_array)),
                max=float(np.max(values_array)),
                std_dev=float(np.std(values_array)),
                p50=float(np.percentile(values_array, 50)),
                p95=float(np.percentile(values_array, 95)),
                p99=float(np.percentile(values_array, 99)),
                labels=labels,
                time_window=time_window
            )

        return result

    def export_metrics(self, filepath: Path = None, format: str = "json") -> Optional[Path]:
        """Export metrics to file."""
        if filepath is None:
            config = get_config()
            results_dir = Path(config.research.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filepath = results_dir / f"metrics_{timestamp}.{format}"

        export_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'collection_interval': self.collection_interval,
            'aggregated_metrics': {k: asdict(v) for k, v in self.get_aggregated_metrics().items()},
            'job_metrics': {k: asdict(v) for k, v in self._job_metrics.items()},
            'operation_statistics': self.get_operation_statistics()
        }

        try:
            with open(filepath, 'w') as f:
                if format == "json":
                    json.dump(export_data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            self.logger.info("Metrics exported", filepath=str(filepath), format=format)
            return filepath

        except Exception as e:
            self.logger.error("Failed to export metrics", error=str(e), filepath=str(filepath))
            return None

    def get_operation_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get operation timing statistics."""
        with self._lock:
            result = {}
            for op_name, timings in self._operation_timers.items():
                if timings:
                    timings_list = list(timings)
                    result[op_name] = {
                        'count': len(timings_list),
                        'mean': statistics.mean(timings_list),
                        'median': statistics.median(timings_list),
                        'std_dev': statistics.stdev(timings_list) if len(timings_list) > 1 else 0.0,
                        'min': min(timings_list),
                        'max': max(timings_list)
                    }
            return result

    def _collection_loop(self):
        """Main metrics collection loop."""
        while self._collecting:
            try:
                # Process queued metrics
                processed = 0
                while processed < 100:  # Process up to 100 metrics per iteration
                    try:
                        metric = self._metrics_queue.get_nowait()
                        with self._lock:
                            self._raw_metrics.append(metric)
                        processed += 1
                    except queue.Empty:
                        break

                time.sleep(self.collection_interval)

            except Exception as e:
                self.logger.error("Error in metrics collection loop", error=str(e))
                time.sleep(1.0)

    def _aggregation_loop(self):
        """Metrics aggregation loop."""
        while self._collecting:
            try:
                # Update aggregated metrics every 30 seconds
                with self._lock:
                    self._aggregated_metrics = self.get_aggregated_metrics()

                time.sleep(30.0)

            except Exception as e:
                self.logger.error("Error in metrics aggregation loop", error=str(e))
                time.sleep(5.0)

    def _system_monitor_loop(self):
        """System monitoring loop."""
        last_disk_io = None
        last_network_io = None

        while self._collecting:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()

                # Calculate deltas for IO metrics
                disk_read_mb = 0.0
                disk_write_mb = 0.0
                if last_disk_io and disk_io:
                    disk_read_mb = (disk_io.read_bytes - last_disk_io.read_bytes) / 1024 / 1024
                    disk_write_mb = (disk_io.write_bytes - last_disk_io.write_bytes) / 1024 / 1024

                network_sent = 0.0
                network_recv = 0.0
                if last_network_io and network_io:
                    network_sent = network_io.bytes_sent - last_network_io.bytes_sent
                    network_recv = network_io.bytes_recv - last_network_io.bytes_recv

                # Get process information
                current_process = psutil.Process()
                process_count = len(psutil.pids())

                system_metrics = SystemMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=cpu_percent,
                    cpu_count=psutil.cpu_count(),
                    memory_percent=memory.percent,
                    memory_total_gb=memory.total / 1024 / 1024 / 1024,
                    memory_used_gb=memory.used / 1024 / 1024 / 1024,
                    memory_available_gb=memory.available / 1024 / 1024 / 1024,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_bytes_sent=network_sent,
                    network_bytes_recv=network_recv,
                    process_count=process_count,
                    thread_count=current_process.num_threads(),
                    file_descriptors=current_process.num_fds() if hasattr(current_process, 'num_fds') else 0,
                    correlation_id=CorrelationContext.get_correlation_id()
                )

                with self._lock:
                    self._system_metrics.append(system_metrics)

                # Record as regular metrics for aggregation
                self.record_gauge("system.cpu_percent", cpu_percent)
                self.record_gauge("system.memory_percent", memory.percent)
                self.record_gauge("system.memory_used_gb", memory.used / 1024 / 1024 / 1024)

                # Update last measurements
                last_disk_io = disk_io
                last_network_io = network_io

                time.sleep(self.collection_interval)

            except Exception as e:
                self.logger.error("Error in system monitoring loop", error=str(e))
                time.sleep(5.0)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        config = get_config()
        _metrics_collector = MetricsCollector(
            collection_interval=getattr(config.logging, 'metrics_interval', 60) / 60.0,  # Convert to seconds
            max_points=10000  # Default value
        )
        _metrics_collector.start_collection()
    return _metrics_collector


# Convenience functions
def record_metric(name: str, value: float, labels: Dict[str, str] = None):
    """Convenience function to record a metric."""
    get_metrics_collector().record_metric(name, value, labels)


def record_counter(name: str, increment: float = 1.0, labels: Dict[str, str] = None):
    """Convenience function to record a counter."""
    get_metrics_collector().record_counter(name, increment, labels)


def record_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Convenience function to record a gauge."""
    get_metrics_collector().record_gauge(name, value, labels)


def time_operation(operation_name: str, labels: Dict[str, str] = None):
    """Convenience function to time an operation."""
    return get_metrics_collector().time_operation(operation_name, labels)


def start_job_metrics(job_id: str, job_type: str = "unknown", executor_type: str = "unknown") -> JobMetrics:
    """Convenience function to start job metrics tracking."""
    return get_metrics_collector().start_job_metrics(job_id, job_type, executor_type)


def finish_job_metrics(job_id: str, success: bool, exit_code: int = 0, **metrics):
    """Convenience function to finish job metrics tracking."""
    get_metrics_collector().finish_job_metrics(job_id, success, exit_code, **metrics)


def export_metrics(filepath: Path = None, format: str = "json") -> Optional[Path]:
    """Convenience function to export metrics."""
    return get_metrics_collector().export_metrics(filepath, format)
