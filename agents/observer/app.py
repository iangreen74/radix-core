"""Radix Observer — real GPU efficiency tracking via Kubernetes pod lifecycle."""

import os
import json
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Configuration
NS = os.environ.get("POD_NAMESPACE", "default")
RET_DAYS = int(os.environ.get("RETENTION_DAYS", "7"))
TS_DIR = Path(os.environ.get("TS_DIR", "/var/radix/ts"))
try:
    TS_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    TS_DIR = Path("/tmp/radix/ts")
    TS_DIR.mkdir(parents=True, exist_ok=True)
TS_FILE = TS_DIR / "radix_timeseries.jsonl"
SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "http://radix-scheduler-agent:8080")
ROLLING_WINDOW_SECONDS = 3600  # 1 hour
MAX_COMPLETED = 10_000


@dataclass
class JobLifecycle:
    """Tracks a single GPU job from submission to completion."""
    pod_name: str
    job_type: str
    namespace: str = "default"
    gpu_type: str = "unknown"
    submitted_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    status: str = "pending"  # pending, running, succeeded, failed

    @property
    def wait_time(self) -> float:
        """Time from submission to start."""
        if self.started_at and self.submitted_at:
            return self.started_at - self.submitted_at
        return 0.0

    @property
    def run_time(self) -> float:
        """Time from start to completion."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def total_time(self) -> float:
        """Total time from submission to completion."""
        if self.completed_at and self.submitted_at:
            return self.completed_at - self.submitted_at
        return 0.0


class EfficiencyTracker:
    """Track and compute GPU scheduling efficiency vs FIFO baseline."""

    def __init__(self):
        self._active: Dict[str, JobLifecycle] = {}
        self._completed: deque = deque(maxlen=MAX_COMPLETED)
        self._lock = threading.Lock()
        self._gpu_node_count = 0
        self._pending_count = 0
        self._running_count = 0

    def on_pod_pending(self, pod_name: str, job_type: str, namespace: str,
                       gpu_type: str, timestamp: float):
        """Record a GPU pod entering Pending state."""
        with self._lock:
            if pod_name not in self._active:
                self._active[pod_name] = JobLifecycle(
                    pod_name=pod_name,
                    job_type=job_type,
                    namespace=namespace,
                    gpu_type=gpu_type,
                    submitted_at=timestamp,
                    status="pending",
                )

    def on_pod_running(self, pod_name: str, timestamp: float):
        """Record a GPU pod entering Running state."""
        with self._lock:
            job = self._active.get(pod_name)
            if job:
                job.started_at = timestamp
                job.status = "running"

    def on_pod_completed(self, pod_name: str, timestamp: float, succeeded: bool = True):
        """Record a GPU pod completing (Succeeded or Failed)."""
        with self._lock:
            job = self._active.pop(pod_name, None)
            if job:
                job.completed_at = timestamp
                job.status = "succeeded" if succeeded else "failed"
                self._completed.append(job)

    def update_counts(self, gpu_nodes: int, pending: int, running: int):
        """Update cluster-level counts."""
        with self._lock:
            self._gpu_node_count = gpu_nodes
            self._pending_count = pending
            self._running_count = running

    def _recent_completed(self) -> List[JobLifecycle]:
        """Get jobs completed within the rolling window."""
        cutoff = time.time() - ROLLING_WINDOW_SECONDS
        with self._lock:
            return [j for j in self._completed if j.completed_at >= cutoff]

    def calculate_fifo_baseline(self, jobs: List[JobLifecycle]) -> float:
        """Estimate total completion time under naive FIFO scheduling.

        FIFO baseline: jobs are assigned to GPUs in arrival order, one per GPU.
        Each GPU processes jobs sequentially. No smart placement or reordering.
        """
        if not jobs:
            return 0.0

        gpu_count = max(self._gpu_node_count, 1)

        # Sort by submission time (FIFO order)
        sorted_jobs = sorted(jobs, key=lambda j: j.submitted_at)

        # Simulate FIFO: each GPU has a queue, assign in round-robin
        gpu_finish_times = [0.0] * gpu_count
        total_times = []

        for i, job in enumerate(sorted_jobs):
            gpu_idx = i % gpu_count
            # Job starts when GPU is free or when job was submitted, whichever is later
            start = max(gpu_finish_times[gpu_idx], job.submitted_at)
            # Use actual run_time as the work duration
            finish = start + max(job.run_time, 1.0)
            gpu_finish_times[gpu_idx] = finish
            total_times.append(finish - job.submitted_at)

        return sum(total_times) / len(total_times) if total_times else 0.0

    def get_metrics(self) -> Dict:
        """Compute current efficiency metrics."""
        recent = self._recent_completed()
        succeeded = [j for j in recent if j.status == "succeeded"]

        # Actual average completion time (with Radix scheduling)
        avg_total = 0.0
        avg_wait = 0.0
        if succeeded:
            avg_total = sum(j.total_time for j in succeeded) / len(succeeded)
            avg_wait = sum(j.wait_time for j in succeeded) / len(succeeded)

        # FIFO baseline
        baseline = self.calculate_fifo_baseline(succeeded) if succeeded else 0.0

        # Efficiency: how much faster than FIFO
        efficiency_pct = 0.0
        if baseline > 0 and avg_total > 0:
            efficiency_pct = max(0.0, (baseline - avg_total) / baseline * 100)

        # Throughput
        throughput = len(succeeded)  # jobs per hour (window is 1 hour)

        # GPU utilization estimate
        gpu_util = 0.0
        if self._gpu_node_count > 0 and succeeded:
            total_gpu_seconds = sum(j.run_time for j in succeeded)
            possible_gpu_seconds = self._gpu_node_count * ROLLING_WINDOW_SECONDS
            gpu_util = min(100.0, total_gpu_seconds / possible_gpu_seconds * 100)

        with self._lock:
            return {
                "gpu_nodes": self._gpu_node_count,
                "pending": self._pending_count,
                "running": self._running_count,
                "completed_last_hour": len(succeeded),
                "avg_wait_time_seconds": round(avg_wait, 1),
                "avg_completion_time_seconds": round(avg_total, 1),
                "baseline_completion_time_seconds": round(baseline, 1),
                "efficiency_pct": round(efficiency_pct, 1),
                "throughput_jobs_per_hour": throughput,
                "gpu_utilization_pct": round(gpu_util, 1),
            }


# Global tracker instance
tracker = EfficiencyTracker()


def _is_gpu_pod(pod) -> bool:
    """Check if a pod requests GPU resources."""
    for container in pod.spec.containers or []:
        requests = (container.resources.requests or {}) if container.resources else {}
        for key in requests:
            if "nvidia.com/gpu" in key:
                return True
    return False


def _extract_job_type(pod) -> str:
    """Extract job type from pod annotations or labels."""
    annotations = pod.metadata.annotations or {}
    if "app.kubernetes.io/job-type" in annotations:
        return annotations["app.kubernetes.io/job-type"]
    labels = pod.metadata.labels or {}
    return labels.get("app.kubernetes.io/job-type",
                      labels.get("job-type", "unknown"))


def _extract_gpu_type(pod) -> str:
    """Extract GPU type from node selector or annotations."""
    annotations = pod.metadata.annotations or {}
    if "scheduler.radix.ai/chosen-gpu" in annotations:
        return annotations["scheduler.radix.ai/chosen-gpu"]
    node_selector = pod.spec.node_selector or {}
    return node_selector.get("gpu.nvidia.com/class",
                            node_selector.get("nvidia.com/gpu.product", "unknown"))


def _report_to_scheduler(job: JobLifecycle):
    """Fire-and-forget POST to scheduler /v1/observe."""
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            client.post(f"{SCHEDULER_URL}/v1/observe", json={
                "job_type": job.job_type,
                "gpu_type": job.gpu_type,
                "runtime_seconds": job.run_time,
                "wait_seconds": job.wait_time,
                "status": job.status,
            })
    except Exception as e:
        logger.debug(f"Failed to report to scheduler: {e}")


def _pod_watcher():
    """Background thread watching GPU pod lifecycle transitions."""
    try:
        from kubernetes import client, config, watch as k8s_watch
    except ImportError:
        logger.warning("kubernetes library not available, pod watcher disabled")
        return

    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception:
            logger.warning("Cannot load kube config, pod watcher disabled")
            return

    v1 = client.CoreV1Api()
    w = k8s_watch.Watch()

    logger.info("Pod watcher started")

    while True:
        try:
            for event in w.stream(v1.list_pod_for_all_namespaces, timeout_seconds=300):
                pod = event["object"]
                event_type = event["type"]

                if not _is_gpu_pod(pod):
                    continue

                pod_name = pod.metadata.name
                namespace = pod.metadata.namespace or "default"
                phase = pod.status.phase
                now = time.time()

                if phase == "Pending":
                    job_type = _extract_job_type(pod)
                    gpu_type = _extract_gpu_type(pod)
                    tracker.on_pod_pending(pod_name, job_type, namespace, gpu_type, now)

                elif phase == "Running":
                    tracker.on_pod_running(pod_name, now)

                elif phase in ("Succeeded", "Failed"):
                    tracker.on_pod_completed(pod_name, now, succeeded=(phase == "Succeeded"))
                    # Report to scheduler for learning loop
                    with tracker._lock:
                        recent = [j for j in tracker._completed if j.pod_name == pod_name]
                    if recent:
                        threading.Thread(target=_report_to_scheduler, args=(recent[-1],), daemon=True).start()

        except Exception as e:
            logger.error(f"Pod watcher error: {e}")
            time.sleep(5)


def _cluster_counter():
    """Background thread that periodically counts GPU nodes and pod states."""
    try:
        from kubernetes import client, config
    except ImportError:
        return

    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception:
            return

    v1 = client.CoreV1Api()

    while True:
        try:
            nodes = v1.list_node().items
            pods = v1.list_pod_for_all_namespaces().items

            gpu_nodes = sum(
                1 for n in nodes
                if any("nvidia.com/gpu" in (k or "") for k in (n.status.allocatable or {}))
            )
            gpu_pods = [p for p in pods if _is_gpu_pod(p)]
            pending = sum(1 for p in gpu_pods if p.status.phase == "Pending")
            running = sum(1 for p in gpu_pods if p.status.phase == "Running")

            tracker.update_counts(gpu_nodes, pending, running)
        except Exception as e:
            logger.debug(f"Cluster counter error: {e}")

        time.sleep(15)


def _timeseries_writer():
    """Background thread writing metrics to JSONL every 60 seconds."""
    while True:
        time.sleep(60)
        try:
            metrics = tracker.get_metrics()
            metrics["timestamp"] = time.time()
            with TS_FILE.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.debug(f"Timeseries writer error: {e}")


# FastAPI app
app = FastAPI(title="Radix Observer", version="2.0.0")


@app.on_event("startup")
def startup():
    """Start background threads."""
    for target in [_pod_watcher, _cluster_counter, _timeseries_writer]:
        t = threading.Thread(target=target, daemon=True)
        t.start()
    logger.info("Observer started with efficiency tracking")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/preview")
def preview():
    """Return current efficiency metrics for the dashboard."""
    return tracker.get_metrics()


@app.get("/v1/timeseries")
def timeseries():
    """Return historical timeseries data."""
    data = []
    if TS_FILE.exists():
        with TS_FILE.open() as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return {"data": data[-1000:]}
