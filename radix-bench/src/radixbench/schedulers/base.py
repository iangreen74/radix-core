"""Base classes for GPU schedulers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GPU:
    """GPU resource representation."""
    gpu_id: str
    memory_gb: float
    compute_capability: str = "8.0"
    is_available: bool = True
    current_job: Optional[str] = None


@dataclass
class Job:
    """Job representation for scheduling."""
    job_id: str
    submit_time: float
    runtime_estimate: float
    memory_gb: float
    priority: int = 0
    user: str = "default"
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    assigned_gpu: Optional[str] = None

    @property
    def actual_runtime(self) -> Optional[float]:
        """Calculate actual runtime if job is completed."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def wait_time(self) -> Optional[float]:
        """Calculate wait time if job has started."""
        if self.start_time is not None:
            return self.start_time - self.submit_time
        return None


@dataclass
class Assignment:
    """Job-to-GPU assignment."""
    job: Job
    gpu: GPU
    start_time: float


class Scheduler(ABC):
    """Abstract base class for GPU schedulers."""

    def __init__(self):
        self.name = self.__class__.__name__
        self.submitted_jobs: List[Job] = []
        self.running_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []

    @abstractmethod
    def submit(self, job: Job) -> None:
        """Submit a job to the scheduler."""
        pass

    @abstractmethod
    def schedule(self, cluster: List[GPU], current_time: float) -> List[Assignment]:
        """Schedule jobs on available GPUs."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get basic scheduler metrics."""
        if not self.completed_jobs:
            return {}

        wait_times = [job.wait_time for job in self.completed_jobs if job.wait_time is not None]
        runtimes = [job.actual_runtime for job in self.completed_jobs if job.actual_runtime is not None]

        return {
            "total_jobs": len(self.completed_jobs),
            "avg_wait_time": sum(wait_times) / len(wait_times) if wait_times else 0.0,
            "avg_runtime": sum(runtimes) / len(runtimes) if runtimes else 0.0,
            "throughput": len(self.completed_jobs) / max(job.end_time for job in self.completed_jobs) if self.completed_jobs else 0.0
        }
