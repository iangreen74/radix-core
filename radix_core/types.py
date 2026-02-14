"""
Core Type Definitions for Radix

This module defines the fundamental data types used throughout the Radix
system for job management, scheduling, and execution.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


class JobStatus(Enum):
    """Status of a job in the system."""

    PENDING = auto()  # Job created but not yet scheduled
    SCHEDULED = auto()  # Job scheduled but not yet running
    RUNNING = auto()  # Job currently executing
    COMPLETED = auto()  # Job completed successfully
    FAILED = auto()  # Job failed with error
    CANCELLED = auto()  # Job cancelled by user or system
    TIMEOUT = auto()  # Job exceeded time limit


class ResourceType(Enum):
    """Types of computational resources."""

    CPU = auto()
    MEMORY = auto()
    GPU = auto()
    STORAGE = auto()
    NETWORK = auto()


class ExecutorType(Enum):
    """Types of job executors."""

    LOCAL_SUBPROCESS = auto()
    THREADPOOL = auto()
    RAY_LOCAL = auto()


@dataclass
class ResourceRequirements:
    """Resource requirements for a job."""

    cpu_cores: float = 1.0  # Number of CPU cores needed
    memory_mb: int = 512  # Memory in megabytes
    gpu_count: int = 0  # Number of GPUs needed
    gpu_memory_mb: int = 0  # GPU memory in megabytes
    storage_mb: int = 100  # Storage space in megabytes
    network_mbps: float = 0.0  # Network bandwidth in Mbps

    # Time limits
    max_runtime_seconds: Optional[int] = None

    # Special requirements
    requires_local_only: bool = True  # Safety: always local only

    def __post_init__(self):
        """Validate resource requirements."""
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")

        if self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive")

        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")

        if self.gpu_memory_mb < 0:
            raise ValueError("gpu_memory_mb cannot be negative")

        if self.storage_mb < 0:
            raise ValueError("storage_mb cannot be negative")

        if self.network_mbps < 0:
            raise ValueError("network_mbps cannot be negative")

        # Safety check: ensure local-only execution
        if not self.requires_local_only:
            raise ValueError("requires_local_only must be True for safety")

    def total_cost_estimate(self) -> float:
        """Estimate total cost for these resources (always $0.00 in dry-run)."""
        # In dry-run mode, all costs are $0.00
        return 0.0

    def is_satisfied_by(self, available: "ResourceRequirements") -> bool:
        """Check if requirements are satisfied by available resources."""
        return (
            self.cpu_cores <= available.cpu_cores
            and self.memory_mb <= available.memory_mb
            and self.gpu_count <= available.gpu_count
            and self.gpu_memory_mb <= available.gpu_memory_mb
            and self.storage_mb <= available.storage_mb
            and self.network_mbps <= available.network_mbps
        )


@dataclass
class Job:
    """Represents a single job to be executed."""

    # Identification
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Execution details
    command: Optional[str] = None
    function: Optional[Callable] = None
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Requirements and constraints
    requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    priority: int = 0  # Higher numbers = higher priority

    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Job IDs this job depends on

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Status tracking
    status: JobStatus = JobStatus.PENDING

    def __post_init__(self):
        """Validate job configuration."""
        if not self.job_id:
            raise ValueError("job_id cannot be empty")

        if not self.command and not self.function:
            raise ValueError("Either command or function must be specified")

        if self.command and self.function:
            raise ValueError("Cannot specify both command and function")

        if not self.name:
            self.name = f"job_{self.job_id[:8]}"

    def is_ready_to_run(self, completed_jobs: set) -> bool:
        """Check if job is ready to run (all dependencies completed)."""
        return all(dep_id in completed_jobs for dep_id in self.dependencies)

    def estimated_duration(self) -> float:
        """Estimate job duration in seconds (simplified heuristic)."""
        # Simple heuristic based on resource requirements
        base_duration = 10.0  # Base 10 seconds

        # Scale by CPU requirements
        cpu_factor = max(1.0, self.requirements.cpu_cores)

        # Scale by memory requirements (higher memory = more complex job)
        memory_factor = max(1.0, self.requirements.memory_mb / 1024.0)

        # GPU jobs typically take longer to initialize
        gpu_factor = 1.5 if self.requirements.gpu_count > 0 else 1.0

        return base_duration * cpu_factor * memory_factor * gpu_factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "description": self.description,
            "command": self.command,
            "args": self.args,
            "kwargs": self.kwargs,
            "requirements": {
                "cpu_cores": self.requirements.cpu_cores,
                "memory_mb": self.requirements.memory_mb,
                "gpu_count": self.requirements.gpu_count,
                "gpu_memory_mb": self.requirements.gpu_memory_mb,
                "storage_mb": self.requirements.storage_mb,
                "network_mbps": self.requirements.network_mbps,
                "max_runtime_seconds": self.requirements.max_runtime_seconds,
            },
            "priority": self.priority,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "status": self.status.name,
        }


@dataclass
class JobResult:
    """Result of job execution."""

    job_id: str
    status: JobStatus

    # Execution details
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    executor_type: Optional[str] = None
    node_id: Optional[str] = None

    # Results
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    result_data: Any = None

    # Resource usage
    cpu_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    gpu_time_seconds: float = 0.0

    # Error information
    error_message: Optional[str] = None
    exception: Optional[Exception] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def succeeded(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED and (
            self.return_code is None or self.return_code == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "job_id": self.job_id,
            "status": self.status.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "executor_type": self.executor_type,
            "node_id": self.node_id,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "cpu_time_seconds": self.cpu_time_seconds,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_time_seconds": self.gpu_time_seconds,
            "error_message": self.error_message,
            "succeeded": self.succeeded,
            "metadata": self.metadata,
        }


@dataclass
class ResourceAllocation:
    """Represents allocated resources for a job."""

    job_id: str
    node_id: str

    # Allocated resources
    cpu_cores: float
    memory_mb: int
    gpu_indices: List[int] = field(default_factory=list)
    gpu_memory_mb: int = 0

    # Allocation metadata
    allocated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def matches_requirements(self, requirements: ResourceRequirements) -> bool:
        """Check if allocation matches job requirements."""
        return (
            self.cpu_cores >= requirements.cpu_cores
            and self.memory_mb >= requirements.memory_mb
            and len(self.gpu_indices) >= requirements.gpu_count
            and self.gpu_memory_mb >= requirements.gpu_memory_mb
        )


@dataclass
class SchedulePlan:
    """Represents a schedule for executing jobs."""

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    jobs: List[Job] = field(default_factory=list)

    # Resource allocations
    allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)

    # Scheduling metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    estimated_duration_seconds: float = 0.0
    estimated_cost_usd: float = 0.0  # Always $0.00 in dry-run mode

    # Execution order
    execution_order: List[str] = field(default_factory=list)  # Job IDs in execution order

    def __post_init__(self):
        """Validate schedule plan."""
        # Ensure all jobs have allocations
        job_ids = {job.job_id for job in self.jobs}
        allocation_job_ids = set(self.allocations.keys())

        missing_allocations = job_ids - allocation_job_ids
        if missing_allocations:
            raise ValueError(f"Missing allocations for jobs: {missing_allocations}")

        # Ensure execution order contains all jobs
        execution_set = set(self.execution_order)
        if execution_set != job_ids:
            raise ValueError("Execution order must contain exactly all job IDs")

        # Safety: ensure cost is $0.00 in dry-run mode
        if self.estimated_cost_usd != 0.0:
            raise ValueError("estimated_cost_usd must be $0.00 in dry-run mode")

    def get_job_by_id(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None

    def get_ready_jobs(self, completed_jobs: set) -> List[Job]:
        """Get jobs that are ready to run."""
        ready_jobs = []
        for job in self.jobs:
            if job.status == JobStatus.PENDING and job.is_ready_to_run(completed_jobs):
                ready_jobs.append(job)
        return ready_jobs

    def total_resource_requirements(self) -> ResourceRequirements:
        """Calculate total resource requirements for all jobs."""
        total_cpu = sum(job.requirements.cpu_cores for job in self.jobs)
        total_memory = sum(job.requirements.memory_mb for job in self.jobs)
        total_gpu = sum(job.requirements.gpu_count for job in self.jobs)
        total_gpu_memory = sum(job.requirements.gpu_memory_mb for job in self.jobs)
        total_storage = sum(job.requirements.storage_mb for job in self.jobs)
        total_network = max((job.requirements.network_mbps for job in self.jobs), default=0.0)

        return ResourceRequirements(
            cpu_cores=total_cpu,
            memory_mb=total_memory,
            gpu_count=total_gpu,
            gpu_memory_mb=total_gpu_memory,
            storage_mb=total_storage,
            network_mbps=total_network,
        )


@dataclass
class ExecutionResult:
    """Result of executing a schedule plan."""

    plan_id: str

    # Execution summary
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int

    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None

    # Individual job results
    job_results: Dict[str, JobResult] = field(default_factory=dict)

    # Resource usage summary
    total_cpu_time_seconds: float = 0.0
    total_memory_peak_mb: float = 0.0
    total_gpu_time_seconds: float = 0.0

    # Cost tracking (always $0.00 in dry-run mode)
    actual_cost_usd: float = 0.0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total execution duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as fraction of completed jobs."""
        if self.total_jobs == 0:
            return 0.0
        return self.completed_jobs / self.total_jobs

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.end_time is not None

    def add_job_result(self, result: JobResult):
        """Add a job result to the execution result."""
        self.job_results[result.job_id] = result

        # Update counters based on job status
        if result.status == JobStatus.COMPLETED:
            self.completed_jobs += 1
        elif result.status == JobStatus.FAILED:
            self.failed_jobs += 1
        elif result.status == JobStatus.CANCELLED:
            self.cancelled_jobs += 1

        # Update resource usage
        self.total_cpu_time_seconds += result.cpu_time_seconds
        self.total_memory_peak_mb = max(self.total_memory_peak_mb, result.memory_peak_mb)
        self.total_gpu_time_seconds += result.gpu_time_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary."""
        return {
            "plan_id": self.plan_id,
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "cancelled_jobs": self.cancelled_jobs,
            "success_rate": self.success_rate,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_cpu_time_seconds": self.total_cpu_time_seconds,
            "total_memory_peak_mb": self.total_memory_peak_mb,
            "total_gpu_time_seconds": self.total_gpu_time_seconds,
            "actual_cost_usd": self.actual_cost_usd,
            "is_complete": self.is_complete,
        }


# Type aliases for convenience
JobID = str
NodeID = str
PlanID = str
