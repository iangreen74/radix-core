"""
Job Planning Components for Radix Scheduler

Implements various planning algorithms for job scheduling and resource allocation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..logging import get_logger
from ..types import Job


@dataclass
class ExecutionPlan:
    """Execution plan for a set of jobs."""

    plan_id: str
    scheduled_jobs: List[Job]
    estimated_completion_time: datetime
    resource_requirements: Dict[str, float]
    dependencies_resolved: bool = True
    plan_metadata: Dict[str, Any] = field(default_factory=dict)


class SchedulePlanner(ABC):
    """Abstract base class for schedule planners."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"radix.scheduler.planner.{name}")

    @abstractmethod
    def create_execution_plan(self, jobs: List[Job]) -> ExecutionPlan:
        """Create an execution plan for the given jobs."""
        pass

    def validate_dependencies(self, jobs: List[Job]) -> bool:
        """Validate that job dependencies can be satisfied."""
        job_ids = {job.job_id for job in jobs}

        for job in jobs:
            for dep_id in job.dependencies:
                if dep_id not in job_ids:
                    self.logger.warning(f"Job {job.job_id} has unresolvable dependency: {dep_id}")
                    return False

        return True

    def topological_sort(self, jobs: List[Job]) -> List[Job]:
        """Sort jobs in topological order based on dependencies."""
        # Simple topological sort implementation
        in_degree = {job.job_id: 0 for job in jobs}

        # Calculate in-degrees
        for job in jobs:
            for dep_id in job.dependencies:
                if dep_id in in_degree:
                    in_degree[job.job_id] += 1

        # Process jobs with no dependencies first
        result = []
        queue = [job for job in jobs if in_degree[job.job_id] == 0]

        while queue:
            current_job = queue.pop(0)
            result.append(current_job)

            # Find jobs that depend on current job
            for job in jobs:
                if current_job.job_id in job.dependencies:
                    in_degree[job.job_id] -= 1
                    if in_degree[job.job_id] == 0:
                        queue.append(job)

        return result


class GreedyPlanner(SchedulePlanner):
    """Greedy planner that schedules jobs based on priority and resource availability."""

    def __init__(self):
        super().__init__("greedy")

    def create_execution_plan(self, jobs: List[Job]) -> ExecutionPlan:
        """Create execution plan using greedy approach."""
        if not jobs:
            return ExecutionPlan(
                plan_id=f"greedy_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                scheduled_jobs=[],
                estimated_completion_time=datetime.now(),
                resource_requirements={},
            )

        # Validate dependencies
        dependencies_ok = self.validate_dependencies(jobs)

        # Sort by dependencies first, then by priority
        if dependencies_ok:
            scheduled_jobs = self.topological_sort(jobs)
            # Within each level, sort by priority
            scheduled_jobs.sort(key=lambda x: (-x.priority, x.job_id))
        else:
            # Fallback to priority-only sorting
            scheduled_jobs = sorted(jobs, key=lambda x: (-x.priority, x.job_id))

        # Calculate resource requirements
        total_cpu = sum(job.requirements.cpu_cores for job in scheduled_jobs)
        total_memory = sum(job.requirements.memory_mb for job in scheduled_jobs)
        total_gpu = sum(job.requirements.gpu_count for job in scheduled_jobs)

        # Estimate completion time (simplified)
        total_runtime = sum(job.requirements.max_runtime_seconds or 60 for job in scheduled_jobs)
        estimated_completion = datetime.now()

        plan = ExecutionPlan(
            plan_id=f"greedy_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scheduled_jobs=scheduled_jobs,
            estimated_completion_time=estimated_completion,
            resource_requirements={
                "cpu_cores": total_cpu,
                "memory_gb": total_memory,
                "gpu_count": total_gpu,
            },
            dependencies_resolved=dependencies_ok,
            plan_metadata={
                "planner_type": "greedy",
                "job_count": len(scheduled_jobs),
                "estimated_runtime_seconds": total_runtime,
            },
        )

        self.logger.info(
            "Created greedy execution plan",
            job_count=len(scheduled_jobs),
            dependencies_resolved=dependencies_ok,
            total_cpu=total_cpu,
            total_memory=total_memory,
        )

        return plan


class OptimalPlanner(SchedulePlanner):
    """Optimal planner that tries to find the best schedule (simplified implementation)."""

    def __init__(self):
        super().__init__("optimal")

    def create_execution_plan(self, jobs: List[Job]) -> ExecutionPlan:
        """Create optimal execution plan."""
        if not jobs:
            return ExecutionPlan(
                plan_id=f"optimal_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                scheduled_jobs=[],
                estimated_completion_time=datetime.now(),
                resource_requirements={},
            )

        # For simplicity, use greedy approach but with more sophisticated scoring
        dependencies_ok = self.validate_dependencies(jobs)

        if dependencies_ok:
            scheduled_jobs = self.topological_sort(jobs)
        else:
            scheduled_jobs = jobs.copy()

        # Score jobs based on multiple criteria
        def job_score(job: Job) -> float:
            # Higher priority, shorter runtime, fewer resources = higher score
            resource_cost = (
                job.requirements.cpu_cores
                + job.requirements.memory_mb
                + job.requirements.gpu_count * 2
            )

            runtime = (job.requirements.max_runtime_seconds or 60) / 60
            return job.priority * 100 - runtime - resource_cost

        # Sort by score within dependency levels
        scheduled_jobs.sort(key=job_score, reverse=True)

        # Calculate resource requirements
        total_cpu = sum(job.requirements.cpu_cores for job in scheduled_jobs)
        total_memory = sum(job.requirements.memory_mb for job in scheduled_jobs)
        total_gpu = sum(job.requirements.gpu_count for job in scheduled_jobs)

        plan = ExecutionPlan(
            plan_id=f"optimal_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scheduled_jobs=scheduled_jobs,
            estimated_completion_time=datetime.now(),
            resource_requirements={
                "cpu_cores": total_cpu,
                "memory_gb": total_memory,
                "gpu_count": total_gpu,
            },
            dependencies_resolved=dependencies_ok,
            plan_metadata={
                "planner_type": "optimal",
                "job_count": len(scheduled_jobs),
                "optimization_score": sum(job_score(job) for job in scheduled_jobs),
            },
        )

        self.logger.info(
            "Created optimal execution plan",
            job_count=len(scheduled_jobs),
            dependencies_resolved=dependencies_ok,
            optimization_score=plan.plan_metadata["optimization_score"],
        )

        return plan


# Default planner factory
def get_planner(planner_type: str = "greedy") -> SchedulePlanner:
    """Get a planner instance by type."""
    if planner_type == "greedy":
        return GreedyPlanner()
    elif planner_type == "optimal":
        return OptimalPlanner()
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")


# Alias for backwards compatibility
JobPlanner = GreedyPlanner
