"""
Scheduling Policies for Radix

This module implements various scheduling policies that determine the order
and priority of job execution within the Radix orchestration system.
"""

import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta

from ..types import Job, ResourceRequirements
from ..config import get_config
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision for a job."""

    job_id: str
    priority_score: float
    estimated_start_time: datetime
    estimated_duration: float
    resource_allocation: Dict[str, Any]
    reasoning: str
    metadata: Dict[str, Any]


class SchedulingPolicy(ABC):
    """
    Abstract base class for scheduling policies.

    Scheduling policies determine the order in which ready jobs are executed
    based on various criteria such as priority, fairness, resource requirements, etc.
    """

    def __init__(self, name: str):
        self.name = name
        self.config = get_config()
        self.decisions_made = 0
        self.total_wait_time = 0.0
        self.job_history: List[SchedulingDecision] = []

    @abstractmethod
    def select_jobs(self, ready_jobs: List[Job], available_resources: ResourceRequirements,
                   max_jobs: int = None) -> List[SchedulingDecision]:
        """
        Select jobs to schedule from the list of ready jobs.

        Args:
            ready_jobs: List of jobs ready for execution
            available_resources: Currently available resources
            max_jobs: Maximum number of jobs to select (None for no limit)

        Returns:
            List of scheduling decisions ordered by priority
        """
        pass

    @abstractmethod
    def calculate_priority(self, job: Job, context: Dict[str, Any] = None) -> float:
        """
        Calculate priority score for a job.

        Args:
            job: Job to calculate priority for
            context: Additional context for priority calculation

        Returns:
            Priority score (higher = higher priority)
        """
        pass

    def record_decision(self, decision: SchedulingDecision):
        """Record a scheduling decision for analysis."""
        self.decisions_made += 1
        self.job_history.append(decision)

        # Keep history manageable
        if len(self.job_history) > 1000:
            self.job_history = self.job_history[-500:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this scheduling policy's performance."""
        if not self.job_history:
            return {
                'policy_name': self.name,
                'decisions_made': self.decisions_made,
                'avg_priority_score': 0.0,
                'avg_estimated_duration': 0.0
            }

        priority_scores = [d.priority_score for d in self.job_history]
        durations = [d.estimated_duration for d in self.job_history]

        return {
            'policy_name': self.name,
            'decisions_made': self.decisions_made,
            'avg_priority_score': sum(priority_scores) / len(priority_scores),
            'max_priority_score': max(priority_scores),
            'min_priority_score': min(priority_scores),
            'avg_estimated_duration': sum(durations) / len(durations),
            'total_estimated_duration': sum(durations)
        }


class FIFOPolicy(SchedulingPolicy):
    """
    First-In-First-Out scheduling policy.

    Jobs are scheduled in the order they were submitted, with no consideration
    for priority or resource requirements beyond basic feasibility.
    """

    def __init__(self):
        super().__init__("FIFO")

    def select_jobs(self, ready_jobs: List[Job], available_resources: ResourceRequirements,
                   max_jobs: int = None) -> List[SchedulingDecision]:
        """Select jobs in FIFO order."""
        if not ready_jobs:
            return []

        # Sort by creation time (FIFO)
        sorted_jobs = sorted(ready_jobs, key=lambda j: j.created_at)

        decisions = []
        current_time = datetime.utcnow()

        for job in sorted_jobs:
            if max_jobs and len(decisions) >= max_jobs:
                break

            # Check if resources are available
            if not job.requirements.is_satisfied_by(available_resources):
                continue

            priority_score = self.calculate_priority(job)

            decision = SchedulingDecision(
                job_id=job.job_id,
                priority_score=priority_score,
                estimated_start_time=current_time,
                estimated_duration=job.estimated_duration(),
                resource_allocation=self._create_resource_allocation(job.requirements),
                reasoning=f"FIFO order (created at {job.created_at.isoformat()})",
                metadata={'policy': self.name, 'creation_time': job.created_at.isoformat()}
            )

            decisions.append(decision)
            self.record_decision(decision)

            # Update available resources for next iteration
            available_resources = self._subtract_resources(available_resources, job.requirements)

        return decisions

    def calculate_priority(self, job: Job, context: Dict[str, Any] = None) -> float:
        """Calculate FIFO priority (earlier jobs have higher priority)."""
        # Use negative timestamp so earlier jobs have higher priority
        return -job.created_at.timestamp()

    def _create_resource_allocation(self, requirements: ResourceRequirements) -> Dict[str, Any]:
        """Create resource allocation dictionary."""
        return {
            'cpu_cores': requirements.cpu_cores,
            'memory_mb': requirements.memory_mb,
            'gpu_count': requirements.gpu_count,
            'gpu_memory_mb': requirements.gpu_memory_mb
        }

    def _subtract_resources(self, available: ResourceRequirements,
                          required: ResourceRequirements) -> ResourceRequirements:
        """Subtract required resources from available resources."""
        return ResourceRequirements(
            cpu_cores=max(0, available.cpu_cores - required.cpu_cores),
            memory_mb=max(0, available.memory_mb - required.memory_mb),
            gpu_count=max(0, available.gpu_count - required.gpu_count),
            gpu_memory_mb=max(0, available.gpu_memory_mb - required.gpu_memory_mb),
            storage_mb=max(0, available.storage_mb - required.storage_mb),
            network_mbps=max(0, available.network_mbps - required.network_mbps)
        )


class PriorityPolicy(SchedulingPolicy):
    """
    Priority-based scheduling policy.

    Jobs are scheduled based on their priority value, with higher priority
    jobs scheduled first. Ties are broken by submission time.
    """

    def __init__(self, priority_weights: Dict[str, float] = None):
        super().__init__("Priority")
        self.priority_weights = priority_weights or {
            'base_priority': 1.0,
            'age_factor': 0.1,
            'resource_efficiency': 0.2,
            'estimated_duration': -0.05  # Prefer shorter jobs slightly
        }

    def select_jobs(self, ready_jobs: List[Job], available_resources: ResourceRequirements,
                   max_jobs: int = None) -> List[SchedulingDecision]:
        """Select jobs based on priority scores."""
        if not ready_jobs:
            return []

        # Calculate priority scores for all jobs
        job_priorities = []
        for job in ready_jobs:
            if job.requirements.is_satisfied_by(available_resources):
                priority_score = self.calculate_priority(job)
                job_priorities.append((priority_score, job))

        # Sort by priority (highest first)
        job_priorities.sort(key=lambda x: x[0], reverse=True)

        decisions = []
        current_time = datetime.utcnow()
        remaining_resources = available_resources

        for priority_score, job in job_priorities:
            if max_jobs and len(decisions) >= max_jobs:
                break

            if not job.requirements.is_satisfied_by(remaining_resources):
                continue

            decision = SchedulingDecision(
                job_id=job.job_id,
                priority_score=priority_score,
                estimated_start_time=current_time,
                estimated_duration=job.estimated_duration(),
                resource_allocation=self._create_resource_allocation(job.requirements),
                reasoning=f"Priority score: {priority_score:.2f}",
                metadata={
                    'policy': self.name,
                    'base_priority': job.priority,
                    'calculated_priority': priority_score
                }
            )

            decisions.append(decision)
            self.record_decision(decision)

            # Update remaining resources
            remaining_resources = self._subtract_resources(remaining_resources, job.requirements)

        return decisions

    def calculate_priority(self, job: Job, context: Dict[str, Any] = None) -> float:
        """Calculate comprehensive priority score."""
        current_time = datetime.utcnow()

        # Base priority from job
        base_priority = job.priority * self.priority_weights['base_priority']

        # Age factor (older jobs get higher priority)
        age_hours = (current_time - job.created_at).total_seconds() / 3600.0
        age_factor = age_hours * self.priority_weights['age_factor']

        # Resource efficiency (prefer jobs that use resources efficiently)
        resource_efficiency = self._calculate_resource_efficiency(job.requirements)
        efficiency_factor = resource_efficiency * self.priority_weights['resource_efficiency']

        # Duration factor (slight preference for shorter jobs)
        duration_factor = job.estimated_duration() * self.priority_weights['estimated_duration']

        total_priority = base_priority + age_factor + efficiency_factor + duration_factor

        logger.debug("Priority calculation",
                    job_id=job.job_id,
                    base_priority=base_priority,
                    age_factor=age_factor,
                    efficiency_factor=efficiency_factor,
                    duration_factor=duration_factor,
                    total_priority=total_priority)

        return total_priority

    def _calculate_resource_efficiency(self, requirements: ResourceRequirements) -> float:
        """Calculate how efficiently a job uses resources."""
        # Simple heuristic: jobs that use multiple resource types efficiently
        # get higher scores
        resource_usage = 0.0

        if requirements.cpu_cores > 0:
            resource_usage += min(requirements.cpu_cores, 8.0) / 8.0  # Normalize to 8 cores

        if requirements.memory_mb > 0:
            resource_usage += min(requirements.memory_mb, 16384) / 16384  # Normalize to 16GB

        if requirements.gpu_count > 0:
            resource_usage += min(requirements.gpu_count, 4) / 4  # Normalize to 4 GPUs

        return resource_usage

    def _create_resource_allocation(self, requirements: ResourceRequirements) -> Dict[str, Any]:
        """Create resource allocation dictionary."""
        return {
            'cpu_cores': requirements.cpu_cores,
            'memory_mb': requirements.memory_mb,
            'gpu_count': requirements.gpu_count,
            'gpu_memory_mb': requirements.gpu_memory_mb
        }

    def _subtract_resources(self, available: ResourceRequirements,
                          required: ResourceRequirements) -> ResourceRequirements:
        """Subtract required resources from available resources."""
        return ResourceRequirements(
            cpu_cores=max(0, available.cpu_cores - required.cpu_cores),
            memory_mb=max(0, available.memory_mb - required.memory_mb),
            gpu_count=max(0, available.gpu_count - required.gpu_count),
            gpu_memory_mb=max(0, available.gpu_memory_mb - required.gpu_memory_mb),
            storage_mb=max(0, available.storage_mb - required.storage_mb),
            network_mbps=max(0, available.network_mbps - required.network_mbps)
        )


class FairSharePolicy(SchedulingPolicy):
    """
    Fair share scheduling policy.

    Ensures fair resource allocation across different users, projects, or job types
    by tracking resource usage and prioritizing underutilized entities.
    """

    def __init__(self, fairness_window_hours: float = 24.0):
        super().__init__("FairShare")
        self.fairness_window = timedelta(hours=fairness_window_hours)
        self.usage_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.fair_shares: Dict[str, float] = defaultdict(lambda: 1.0)  # Equal shares by default

    def select_jobs(self, ready_jobs: List[Job], available_resources: ResourceRequirements,
                   max_jobs: int = None) -> List[SchedulingDecision]:
        """Select jobs based on fair share principles."""
        if not ready_jobs:
            return []

        # Update usage history
        self._cleanup_old_usage()

        # Group jobs by user/project (using tags)
        job_groups = self._group_jobs_by_entity(ready_jobs)

        # Calculate fair share priorities
        entity_priorities = self._calculate_entity_priorities(job_groups.keys())

        # Select jobs using fair share algorithm
        decisions = []
        current_time = datetime.utcnow()
        remaining_resources = available_resources

        # Create priority queue of (priority, entity, job_index)
        priority_queue = []

        for entity, jobs in job_groups.items():
            entity_priority = entity_priorities[entity]
            for i, job in enumerate(jobs):
                if job.requirements.is_satisfied_by(remaining_resources):
                    job_priority = self.calculate_priority(job, {'entity_priority': entity_priority})
                    heapq.heappush(priority_queue, (-job_priority, entity, i, job))

        # Select jobs from priority queue
        while priority_queue and (not max_jobs or len(decisions) < max_jobs):
            neg_priority, entity, job_index, job = heapq.heappop(priority_queue)
            priority_score = -neg_priority

            if not job.requirements.is_satisfied_by(remaining_resources):
                continue

            decision = SchedulingDecision(
                job_id=job.job_id,
                priority_score=priority_score,
                estimated_start_time=current_time,
                estimated_duration=job.estimated_duration(),
                resource_allocation=self._create_resource_allocation(job.requirements),
                reasoning=f"Fair share priority: {priority_score:.2f} (entity: {entity})",
                metadata={
                    'policy': self.name,
                    'entity': entity,
                    'entity_priority': entity_priorities[entity],
                    'fair_share_score': priority_score
                }
            )

            decisions.append(decision)
            self.record_decision(decision)

            # Record resource usage for this entity
            self._record_usage(entity, job.requirements, job.estimated_duration())

            # Update remaining resources
            remaining_resources = self._subtract_resources(remaining_resources, job.requirements)

        return decisions

    def calculate_priority(self, job: Job, context: Dict[str, Any] = None) -> float:
        """Calculate fair share priority score."""
        context = context or {}
        entity_priority = context.get('entity_priority', 1.0)

        # Base priority from job
        base_priority = job.priority

        # Age factor
        age_hours = (datetime.utcnow() - job.created_at).total_seconds() / 3600.0
        age_factor = min(age_hours, 24.0)  # Cap at 24 hours

        # Combine factors
        total_priority = (base_priority + age_factor) * entity_priority

        return total_priority

    def set_fair_share(self, entity: str, share: float):
        """Set the fair share allocation for an entity."""
        self.fair_shares[entity] = max(0.0, share)

    def _group_jobs_by_entity(self, jobs: List[Job]) -> Dict[str, List[Job]]:
        """Group jobs by entity (user, project, etc.)."""
        groups = defaultdict(list)

        for job in jobs:
            # Try to extract entity from job tags
            entity = job.tags.get('user', job.tags.get('project', 'default'))
            groups[entity].append(job)

        return dict(groups)

    def _calculate_entity_priorities(self, entities: Set[str]) -> Dict[str, float]:
        """Calculate priority multipliers for entities based on fair share."""
        priorities = {}

        for entity in entities:
            fair_share = self.fair_shares[entity]
            actual_usage = self._get_recent_usage(entity)

            # Calculate priority as inverse of usage ratio
            if actual_usage > 0:
                usage_ratio = actual_usage / fair_share
                priority = 1.0 / max(usage_ratio, 0.1)  # Avoid division by zero
            else:
                priority = 2.0  # High priority for entities with no recent usage

            priorities[entity] = priority

        return priorities

    def _get_recent_usage(self, entity: str) -> float:
        """Get recent resource usage for an entity."""
        if entity not in self.usage_history:
            return 0.0

        cutoff_time = datetime.utcnow() - self.fairness_window
        recent_usage = 0.0

        for usage_record in self.usage_history[entity]:
            if usage_record['timestamp'] >= cutoff_time:
                # Simple usage metric: CPU hours + memory GB hours + GPU hours
                cpu_hours = usage_record['cpu_cores'] * usage_record['duration_hours']
                memory_gb_hours = (usage_record['memory_mb'] / 1024.0) * usage_record['duration_hours']
                gpu_hours = usage_record['gpu_count'] * usage_record['duration_hours']

                recent_usage += cpu_hours + memory_gb_hours + gpu_hours

        return recent_usage

    def _record_usage(self, entity: str, requirements: ResourceRequirements, duration_seconds: float):
        """Record resource usage for an entity."""
        usage_record = {
            'timestamp': datetime.utcnow(),
            'cpu_cores': requirements.cpu_cores,
            'memory_mb': requirements.memory_mb,
            'gpu_count': requirements.gpu_count,
            'duration_hours': duration_seconds / 3600.0
        }

        self.usage_history[entity].append(usage_record)

    def _cleanup_old_usage(self):
        """Remove old usage records outside the fairness window."""
        cutoff_time = datetime.utcnow() - self.fairness_window

        for entity in self.usage_history:
            self.usage_history[entity] = [
                record for record in self.usage_history[entity]
                if record['timestamp'] >= cutoff_time
            ]

    def _create_resource_allocation(self, requirements: ResourceRequirements) -> Dict[str, Any]:
        """Create resource allocation dictionary."""
        return {
            'cpu_cores': requirements.cpu_cores,
            'memory_mb': requirements.memory_mb,
            'gpu_count': requirements.gpu_count,
            'gpu_memory_mb': requirements.gpu_memory_mb
        }

    def _subtract_resources(self, available: ResourceRequirements,
                          required: ResourceRequirements) -> ResourceRequirements:
        """Subtract required resources from available resources."""
        return ResourceRequirements(
            cpu_cores=max(0, available.cpu_cores - required.cpu_cores),
            memory_mb=max(0, available.memory_mb - required.memory_mb),
            gpu_count=max(0, available.gpu_count - required.gpu_count),
            gpu_memory_mb=max(0, available.gpu_memory_mb - required.gpu_memory_mb),
            storage_mb=max(0, available.storage_mb - required.storage_mb),
            network_mbps=max(0, available.network_mbps - required.network_mbps)
        )


class ShortestJobFirstPolicy(SchedulingPolicy):
    """
    Shortest Job First (SJF) scheduling policy.

    Prioritizes jobs with the shortest estimated execution time to minimize
    average waiting time and improve system throughput.
    """

    def __init__(self):
        super().__init__("ShortestJobFirst")

    def select_jobs(self, ready_jobs: List[Job], available_resources: ResourceRequirements,
                   max_jobs: int = None) -> List[SchedulingDecision]:
        """Select jobs based on shortest estimated duration."""
        if not ready_jobs:
            return []

        # Filter jobs that can fit in available resources
        feasible_jobs = [job for job in ready_jobs
                        if job.requirements.is_satisfied_by(available_resources)]

        # Sort by estimated duration (shortest first)
        sorted_jobs = sorted(feasible_jobs, key=lambda j: j.estimated_duration())

        decisions = []
        current_time = datetime.utcnow()
        remaining_resources = available_resources

        for job in sorted_jobs:
            if max_jobs and len(decisions) >= max_jobs:
                break

            if not job.requirements.is_satisfied_by(remaining_resources):
                continue

            priority_score = self.calculate_priority(job)

            decision = SchedulingDecision(
                job_id=job.job_id,
                priority_score=priority_score,
                estimated_start_time=current_time,
                estimated_duration=job.estimated_duration(),
                resource_allocation=self._create_resource_allocation(job.requirements),
                reasoning=f"Shortest job first (duration: {job.estimated_duration():.1f}s)",
                metadata={
                    'policy': self.name,
                    'estimated_duration': job.estimated_duration()
                }
            )

            decisions.append(decision)
            self.record_decision(decision)

            # Update remaining resources
            remaining_resources = self._subtract_resources(remaining_resources, job.requirements)

        return decisions

    def calculate_priority(self, job: Job, context: Dict[str, Any] = None) -> float:
        """Calculate SJF priority (shorter jobs have higher priority)."""
        # Use negative duration so shorter jobs have higher priority
        base_priority = -job.estimated_duration()

        # Add small bonus for job priority to break ties
        priority_bonus = job.priority * 0.1

        return base_priority + priority_bonus

    def _create_resource_allocation(self, requirements: ResourceRequirements) -> Dict[str, Any]:
        """Create resource allocation dictionary."""
        return {
            'cpu_cores': requirements.cpu_cores,
            'memory_mb': requirements.memory_mb,
            'gpu_count': requirements.gpu_count,
            'gpu_memory_mb': requirements.gpu_memory_mb
        }

    def _subtract_resources(self, available: ResourceRequirements,
                          required: ResourceRequirements) -> ResourceRequirements:
        """Subtract required resources from available resources."""
        return ResourceRequirements(
            cpu_cores=max(0, available.cpu_cores - required.cpu_cores),
            memory_mb=max(0, available.memory_mb - required.memory_mb),
            gpu_count=max(0, available.gpu_count - required.gpu_count),
            gpu_memory_mb=max(0, available.gpu_memory_mb - required.gpu_memory_mb),
            storage_mb=max(0, available.storage_mb - required.storage_mb),
            network_mbps=max(0, available.network_mbps - required.network_mbps)
        )
