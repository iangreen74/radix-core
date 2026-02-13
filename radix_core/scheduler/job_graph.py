"""
Job Graph Implementation for Radix

This module provides a directed acyclic graph (DAG) representation for job
dependencies and execution ordering. The job graph is the foundation for
scheduling and orchestration decisions.
"""

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Iterator
from datetime import datetime

from ..types import Job, JobStatus, JobID
from ..errors import DependencyError, ValidationError
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class JobNode:
    """Represents a node in the job graph."""

    job: Job
    dependencies: Set[JobID] = field(default_factory=set)
    dependents: Set[JobID] = field(default_factory=set)

    # Scheduling metadata
    ready_time: Optional[datetime] = None
    scheduled_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None

    @property
    def job_id(self) -> JobID:
        """Get the job ID."""
        return self.job.job_id

    @property
    def is_ready(self) -> bool:
        """Check if job is ready to run (no pending dependencies)."""
        return len(self.dependencies) == 0

    def add_dependency(self, job_id: JobID):
        """Add a dependency to this job."""
        self.dependencies.add(job_id)

    def remove_dependency(self, job_id: JobID):
        """Remove a dependency from this job."""
        self.dependencies.discard(job_id)

    def add_dependent(self, job_id: JobID):
        """Add a dependent job."""
        self.dependents.add(job_id)

    def remove_dependent(self, job_id: JobID):
        """Remove a dependent job."""
        self.dependents.discard(job_id)


@dataclass
class JobEdge:
    """Represents an edge (dependency) in the job graph."""

    from_job_id: JobID
    to_job_id: JobID
    edge_type: str = "dependency"  # Type of dependency
    metadata: Dict[str, any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.from_job_id, self.to_job_id))


class JobGraph:
    """
    Directed Acyclic Graph (DAG) for managing job dependencies and execution order.

    This class provides the core data structure for representing complex job
    workflows with dependencies, enabling sophisticated scheduling algorithms.
    """

    def __init__(self, graph_id: Optional[str] = None):
        self.graph_id = graph_id or str(uuid.uuid4())
        self.nodes: Dict[JobID, JobNode] = {}
        self.edges: Set[JobEdge] = set()

        # Caches for performance
        self._topological_order: Optional[List[JobID]] = None
        self._ready_jobs_cache: Optional[Set[JobID]] = None
        self._is_valid_cache: Optional[bool] = None

        # Metadata
        self.created_at = datetime.utcnow()
        self.metadata: Dict[str, any] = {}

    def add_job(self, job: Job) -> JobNode:
        """
        Add a job to the graph.

        Args:
            job: Job to add

        Returns:
            JobNode representing the added job

        Raises:
            ValidationError: If job already exists
        """
        if job.job_id in self.nodes:
            raise ValidationError(
                "job_id", job.job_id, "unique job ID",
                message=f"Job {job.job_id} already exists in graph"
            )

        node = JobNode(job=job)
        self.nodes[job.job_id] = node

        # Add dependencies from job definition
        for dep_id in job.dependencies:
            if dep_id in self.nodes:
                self.add_dependency(dep_id, job.job_id)
            else:
                # Store pending dependency
                node.add_dependency(dep_id)

        self._invalidate_caches()

        logger.info("Job added to graph",
                   job_id=job.job_id,
                   graph_id=self.graph_id,
                   dependencies=len(job.dependencies))

        return node

    def remove_job(self, job_id: JobID) -> bool:
        """
        Remove a job from the graph.

        Args:
            job_id: ID of job to remove

        Returns:
            True if job was removed, False if not found
        """
        if job_id not in self.nodes:
            return False

        # Remove all edges involving this job
        edges_to_remove = {e for e in self.edges
                          if e.from_job_id == job_id or e.to_job_id == job_id}

        for edge in edges_to_remove:
            self.edges.remove(edge)

            # Update dependency lists
            if edge.from_job_id == job_id and edge.to_job_id in self.nodes:
                self.nodes[edge.to_job_id].remove_dependency(job_id)
            elif edge.to_job_id == job_id and edge.from_job_id in self.nodes:
                self.nodes[edge.from_job_id].remove_dependent(job_id)

        # Remove the node
        del self.nodes[job_id]
        self._invalidate_caches()

        logger.info("Job removed from graph",
                   job_id=job_id,
                   graph_id=self.graph_id)

        return True

    def add_dependency(self, from_job_id: JobID, to_job_id: JobID) -> bool:
        """
        Add a dependency edge between two jobs.

        Args:
            from_job_id: Job that must complete first
            to_job_id: Job that depends on the first job

        Returns:
            True if dependency was added, False if it would create a cycle

        Raises:
            ValidationError: If either job doesn't exist
        """
        if from_job_id not in self.nodes:
            raise ValidationError(
                "from_job_id", from_job_id, "existing job ID",
                message=f"Job {from_job_id} not found in graph"
            )

        if to_job_id not in self.nodes:
            raise ValidationError(
                "to_job_id", to_job_id, "existing job ID",
                message=f"Job {to_job_id} not found in graph"
            )

        # Check for self-dependency
        if from_job_id == to_job_id:
            raise ValidationError(
                "dependency", f"{from_job_id} -> {to_job_id}", "non-self dependency",
                message="Job cannot depend on itself"
            )

        # Check if this would create a cycle
        if self._would_create_cycle(from_job_id, to_job_id):
            logger.warning("Dependency would create cycle",
                          from_job=from_job_id,
                          to_job=to_job_id)
            return False

        # Add the edge
        edge = JobEdge(from_job_id, to_job_id)
        self.edges.add(edge)

        # Update node dependency lists
        self.nodes[from_job_id].add_dependent(to_job_id)
        self.nodes[to_job_id].add_dependency(from_job_id)

        self._invalidate_caches()

        logger.info("Dependency added",
                   from_job=from_job_id,
                   to_job=to_job_id,
                   graph_id=self.graph_id)

        return True

    def remove_dependency(self, from_job_id: JobID, to_job_id: JobID) -> bool:
        """
        Remove a dependency edge between two jobs.

        Args:
            from_job_id: Source job
            to_job_id: Target job

        Returns:
            True if dependency was removed, False if not found
        """
        edge = JobEdge(from_job_id, to_job_id)

        if edge not in self.edges:
            return False

        self.edges.remove(edge)

        # Update node dependency lists
        if from_job_id in self.nodes:
            self.nodes[from_job_id].remove_dependent(to_job_id)

        if to_job_id in self.nodes:
            self.nodes[to_job_id].remove_dependency(from_job_id)

        self._invalidate_caches()

        logger.info("Dependency removed",
                   from_job=from_job_id,
                   to_job=to_job_id)

        return True

    def get_ready_jobs(self) -> List[Job]:
        """
        Get all jobs that are ready to run (no pending dependencies).

        Returns:
            List of jobs ready for execution
        """
        if self._ready_jobs_cache is None:
            ready_job_ids = {job_id for job_id, node in self.nodes.items()
                           if node.is_ready and node.job.status == JobStatus.PENDING}
            self._ready_jobs_cache = ready_job_ids

        return [self.nodes[job_id].job for job_id in self._ready_jobs_cache]

    def get_topological_order(self) -> List[JobID]:
        """
        Get jobs in topological order (dependencies before dependents).

        Returns:
            List of job IDs in topological order

        Raises:
            DependencyError: If graph contains cycles
        """
        if self._topological_order is not None:
            return self._topological_order.copy()

        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for job_id in self.nodes:
            in_degree[job_id] = len(self.nodes[job_id].dependencies)

        # Initialize queue with jobs that have no dependencies
        queue = deque([job_id for job_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            job_id = queue.popleft()
            result.append(job_id)

            # Reduce in-degree for dependent jobs
            for dependent_id in self.nodes[job_id].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        # Check for cycles
        if len(result) != len(self.nodes):
            remaining_jobs = set(self.nodes.keys()) - set(result)
            raise DependencyError(
                "graph", list(remaining_jobs),
                message="Circular dependencies detected in job graph"
            )

        self._topological_order = result
        return result.copy()

    def mark_job_completed(self, job_id: JobID):
        """
        Mark a job as completed and update dependent jobs.

        Args:
            job_id: ID of completed job
        """
        if job_id not in self.nodes:
            return

        node = self.nodes[job_id]
        node.job.status = JobStatus.COMPLETED
        node.completion_time = datetime.utcnow()

        # Remove this job as a dependency from its dependents
        for dependent_id in node.dependents:
            if dependent_id in self.nodes:
                self.nodes[dependent_id].remove_dependency(job_id)

        self._invalidate_caches()

        logger.info("Job marked completed",
                   job_id=job_id,
                   dependents=len(node.dependents))

    def get_critical_path(self) -> Tuple[List[JobID], float]:
        """
        Calculate the critical path through the job graph.

        Returns:
            Tuple of (job_ids_in_critical_path, total_duration)
        """
        # Calculate earliest start times using topological order
        topo_order = self.get_topological_order()
        earliest_start = {}
        earliest_finish = {}

        for job_id in topo_order:
            job = self.nodes[job_id].job
            duration = job.estimated_duration()

            # Calculate earliest start time
            max_predecessor_finish = 0.0
            for dep_id in self.nodes[job_id].dependencies:
                if dep_id in earliest_finish:
                    max_predecessor_finish = max(max_predecessor_finish, earliest_finish[dep_id])

            earliest_start[job_id] = max_predecessor_finish
            earliest_finish[job_id] = earliest_start[job_id] + duration

        # Find the job with the latest finish time
        if not earliest_finish:
            return [], 0.0

        project_duration = max(earliest_finish.values())

        # Trace back the critical path
        critical_path = []
        current_finish_time = project_duration

        # Start from jobs that finish at project completion time
        remaining_jobs = set(job_id for job_id, finish_time in earliest_finish.items()
                           if abs(finish_time - project_duration) < 0.001)

        while remaining_jobs:
            # Find job on critical path
            for job_id in remaining_jobs:
                job_duration = self.nodes[job_id].job.estimated_duration()
                expected_start = current_finish_time - job_duration

                if abs(earliest_start[job_id] - expected_start) < 0.001:
                    critical_path.insert(0, job_id)
                    current_finish_time = earliest_start[job_id]
                    remaining_jobs = self.nodes[job_id].dependencies.copy()
                    break
            else:
                # No more jobs on critical path
                break

        return critical_path, project_duration

    def get_parallel_levels(self) -> List[List[JobID]]:
        """
        Get jobs grouped by parallel execution levels.

        Returns:
            List of levels, where each level contains jobs that can run in parallel
        """
        levels = []
        remaining_jobs = set(self.nodes.keys())

        while remaining_jobs:
            # Find jobs with no dependencies among remaining jobs
            current_level = []
            for job_id in remaining_jobs:
                node = self.nodes[job_id]
                if not (node.dependencies & remaining_jobs):
                    current_level.append(job_id)

            if not current_level:
                # This shouldn't happen if graph is acyclic
                raise DependencyError(
                    "graph", list(remaining_jobs),
                    message="Unable to resolve dependencies - possible cycle"
                )

            levels.append(current_level)
            remaining_jobs -= set(current_level)

        return levels

    def validate(self) -> List[str]:
        """
        Validate the job graph for consistency and correctness.

        Returns:
            List of validation errors (empty if valid)
        """
        if self._is_valid_cache is not None:
            return []

        errors = []

        # Check for missing dependencies
        for job_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    errors.append(f"Job {job_id} depends on missing job {dep_id}")

        # Check for cycles
        try:
            self.get_topological_order()
        except DependencyError as e:
            errors.append(str(e))

        # Check edge consistency
        for edge in self.edges:
            if edge.from_job_id not in self.nodes:
                errors.append(f"Edge references missing job {edge.from_job_id}")
            if edge.to_job_id not in self.nodes:
                errors.append(f"Edge references missing job {edge.to_job_id}")

        # Check node-edge consistency
        for job_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                edge = JobEdge(dep_id, job_id)
                if edge not in self.edges:
                    errors.append(f"Missing edge for dependency {dep_id} -> {job_id}")

        if not errors:
            self._is_valid_cache = True

        return errors

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the job graph."""
        stats = {
            'total_jobs': len(self.nodes),
            'total_edges': len(self.edges),
            'ready_jobs': len(self.get_ready_jobs()),
            'completed_jobs': sum(1 for node in self.nodes.values()
                                if node.job.status == JobStatus.COMPLETED),
            'running_jobs': sum(1 for node in self.nodes.values()
                              if node.job.status == JobStatus.RUNNING),
            'failed_jobs': sum(1 for node in self.nodes.values()
                             if node.job.status == JobStatus.FAILED),
        }

        # Calculate complexity metrics
        if self.nodes:
            dependencies_per_job = [len(node.dependencies) for node in self.nodes.values()]
            dependents_per_job = [len(node.dependents) for node in self.nodes.values()]

            stats.update({
                'avg_dependencies_per_job': sum(dependencies_per_job) / len(dependencies_per_job),
                'max_dependencies_per_job': max(dependencies_per_job),
                'avg_dependents_per_job': sum(dependents_per_job) / len(dependents_per_job),
                'max_dependents_per_job': max(dependents_per_job),
            })

            # Critical path analysis
            try:
                critical_path, duration = self.get_critical_path()
                stats.update({
                    'critical_path_length': len(critical_path),
                    'critical_path_duration': duration,
                })
            except Exception:
                stats.update({
                    'critical_path_length': 0,
                    'critical_path_duration': 0.0,
                })

            # Parallel levels
            try:
                levels = self.get_parallel_levels()
                stats.update({
                    'parallel_levels': len(levels),
                    'max_parallel_jobs': max(len(level) for level in levels) if levels else 0,
                })
            except Exception:
                stats.update({
                    'parallel_levels': 0,
                    'max_parallel_jobs': 0,
                })

        return stats

    def to_dict(self) -> Dict[str, any]:
        """Convert job graph to dictionary representation."""
        return {
            'graph_id': self.graph_id,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'nodes': {
                job_id: {
                    'job': node.job.to_dict(),
                    'dependencies': list(node.dependencies),
                    'dependents': list(node.dependents),
                    'ready_time': node.ready_time.isoformat() if node.ready_time else None,
                    'scheduled_time': node.scheduled_time.isoformat() if node.scheduled_time else None,
                    'start_time': node.start_time.isoformat() if node.start_time else None,
                    'completion_time': node.completion_time.isoformat() if node.completion_time else None,
                }
                for job_id, node in self.nodes.items()
            },
            'edges': [
                {
                    'from_job_id': edge.from_job_id,
                    'to_job_id': edge.to_job_id,
                    'edge_type': edge.edge_type,
                    'metadata': edge.metadata
                }
                for edge in self.edges
            ],
            'statistics': self.get_statistics()
        }

    def _would_create_cycle(self, from_job_id: JobID, to_job_id: JobID) -> bool:
        """Check if adding an edge would create a cycle."""
        # Use DFS to check if there's already a path from to_job_id to from_job_id
        visited = set()
        stack = [to_job_id]

        while stack:
            current = stack.pop()
            if current == from_job_id:
                return True

            if current in visited:
                continue

            visited.add(current)

            if current in self.nodes:
                stack.extend(self.nodes[current].dependents)

        return False

    def _invalidate_caches(self):
        """Invalidate cached computations."""
        self._topological_order = None
        self._ready_jobs_cache = None
        self._is_valid_cache = None

    def __len__(self) -> int:
        """Return number of jobs in graph."""
        return len(self.nodes)

    def __contains__(self, job_id: JobID) -> bool:
        """Check if job is in graph."""
        return job_id in self.nodes

    def __iter__(self) -> Iterator[JobNode]:
        """Iterate over job nodes."""
        return iter(self.nodes.values())
