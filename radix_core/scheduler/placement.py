"""
Job Placement Components for Radix Scheduler

Implements placement strategies for distributing jobs across available resources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..types import Job, ResourceRequirements
from ..logging import get_logger


@dataclass
class ResourceNode:
    """Represents a compute resource node."""
    node_id: str
    available_cpu: float
    available_memory: float  # in MB
    available_gpu: int
    total_cpu: float
    total_memory: float  # in MB
    total_gpu: int
    current_load: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobPlacement:
    """Represents the placement of a job on a resource node."""
    job: Job
    node_id: str
    allocated_cpu: float
    allocated_memory: float
    allocated_gpu: int
    placement_score: float = 0.0
    placement_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacementPlan:
    """Complete placement plan for a set of jobs."""
    plan_id: str
    placements: List[JobPlacement]
    unplaceable_jobs: List[Job] = field(default_factory=list)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    plan_metadata: Dict[str, Any] = field(default_factory=dict)


class PlacementStrategy(ABC):
    """Abstract base class for placement strategies."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"radix.scheduler.placement.{name}")

    @abstractmethod
    def place_jobs(self, jobs: List[Job],
                   nodes: List[ResourceNode] = None) -> PlacementPlan:
        """Place jobs on available resource nodes."""
        pass

    def can_place_job(self, job: Job, node: ResourceNode) -> bool:
        """Check if a job can be placed on a given node."""
        return (node.available_cpu >= job.requirements.cpu_cores and
                node.available_memory >= job.requirements.memory_mb and
                node.available_gpu >= job.requirements.gpu_count)

    def calculate_placement_score(self, job: Job, node: ResourceNode) -> float:
        """Calculate a score for placing a job on a node."""
        # Higher score is better
        # Consider resource utilization efficiency
        cpu_util = job.requirements.cpu_cores / max(node.available_cpu, 1.0)
        memory_util = job.requirements.memory_mb / max(node.available_memory, 1.0)
        gpu_util = job.requirements.gpu_count / max(node.available_gpu, 1.0) if node.available_gpu > 0 else 0.0

        # Prefer balanced utilization
        max_util = max(cpu_util, memory_util, gpu_util)
        balance_score = 1.0 - abs(cpu_util - memory_util) - abs(cpu_util - gpu_util)

        # Consider current load
        load_score = 1.0 - node.current_load

        return balance_score * 0.6 + load_score * 0.4 - max_util * 0.1


class LocalPlacement(PlacementStrategy):
    """Simple local placement strategy (single node)."""

    def __init__(self):
        super().__init__("local")
        # Create a default local node
        self.local_node = ResourceNode(
            node_id="local",
            available_cpu=4.0,  # Default 4 cores
            available_memory=8192.0,  # Default 8GB
            available_gpu=0,  # No GPU by default (safety)
            total_cpu=4.0,
            total_memory=8192.0,
            total_gpu=0
        )

    def place_jobs(self, jobs: List[Job],
                   nodes: List[ResourceNode] = None) -> PlacementPlan:
        """Place jobs on local node."""
        if nodes is None:
            nodes = [self.local_node]

        if not nodes:
            return PlacementPlan(
                plan_id=f"local_placement_{id(self)}",
                placements=[],
                unplaceable_jobs=jobs.copy()
            )

        # Use first available node (local placement)
        target_node = nodes[0]
        placements = []
        unplaceable_jobs = []

        # Track allocated resources
        allocated_cpu = 0.0
        allocated_memory = 0.0
        allocated_gpu = 0

        for job in jobs:
            # Check if job can fit on remaining resources
            remaining_cpu = target_node.available_cpu - allocated_cpu
            remaining_memory = target_node.available_memory - allocated_memory
            remaining_gpu = target_node.available_gpu - allocated_gpu

            if (job.requirements.cpu_cores <= remaining_cpu and
                job.requirements.memory_mb <= remaining_memory and
                job.requirements.gpu_count <= remaining_gpu):

                # Place the job
                placement = JobPlacement(
                    job=job,
                    node_id=target_node.node_id,
                    allocated_cpu=job.requirements.cpu_cores,
                    allocated_memory=job.requirements.memory_mb,
                    allocated_gpu=job.requirements.gpu_count,
                    placement_score=1.0  # Simple scoring for local placement
                )
                placements.append(placement)

                # Update allocated resources
                allocated_cpu += job.requirements.cpu_cores
                allocated_memory += job.requirements.memory_mb
                allocated_gpu += job.requirements.gpu_count

            else:
                # Job cannot be placed
                unplaceable_jobs.append(job)

        plan = PlacementPlan(
            plan_id=f"local_placement_{id(self)}",
            placements=placements,
            unplaceable_jobs=unplaceable_jobs,
            resource_utilization={
                "cpu_utilization": allocated_cpu / target_node.total_cpu,
                "memory_utilization": allocated_memory / target_node.total_memory,
                "gpu_utilization": allocated_gpu / max(target_node.total_gpu, 1)
            },
            plan_metadata={
                "strategy": "local",
                "target_node": target_node.node_id,
                "placed_jobs": len(placements),
                "unplaced_jobs": len(unplaceable_jobs)
            }
        )

        self.logger.info(f"Local placement completed",
                        placed_jobs=len(placements),
                        unplaced_jobs=len(unplaceable_jobs),
                        cpu_utilization=plan.resource_utilization["cpu_utilization"])

        return plan


class LoadBalancedPlacement(PlacementStrategy):
    """Load-balanced placement strategy across multiple nodes."""

    def __init__(self):
        super().__init__("load_balanced")

    def place_jobs(self, jobs: List[Job],
                   nodes: List[ResourceNode] = None) -> PlacementPlan:
        """Place jobs using load balancing across nodes."""
        if not nodes:
            # Create default nodes for load balancing
            nodes = [
                ResourceNode(
                    node_id=f"node_{i}",
                    available_cpu=4.0,
                    available_memory=8192.0,
                    available_gpu=0,
                    total_cpu=4.0,
                    total_memory=8192.0,
                    total_gpu=0
                )
                for i in range(3)  # Default 3 nodes
            ]

        placements = []
        unplaceable_jobs = []

        # Track allocated resources per node
        node_allocations = {
            node.node_id: {
                "cpu": 0.0,
                "memory": 0.0,
                "gpu": 0,
                "load": node.current_load
            }
            for node in nodes
        }

        # Sort jobs by resource requirements (largest first)
        sorted_jobs = sorted(jobs, key=lambda j: (
            j.resources.cpu_cores + j.resources.memory_gb + j.resources.gpu_count
        ), reverse=True)

        for job in sorted_jobs:
            best_node = None
            best_score = -float('inf')

            # Find the best node for this job
            for node in nodes:
                alloc = node_allocations[node.node_id]

                # Check if job can fit
                remaining_cpu = node.available_cpu - alloc["cpu"]
                remaining_memory = node.available_memory - alloc["memory"]
                remaining_gpu = node.available_gpu - alloc["gpu"]

                if (job.requirements.cpu_cores <= remaining_cpu and
                    job.requirements.memory_mb <= remaining_memory and
                    job.requirements.gpu_count <= remaining_gpu):

                    # Calculate placement score
                    score = self.calculate_placement_score(job, node)

                    # Adjust score based on current allocation
                    load_penalty = alloc["load"] * 0.5
                    adjusted_score = score - load_penalty

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_node = node

            if best_node:
                # Place the job
                placement = JobPlacement(
                    job=job,
                    node_id=best_node.node_id,
                    allocated_cpu=job.requirements.cpu_cores,
                    allocated_memory=job.requirements.memory_mb,
                    allocated_gpu=job.requirements.gpu_count,
                    placement_score=best_score
                )
                placements.append(placement)

                # Update allocations
                alloc = node_allocations[best_node.node_id]
                alloc["cpu"] += job.requirements.cpu_cores
                alloc["memory"] += job.requirements.memory_mb
                alloc["gpu"] += job.requirements.gpu_count
                alloc["load"] += 0.1  # Increase load

            else:
                # Job cannot be placed anywhere
                unplaceable_jobs.append(job)

        # Calculate overall resource utilization
        total_cpu_used = sum(alloc["cpu"] for alloc in node_allocations.values())
        total_memory_used = sum(alloc["memory"] for alloc in node_allocations.values())
        total_gpu_used = sum(alloc["gpu"] for alloc in node_allocations.values())

        total_cpu_available = sum(node.total_cpu for node in nodes)
        total_memory_available = sum(node.total_memory for node in nodes)
        total_gpu_available = sum(node.total_gpu for node in nodes)

        plan = PlacementPlan(
            plan_id=f"load_balanced_placement_{id(self)}",
            placements=placements,
            unplaceable_jobs=unplaceable_jobs,
            resource_utilization={
                "cpu_utilization": total_cpu_used / max(total_cpu_available, 1),
                "memory_utilization": total_memory_used / max(total_memory_available, 1),
                "gpu_utilization": total_gpu_used / max(total_gpu_available, 1)
            },
            plan_metadata={
                "strategy": "load_balanced",
                "node_count": len(nodes),
                "placed_jobs": len(placements),
                "unplaced_jobs": len(unplaceable_jobs),
                "node_allocations": node_allocations
            }
        )

        self.logger.info(f"Load-balanced placement completed",
                        placed_jobs=len(placements),
                        unplaced_jobs=len(unplaceable_jobs),
                        node_count=len(nodes),
                        cpu_utilization=plan.resource_utilization["cpu_utilization"])

        return plan


# Default placement factory
def get_placement_strategy(strategy_type: str = "local") -> PlacementStrategy:
    """Get a placement strategy instance by type."""
    if strategy_type == "local":
        return LocalPlacement()
    elif strategy_type == "load_balanced":
        return LoadBalancedPlacement()
    else:
        raise ValueError(f"Unknown placement strategy: {strategy_type}")


# Alias for backwards compatibility
ResourcePlacer = LocalPlacement
