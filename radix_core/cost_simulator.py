"""
Cost Simulation for Radix

This module provides cost estimation and simulation capabilities for research
purposes. In dry-run mode, all costs are $0.00 to prevent accidental spending.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import get_config
from .errors import CostCapExceededError, SafetyViolationError
from .types import Job


@dataclass
class CostEstimate:
    """Detailed cost estimate for an operation."""

    operation_type: str
    estimated_cost_usd: float
    breakdown: Dict[str, float]
    duration_hours: float
    timestamp: datetime

    # Resource usage details
    cpu_core_hours: float = 0.0
    memory_gb_hours: float = 0.0
    gpu_hours: float = 0.0
    storage_gb_hours: float = 0.0
    network_gb: float = 0.0

    # Safety metadata
    is_simulated: bool = True
    dry_run_mode: bool = True

    def __post_init__(self):
        """Validate cost estimate."""
        config = get_config()

        # In dry-run mode, all costs must be $0.00
        if config.safety.dry_run and self.estimated_cost_usd != 0.0:
            raise SafetyViolationError(
                f"Cost estimate must be $0.00 in dry-run mode, got ${self.estimated_cost_usd:.2f}",
                "Ensure dry-run mode is properly configured",
            )

        # Validate breakdown sums to total
        breakdown_total = sum(self.breakdown.values())
        if abs(breakdown_total - self.estimated_cost_usd) > 0.01:
            raise ValueError(
                f"Cost breakdown ({breakdown_total:.2f}) doesn't match total ({self.estimated_cost_usd:.2f})"
            )


class CostSimulator:
    """
    Simulates costs for various operations without actual resource usage.

    In dry-run mode, all costs are $0.00. This class provides realistic
    cost modeling for research purposes while maintaining safety.
    """

    # Simulated pricing (used for research modeling only)
    SIMULATED_PRICING = {
        "cpu_core_hour": 0.05,  # $0.05 per CPU core hour
        "memory_gb_hour": 0.01,  # $0.01 per GB memory hour
        "gpu_hour": 0.50,  # $0.50 per GPU hour
        "storage_gb_hour": 0.0001,  # $0.0001 per GB storage hour
        "network_gb": 0.10,  # $0.10 per GB network transfer
    }

    def __init__(self):
        self.config = get_config()
        self.estimates_cache: Dict[str, CostEstimate] = {}

    def estimate_job_cost(self, job: Job) -> CostEstimate:
        """
        Estimate cost for executing a single job.

        Args:
            job: Job to estimate cost for

        Returns:
            Cost estimate (always $0.00 in dry-run mode)
        """
        # Generate cache key
        cache_key = f"job_{job.job_id}_{hash(str(job.requirements))}"

        if cache_key in self.estimates_cache:
            return self.estimates_cache[cache_key]

        # Estimate duration
        duration_hours = job.estimated_duration() / 3600.0

        # Calculate resource usage
        cpu_core_hours = job.requirements.cpu_cores * duration_hours
        memory_gb_hours = (job.requirements.memory_mb / 1024.0) * duration_hours
        gpu_hours = job.requirements.gpu_count * duration_hours
        storage_gb_hours = (job.requirements.storage_mb / 1024.0) * duration_hours
        network_gb = 0.1  # Minimal network usage for local operations

        # Calculate costs (will be $0.00 in dry-run mode)
        breakdown = self._calculate_cost_breakdown(
            cpu_core_hours, memory_gb_hours, gpu_hours, storage_gb_hours, network_gb
        )

        total_cost = sum(breakdown.values())

        # In dry-run mode, override all costs to $0.00
        if self.config.safety.dry_run:
            total_cost = 0.0
            breakdown = {k: 0.0 for k in breakdown.keys()}

        estimate = CostEstimate(
            operation_type=f"job_execution_{job.name}",
            estimated_cost_usd=total_cost,
            breakdown=breakdown,
            duration_hours=duration_hours,
            timestamp=datetime.utcnow(),
            cpu_core_hours=cpu_core_hours,
            memory_gb_hours=memory_gb_hours,
            gpu_hours=gpu_hours,
            storage_gb_hours=storage_gb_hours,
            network_gb=network_gb,
            is_simulated=True,
            dry_run_mode=self.config.safety.dry_run,
        )

        self.estimates_cache[cache_key] = estimate
        return estimate

    def estimate_batch_cost(self, batch_size: int, processing_time_per_item: float = 0.1) -> float:
        """
        Estimate cost for batch processing.

        Args:
            batch_size: Number of items in batch
            processing_time_per_item: Processing time per item in seconds

        Returns:
            Estimated cost (always $0.00 in dry-run mode)
        """
        if self.config.safety.dry_run:
            return 0.0

        # Simulate batch processing cost calculation
        total_time_hours = (batch_size * processing_time_per_item) / 3600.0
        cpu_cost = total_time_hours * self.SIMULATED_PRICING["cpu_core_hour"]
        memory_cost = (
            total_time_hours * 0.5 * self.SIMULATED_PRICING["memory_gb_hour"]
        )  # 0.5 GB average

        return cpu_cost + memory_cost

    def estimate_schedule_cost(self, jobs: List[Job]) -> CostEstimate:
        """
        Estimate cost for executing a schedule of jobs.

        Args:
            jobs: List of jobs to execute

        Returns:
            Combined cost estimate (always $0.00 in dry-run mode)
        """
        if not jobs:
            return CostEstimate(
                operation_type="empty_schedule",
                estimated_cost_usd=0.0,
                breakdown={},
                duration_hours=0.0,
                timestamp=datetime.utcnow(),
            )

        # Estimate individual job costs
        job_estimates = [self.estimate_job_cost(job) for job in jobs]

        # Combine estimates
        total_cost = sum(est.estimated_cost_usd for est in job_estimates)
        total_duration = max(est.duration_hours for est in job_estimates)  # Parallel execution

        # Combine breakdowns
        combined_breakdown: Dict[str, float] = {}
        for estimate in job_estimates:
            for category, cost in estimate.breakdown.items():
                combined_breakdown[category] = combined_breakdown.get(category, 0.0) + cost

        # Sum resource usage
        total_cpu_hours = sum(est.cpu_core_hours for est in job_estimates)
        total_memory_hours = sum(est.memory_gb_hours for est in job_estimates)
        total_gpu_hours = sum(est.gpu_hours for est in job_estimates)
        total_storage_hours = sum(est.storage_gb_hours for est in job_estimates)
        total_network_gb = sum(est.network_gb for est in job_estimates)

        return CostEstimate(
            operation_type=f"schedule_execution_{len(jobs)}_jobs",
            estimated_cost_usd=total_cost,
            breakdown=combined_breakdown,
            duration_hours=total_duration,
            timestamp=datetime.utcnow(),
            cpu_core_hours=total_cpu_hours,
            memory_gb_hours=total_memory_hours,
            gpu_hours=total_gpu_hours,
            storage_gb_hours=total_storage_hours,
            network_gb=total_network_gb,
            is_simulated=True,
            dry_run_mode=self.config.safety.dry_run,
        )

    def estimate_swarm_cost(self, node_count: int, duration_hours: float) -> CostEstimate:
        """
        Estimate cost for running a swarm simulation.

        Args:
            node_count: Number of nodes in swarm
            duration_hours: Duration of swarm operation

        Returns:
            Cost estimate (always $0.00 in dry-run mode)
        """
        # Simulate swarm node costs
        cpu_cores_per_node = 2.0  # Average 2 cores per node
        memory_gb_per_node = 4.0  # Average 4 GB per node

        total_cpu_hours = node_count * cpu_cores_per_node * duration_hours
        total_memory_hours = node_count * memory_gb_per_node * duration_hours

        breakdown = self._calculate_cost_breakdown(
            total_cpu_hours, total_memory_hours, 0.0, 0.0, 0.1 * node_count
        )

        total_cost = sum(breakdown.values())

        # Override to $0.00 in dry-run mode
        if self.config.safety.dry_run:
            total_cost = 0.0
            breakdown = {k: 0.0 for k in breakdown.keys()}

        return CostEstimate(
            operation_type=f"swarm_simulation_{node_count}_nodes",
            estimated_cost_usd=total_cost,
            breakdown=breakdown,
            duration_hours=duration_hours,
            timestamp=datetime.utcnow(),
            cpu_core_hours=total_cpu_hours,
            memory_gb_hours=total_memory_hours,
            gpu_hours=0.0,
            storage_gb_hours=0.0,
            network_gb=0.1 * node_count,
            is_simulated=True,
            dry_run_mode=self.config.safety.dry_run,
        )

    def check_cost_cap(self, estimated_cost: float, operation: str = "operation"):
        """
        Check if estimated cost exceeds configured caps.

        Args:
            estimated_cost: Estimated cost in USD
            operation: Description of operation

        Raises:
            CostCapExceededError: If cost exceeds caps
        """
        # Check against global cost cap
        if estimated_cost > self.config.safety.cost_cap_usd:
            raise CostCapExceededError(estimated_cost, self.config.safety.cost_cap_usd, operation)

        # Check against per-job cost cap
        if estimated_cost > self.config.safety.max_job_cost_usd:
            raise CostCapExceededError(
                estimated_cost, self.config.safety.max_job_cost_usd, operation
            )

    def get_cost_summary(
        self, start_time: datetime, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get summary of all cost estimates within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range (default: now)

        Returns:
            Cost summary statistics
        """
        if end_time is None:
            end_time = datetime.utcnow()

        # Filter estimates by time range
        relevant_estimates = [
            est for est in self.estimates_cache.values() if start_time <= est.timestamp <= end_time
        ]

        if not relevant_estimates:
            return {
                "total_operations": 0,
                "total_cost_usd": 0.0,
                "total_cpu_hours": 0.0,
                "total_memory_gb_hours": 0.0,
                "total_gpu_hours": 0.0,
                "avg_cost_per_operation": 0.0,
                "cost_breakdown": {},
                "dry_run_mode": self.config.safety.dry_run,
            }

        # Calculate summary statistics
        total_cost = sum(est.estimated_cost_usd for est in relevant_estimates)
        total_cpu_hours = sum(est.cpu_core_hours for est in relevant_estimates)
        total_memory_hours = sum(est.memory_gb_hours for est in relevant_estimates)
        total_gpu_hours = sum(est.gpu_hours for est in relevant_estimates)

        # Combine cost breakdowns
        combined_breakdown: Dict[str, float] = {}
        for estimate in relevant_estimates:
            for category, cost in estimate.breakdown.items():
                combined_breakdown[category] = combined_breakdown.get(category, 0.0) + cost

        return {
            "total_operations": len(relevant_estimates),
            "total_cost_usd": total_cost,
            "total_cpu_hours": total_cpu_hours,
            "total_memory_gb_hours": total_memory_hours,
            "total_gpu_hours": total_gpu_hours,
            "avg_cost_per_operation": total_cost / len(relevant_estimates),
            "cost_breakdown": combined_breakdown,
            "dry_run_mode": self.config.safety.dry_run,
            "time_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
        }

    def _calculate_cost_breakdown(
        self,
        cpu_core_hours: float,
        memory_gb_hours: float,
        gpu_hours: float,
        storage_gb_hours: float,
        network_gb: float,
    ) -> Dict[str, float]:
        """Calculate detailed cost breakdown."""
        breakdown = {
            "cpu": cpu_core_hours * self.SIMULATED_PRICING["cpu_core_hour"],
            "memory": memory_gb_hours * self.SIMULATED_PRICING["memory_gb_hour"],
            "gpu": gpu_hours * self.SIMULATED_PRICING["gpu_hour"],
            "storage": storage_gb_hours * self.SIMULATED_PRICING["storage_gb_hour"],
            "network": network_gb * self.SIMULATED_PRICING["network_gb"],
        }

        return breakdown

    def clear_cache(self):
        """Clear the cost estimates cache."""
        self.estimates_cache.clear()

    def export_estimates(self, filepath: str):
        """Export cost estimates to JSON file for analysis."""
        import json

        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run_mode": self.config.safety.dry_run,
            "estimates": [],
        }

        for estimate in self.estimates_cache.values():
            export_data["estimates"].append(
                {
                    "operation_type": estimate.operation_type,
                    "estimated_cost_usd": estimate.estimated_cost_usd,
                    "breakdown": estimate.breakdown,
                    "duration_hours": estimate.duration_hours,
                    "timestamp": estimate.timestamp.isoformat(),
                    "cpu_core_hours": estimate.cpu_core_hours,
                    "memory_gb_hours": estimate.memory_gb_hours,
                    "gpu_hours": estimate.gpu_hours,
                    "storage_gb_hours": estimate.storage_gb_hours,
                    "network_gb": estimate.network_gb,
                    "is_simulated": estimate.is_simulated,
                    "dry_run_mode": estimate.dry_run_mode,
                }
            )

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)


# Global cost simulator instance
_global_cost_simulator: Optional[CostSimulator] = None


def get_cost_simulator() -> CostSimulator:
    """Get the global cost simulator instance."""
    global _global_cost_simulator
    if _global_cost_simulator is None:
        _global_cost_simulator = CostSimulator()
    return _global_cost_simulator


def reset_cost_simulator():
    """Reset the global cost simulator."""
    global _global_cost_simulator
    _global_cost_simulator = None
