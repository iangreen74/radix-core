"""Cost model for GPU scheduling benchmarks."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GPUType(Enum):
    """GPU types with associated costs."""
    V100_16GB = "v100_16gb"
    V100_32GB = "v100_32gb"
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100_80GB = "h100_80gb"


@dataclass
class GPUCostConfig:
    """GPU cost configuration."""
    gpu_type: GPUType
    hourly_rate_usd: float
    memory_gb: float
    compute_capability: str


# Default GPU cost rates (approximate AWS/GCP pricing)
DEFAULT_GPU_COSTS = {
    GPUType.V100_16GB: GPUCostConfig(GPUType.V100_16GB, 2.48, 16, "7.0"),
    GPUType.V100_32GB: GPUCostConfig(GPUType.V100_32GB, 3.06, 32, "7.0"),
    GPUType.A100_40GB: GPUCostConfig(GPUType.A100_40GB, 4.10, 40, "8.0"),
    GPUType.A100_80GB: GPUCostConfig(GPUType.A100_80GB, 5.12, 80, "8.0"),
    GPUType.H100_80GB: GPUCostConfig(GPUType.H100_80GB, 8.00, 80, "9.0"),
}


class CostModel:
    """
    Cost model for GPU scheduling analysis.

    Tracks costs, budget constraints, and cost optimization metrics.
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}

        self.gpu_costs = config.get('gpu_costs', DEFAULT_GPU_COSTS)
        self.budget_limit_usd = config.get('budget_limit_usd', 1000.0)
        self.cost_cap_usd = config.get('cost_cap_usd', 0.00)  # Safety: $0 by default

        # Cost tracking
        self.total_cost_usd = 0.0
        self.cost_by_gpu: Dict[str, float] = {}
        self.cost_by_job: Dict[str, float] = {}
        self.budget_overruns = 0

        logger.info(f"Cost model initialized: budget=${self.budget_limit_usd}, cap=${self.cost_cap_usd}")

    def get_gpu_hourly_rate(self, gpu_memory_gb: float, compute_capability: str = "8.0") -> float:
        """Get hourly rate for GPU based on memory and compute capability."""
        # Find closest matching GPU type
        best_match = None
        min_memory_diff = float('inf')

        for gpu_type, config in self.gpu_costs.items():
            memory_diff = abs(config.memory_gb - gpu_memory_gb)
            if memory_diff < min_memory_diff:
                min_memory_diff = memory_diff
                best_match = config

        if best_match:
            return best_match.hourly_rate_usd

        # Fallback: linear scaling from A100-80GB
        base_rate = self.gpu_costs[GPUType.A100_80GB].hourly_rate_usd
        base_memory = self.gpu_costs[GPUType.A100_80GB].memory_gb
        return base_rate * (gpu_memory_gb / base_memory)

    def calculate_job_cost(self, gpu_memory_gb: float, runtime_hours: float,
                          compute_capability: str = "8.0") -> float:
        """Calculate cost for a job given GPU specs and runtime."""
        hourly_rate = self.get_gpu_hourly_rate(gpu_memory_gb, compute_capability)
        cost = hourly_rate * runtime_hours

        # Apply cost cap (safety mechanism)
        if cost > self.cost_cap_usd:
            logger.warning(f"Job cost ${cost:.4f} exceeds cap ${self.cost_cap_usd}, capping")
            cost = self.cost_cap_usd

        return cost

    def record_job_cost(self, job_id: str, gpu_id: str, gpu_memory_gb: float,
                       runtime_hours: float, compute_capability: str = "8.0") -> float:
        """Record cost for a completed job."""
        cost = self.calculate_job_cost(gpu_memory_gb, runtime_hours, compute_capability)

        # Update tracking
        self.total_cost_usd += cost
        self.cost_by_job[job_id] = cost

        if gpu_id not in self.cost_by_gpu:
            self.cost_by_gpu[gpu_id] = 0.0
        self.cost_by_gpu[gpu_id] += cost

        # Check budget overrun
        if self.total_cost_usd > self.budget_limit_usd:
            self.budget_overruns += 1
            logger.warning(f"Budget overrun: ${self.total_cost_usd:.2f} > ${self.budget_limit_usd}")

        logger.debug(f"Job {job_id} cost: ${cost:.4f} (total: ${self.total_cost_usd:.2f})")
        return cost

    def get_cost_metrics(self) -> Dict[str, float]:
        """Get comprehensive cost metrics."""
        metrics = {
            "total_cost_usd": self.total_cost_usd,
            "budget_limit_usd": self.budget_limit_usd,
            "budget_utilization": self.total_cost_usd / self.budget_limit_usd if self.budget_limit_usd > 0 else 0.0,
            "budget_overruns": self.budget_overruns,
            "avg_job_cost_usd": 0.0,
            "max_job_cost_usd": 0.0,
            "min_job_cost_usd": 0.0,
            "cost_per_gpu_avg": 0.0,
            "total_jobs": len(self.cost_by_job)
        }

        if self.cost_by_job:
            job_costs = list(self.cost_by_job.values())
            metrics["avg_job_cost_usd"] = sum(job_costs) / len(job_costs)
            metrics["max_job_cost_usd"] = max(job_costs)
            metrics["min_job_cost_usd"] = min(job_costs)

        if self.cost_by_gpu:
            gpu_costs = list(self.cost_by_gpu.values())
            metrics["cost_per_gpu_avg"] = sum(gpu_costs) / len(gpu_costs)

        return metrics

    def is_within_budget(self, additional_cost: float = 0.0) -> bool:
        """Check if additional cost would exceed budget."""
        return (self.total_cost_usd + additional_cost) <= self.budget_limit_usd

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0.0, self.budget_limit_usd - self.total_cost_usd)

    def reset(self):
        """Reset cost tracking."""
        self.total_cost_usd = 0.0
        self.cost_by_gpu.clear()
        self.cost_by_job.clear()
        self.budget_overruns = 0
        logger.info("Cost model reset")

    def estimate_scheduler_efficiency(self, total_runtime_hours: float,
                                    total_jobs: int) -> Dict[str, float]:
        """Estimate scheduler cost efficiency metrics."""
        if total_jobs == 0 or total_runtime_hours == 0:
            return {
                "cost_per_job": 0.0,
                "cost_per_hour": 0.0,
                "efficiency_score": 0.0
            }

        cost_per_job = self.total_cost_usd / total_jobs
        cost_per_hour = self.total_cost_usd / total_runtime_hours

        # Efficiency score: inverse of cost per job (higher is better)
        efficiency_score = 1.0 / (cost_per_job + 1e-6)

        return {
            "cost_per_job": cost_per_job,
            "cost_per_hour": cost_per_hour,
            "efficiency_score": efficiency_score
        }
