"""
Integration utilities for the scheduler agent library.
Provides clean interfaces for external modules to use.
"""

from typing import Dict, List, Any, Optional
from .schemas import ScoreRequest, ScoreResponse, GPUCandidate
from .scoring import ScoringService
from .model import SchedulerModel, JobFeatures
from .config import SchedulerConfig


# Global service instance for local integration
_local_service: Optional[ScoringService] = None


def get_local_service() -> ScoringService:
    """Get or create the local scoring service."""
    global _local_service
    if _local_service is None:
        config = SchedulerConfig()
        model = SchedulerModel(config)
        _local_service = ScoringService(model)
    return _local_service


def score_job_local(
    job_type: str,
    features: Dict[str, Any],
    candidate_gpu_types: List[str],
    colocated_job_types: List[str] = None
) -> Dict[str, Any]:
    """
    Score a job locally without HTTP overhead.

    Args:
        job_type: Type of job (e.g., 'train-bert')
        features: Job features dict (gpu_mem_gb, batch_size, etc.)
        candidate_gpu_types: List of available GPU types
        colocated_job_types: Currently running job types

    Returns:
        Dict with ranked GPU candidates and scoring metadata
    """
    if colocated_job_types is None:
        colocated_job_types = []

    service = get_local_service()

    request = ScoreRequest(
        job_type=job_type,
        features=features,
        candidate_gpu_types=candidate_gpu_types,
        colocated_job_types=colocated_job_types
    )

    response = service.score_job(request)

    # Convert to dict for compatibility
    return {
        "candidates": [
            {
                "gpu_type": c.gpu_type,
                "score": c.score,
                "confidence": c.confidence,
                "expected_runtime": c.expected_runtime
            }
            for c in response.candidates
        ],
        "exploration_ratio": response.exploration_ratio,
        "model_uncertainty": response.model_uncertainty
    }


def observe_local(
    job_id: str,
    job_type: str,
    gpu_type: str,
    features: Dict[str, Any],
    actual_runtime: float,
    success: bool = True
) -> Dict[str, Any]:
    """
    Update the model with observed job completion data.

    Args:
        job_id: Unique job identifier
        job_type: Type of job that completed
        gpu_type: GPU type that ran the job
        features: Job features dict
        actual_runtime: Actual completion time in seconds
        success: Whether the job completed successfully

    Returns:
        Dict with update status and new uncertainty
    """
    service = get_local_service()

    from .schemas import ObserveRequest
    request = ObserveRequest(
        job_id=job_id,
        job_type=job_type,
        gpu_type=gpu_type,
        features=features,
        actual_runtime=actual_runtime,
        success=success
    )

    service.observe_runtime(request)

    return {
        "updated": True,
        "new_uncertainty": 0.3  # Default value
    }


def reset_local_state():
    """Reset the local service state (useful for testing)."""
    global _local_service
    _local_service = None


class LocalScoringService:
    """
    Local scoring service for direct integration.
    Provides the same interface as the original integration.py.
    """

    def __init__(self, config=None, sqlite_path=":memory:"):
        if config is None:
            config = SchedulerConfig()
        self.model = SchedulerModel(config, sqlite_path)
        self.service = ScoringService(self.model)

    def score_local(self,
                   job_type: str,
                   features: Dict[str, Any],
                   candidate_gpu_types: List[str],
                   colocated_job_types: List[str] = None) -> Dict[str, Any]:
        """Score a job locally."""
        return score_job_local(job_type, features, candidate_gpu_types, colocated_job_types)

    def observe_local(self,
                     job_id: str,
                     job_type: str,
                     gpu_type: str,
                     features: Dict[str, Any],
                     actual_runtime: float,
                     success: bool = True) -> Dict[str, Any]:
        """Observe job completion."""
        return observe_local(job_id, job_type, gpu_type, features, actual_runtime, success)

    def reset_local_state(self):
        """Reset state."""
        # Create new model instance
        config = SchedulerConfig()
        self.model = SchedulerModel(config, ":memory:")
        self.service = ScoringService(self.model)
