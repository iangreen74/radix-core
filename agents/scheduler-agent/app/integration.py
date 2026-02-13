"""
Integration helpers for scheduler-agent internal scoring.

This module provides direct access to the scoring logic without HTTP overhead,
primarily for unit testing and internal integration scenarios.
"""

from typing import Dict, List, Any, Optional
from .scoring import ScoringService, ScoreRequest
from .model import SchedulerModel
from .config import get_config


class LocalScoringService:
    """Local scoring service for direct integration without HTTP."""

    def __init__(self, config=None, sqlite_path=":memory:"):
        """Initialize local scoring service."""
        self.config = config or get_config()
        self.model = SchedulerModel(self.config, sqlite_path)
        self.scoring_service = ScoringService(self.model)

    def score_local(self,
                   job_type: str,
                   features: Dict[str, Any],
                   candidate_gpu_types: List[str],
                   colocated_job_types: List[str] = None) -> Dict[str, Any]:
        """
        Score a job locally without HTTP overhead.

        Args:
            job_type: Type of job (e.g., "training", "inference")
            features: Job features dictionary
            candidate_gpu_types: List of available GPU types
            colocated_job_types: List of job types on target nodes

        Returns:
            Scoring response dictionary matching /v1/score API format
        """
        # Create scoring request
        request = ScoreRequest(
            job_type=job_type,
            features=features,
            candidate_gpu_types=candidate_gpu_types,
            colocated_job_types=colocated_job_types or []
        )

        # Score using internal service
        response = self.scoring_service.score_job(request)

        # Convert to dictionary format
        return response.dict() if hasattr(response, 'dict') else response

    def observe_local(self,
                     job_type: str,
                     gpu_type: str,
                     runtime_seconds: float,
                     success: bool = True,
                     metadata: Dict[str, Any] = None) -> None:
        """
        Record a runtime observation locally.

        Args:
            job_type: Type of job that completed
            gpu_type: GPU type used
            runtime_seconds: Actual runtime in seconds
            success: Whether the job completed successfully
            metadata: Additional metadata about the execution
        """
        from .scoring import ObserveRequest

        observe_request = ObserveRequest(
            job_type=job_type,
            gpu_type=gpu_type,
            runtime_seconds=runtime_seconds,
            success=success,
            metadata=metadata or {}
        )

        self.scoring_service.observe_runtime(observe_request)

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get current model metrics."""
        return self.model.get_metrics()

    def reset_model(self) -> None:
        """Reset model state (useful for testing)."""
        self.model = SchedulerModel(self.config, ":memory:")
        self.scoring_service = ScoringService(self.model)


# Global instance for easy access
_local_service: Optional[LocalScoringService] = None


def get_local_scoring_service(reset: bool = False) -> LocalScoringService:
    """
    Get or create a local scoring service instance.

    Args:
        reset: If True, create a new instance with fresh state

    Returns:
        LocalScoringService instance
    """
    global _local_service

    if _local_service is None or reset:
        _local_service = LocalScoringService()

    return _local_service


def score_job_local(job_type: str,
                   features: Dict[str, Any],
                   candidate_gpu_types: List[str],
                   colocated_job_types: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function for local job scoring.

    This is the main entry point for integration testing and
    scenarios where HTTP overhead should be avoided.
    """
    service = get_local_scoring_service()
    return service.score_local(job_type, features, candidate_gpu_types, colocated_job_types)


def observe_runtime_local(job_type: str,
                         gpu_type: str,
                         runtime_seconds: float,
                         success: bool = True,
                         metadata: Dict[str, Any] = None) -> None:
    """
    Convenience function for local runtime observation.
    """
    service = get_local_scoring_service()
    service.observe_local(job_type, gpu_type, runtime_seconds, success, metadata)


def reset_local_state() -> None:
    """
    Reset local scoring service state.

    Useful for test isolation and benchmarking scenarios.
    """
    get_local_scoring_service(reset=True)


# Integration utilities for testing
class MockScoringService:
    """Mock scoring service for testing scenarios."""

    def __init__(self, deterministic_scores: Dict[str, float] = None):
        """
        Initialize mock service.

        Args:
            deterministic_scores: Fixed scores per GPU type for testing
        """
        self.deterministic_scores = deterministic_scores or {}
        self.call_count = 0
        self.last_request = None

    def score_local(self, job_type: str, features: Dict[str, Any],
                   candidate_gpu_types: List[str],
                   colocated_job_types: List[str] = None) -> Dict[str, Any]:
        """Mock scoring that returns deterministic results."""
        self.call_count += 1
        self.last_request = {
            "job_type": job_type,
            "features": features,
            "candidate_gpu_types": candidate_gpu_types,
            "colocated_job_types": colocated_job_types
        }

        # Return highest scoring GPU or first candidate
        if self.deterministic_scores:
            best_gpu = max(candidate_gpu_types,
                          key=lambda gpu: self.deterministic_scores.get(gpu, 0))
            score = self.deterministic_scores.get(best_gpu, 50.0)
        else:
            best_gpu = candidate_gpu_types[0] if candidate_gpu_types else "A100-80GB"
            score = 75.0

        return {
            "priority_score": score,
            "gpu_selector": {
                "nodeSelector": {
                    "gpu.nvidia.com/class": best_gpu
                }
            },
            "terms": {
                "chosen_gpu": best_gpu,
                "mu": score / 100.0,
                "sigma": 0.1,
                "information_gain": 0.05,
                "interference": 0.0
            },
            "avoid_co_locate_with": [],
            "reasoning": f"Mock scoring selected {best_gpu}"
        }

    def observe_local(self, *args, **kwargs):
        """Mock observation (no-op)."""
        pass

    def get_call_count(self) -> int:
        """Get number of scoring calls made."""
        return self.call_count

    def get_last_request(self) -> Optional[Dict[str, Any]]:
        """Get the last scoring request."""
        return self.last_request
