"""Scoring logic for information-theoretic GPU scheduler."""

from typing import Dict, List, Optional, Any
from .schemas import ScoreRequest, ScoreResponse, ObserveRequest, GPUCandidate
from .model import JobFeatures, SchedulerModel, ScoringResult


class ScoringService:
    """Service for handling scoring and observation requests."""

    def __init__(self, model: SchedulerModel):
        self.model = model

    def extract_features(self, job_type: str, features_dict: Dict[str, Any]) -> JobFeatures:
        """Extract and validate job features from request."""
        return JobFeatures(
            job_type=job_type,
            gpu_mem_gb=float(features_dict.get("gpu_mem_gb", 0.0)),
            model_params_m=float(features_dict.get("model_params_m", 0.0)),
            batch_size=int(features_dict.get("batch_size", 1)),
            tenant=str(features_dict.get("tenant", "default"))
        )

    def score_job(self, request: ScoreRequest) -> ScoreResponse:
        """Score a job for GPU assignment."""
        # Extract features
        features = self.extract_features(request.job_type, request.features)

        # Validate candidates
        if not request.candidate_gpu_types:
            raise ValueError("No candidate GPU types provided")

        # Score the job
        result = self.model.score_job(
            features=features,
            candidates=request.candidate_gpu_types,
            colocated_types=request.colocated_job_types
        )

        # Convert to new response format
        candidates = []
        for gpu_type in request.candidate_gpu_types:
            score = result.gpu_selector.get(gpu_type, 0.0)
            candidates.append(GPUCandidate(
                gpu_type=gpu_type,
                score=score,
                confidence=0.8,  # Default confidence
                expected_runtime=None
            ))

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        return ScoreResponse(
            candidates=candidates,
            exploration_ratio=0.2,  # Default exploration ratio
            model_uncertainty=0.3   # Default uncertainty
        )

    def observe_runtime(self, request: ObserveRequest):
        """Record an observed runtime for model learning."""
        self.model.observe(
            job_type=request.job_type,
            gpu_type=request.gpu_type,
            runtime=request.actual_runtime,
            colocated_types=[]  # Not used in current model
        )

    def get_model_metrics(self) -> Dict[str, float]:
        """Get current model metrics."""
        return self.model.get_metrics()


def normalize_priority_score(cost: float, all_costs: List[float]) -> float:
    """Normalize cost to priority score [0, 100].

    Lower cost -> higher priority score.
    """
    if len(all_costs) <= 1:
        return 50.0

    max_cost = max(all_costs)
    min_cost = min(all_costs)

    if max_cost == min_cost:
        return 50.0

    # Invert and normalize to [0, 100]
    normalized = 100.0 * (max_cost - cost) / (max_cost - min_cost)
    return max(0.0, min(100.0, normalized))


def validate_gpu_types(gpu_types: List[str]) -> List[str]:
    """Validate and normalize GPU type names."""
    valid_types = []
    for gpu_type in gpu_types:
        # Basic validation - could be extended with a registry
        if gpu_type and isinstance(gpu_type, str):
            # Normalize common variations
            normalized = gpu_type.strip().upper()
            if normalized not in valid_types:
                valid_types.append(normalized)

    return valid_types if valid_types else ["A100-80GB"]  # Fallback


def extract_job_metadata(annotations: Dict[str, str]) -> Dict[str, Any]:
    """Extract job metadata from Kubernetes annotations."""
    metadata = {}

    # Standard annotations
    if "app.kubernetes.io/job-type" in annotations:
        metadata["job_type"] = annotations["app.kubernetes.io/job-type"]

    if "gpu.mem.gi" in annotations:
        try:
            metadata["gpu_mem_gb"] = float(annotations["gpu.mem.gi"])
        except ValueError:
            pass

    if "ml.batch_size" in annotations:
        try:
            metadata["batch_size"] = int(annotations["ml.batch_size"])
        except ValueError:
            pass

    # Custom extraction logic
    for key, value in annotations.items():
        if key.startswith("scheduler.radix.ai/"):
            clean_key = key.replace("scheduler.radix.ai/", "")
            metadata[clean_key] = value

    return metadata
