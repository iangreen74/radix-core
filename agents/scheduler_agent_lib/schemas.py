"""
Pydantic schemas for the scheduler agent API.
Updated to Pydantic v2 syntax for consistency.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Request to score GPU candidates for a job."""
    job_type: str = Field(..., description="Type of job (e.g., 'train-bert')")
    features: Dict[str, Any] = Field(..., description="Job features like gpu_mem_gb, batch_size")
    candidate_gpu_types: List[str] = Field(..., description="Available GPU types to score")
    colocated_job_types: List[str] = Field(default_factory=list, description="Currently running job types")

    model_config = {"extra": "ignore"}


class GPUCandidate(BaseModel):
    """A scored GPU candidate."""
    gpu_type: str = Field(..., description="GPU type (e.g., 'A100-80GB')")
    score: float = Field(..., description="Scheduler score (higher = better)")
    confidence: float = Field(..., description="Confidence in the score [0,1]")
    expected_runtime: Optional[float] = Field(None, description="Expected job completion time")

    model_config = {"extra": "ignore"}


class ScoreResponse(BaseModel):
    """Response with ranked GPU candidates."""
    candidates: List[GPUCandidate] = Field(..., description="GPU candidates ranked by score")
    exploration_ratio: float = Field(..., description="Current exploration vs exploitation ratio")
    model_uncertainty: float = Field(..., description="Overall model uncertainty")

    model_config = {"extra": "ignore"}


class ObserveRequest(BaseModel):
    """Request to update the model with job completion data."""
    job_id: str = Field(..., description="Unique job identifier")
    job_type: str = Field(..., description="Type of job that completed")
    gpu_type: str = Field(..., description="GPU type that ran the job")
    features: Dict[str, Any] = Field(..., description="Job features")
    actual_runtime: float = Field(..., description="Actual job completion time in seconds")
    success: bool = Field(True, description="Whether the job completed successfully")

    model_config = {"extra": "ignore"}


class ObserveResponse(BaseModel):
    """Response after updating the model."""
    updated: bool = Field(..., description="Whether the model was updated")
    new_uncertainty: float = Field(..., description="Updated model uncertainty")

    model_config = {"extra": "ignore"}
