"""
Integration layer between engine scheduler and scheduler-agent.
Provides client interface for the engine to request GPU rankings.
Includes ONNX predictive model integration for runtime and energy prediction.
"""

import asyncio
import random
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import httpx
import pandas as pd
import numpy as np

# ONNX runtime import with fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Joblib import with fallback
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Import the scheduler agent library for local scoring
try:
    from agents.scheduler_agent_lib.integration import score_job_local
    LOCAL_SCORING_AVAILABLE = True
except ImportError:
    LOCAL_SCORING_AVAILABLE = False

from ..logging import get_logger

logger = get_logger(__name__)


class OnnxPredictor:
    """ONNX-based predictor for job runtime and energy consumption."""
    
    def __init__(self, artifacts_dir: Path):
        """Initialize ONNX predictor with artifacts directory.
        
        Args:
            artifacts_dir: Path to directory containing preprocess.joblib, 
                          runtime.onnx, energy.onnx, and feature_spec.json
        """
        self.artifacts_dir = Path(artifacts_dir)
        
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not available. Install with: pip install onnxruntime")
        
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib not available. Install with: pip install joblib")
        
        # Load feature specification
        feature_spec_path = self.artifacts_dir / "feature_spec.json"
        if not feature_spec_path.exists():
            # Fallback to engine location
            feature_spec_path = Path(__file__).parent.parent / "radix_core" / "features_v1.json"
        
        with open(feature_spec_path, 'r') as f:
            self.feature_spec = json.load(f)
        
        self.expected_features = self.feature_spec["features"]
        
        # Load preprocessing pipeline
        preprocess_path = self.artifacts_dir / "preprocess.joblib"
        if not preprocess_path.exists():
            raise FileNotFoundError(f"Preprocessing pipeline not found: {preprocess_path}")
        
        self.preprocessor = joblib.load(preprocess_path)
        
        # ONNX-first loading strategy
        runtime_onnx_path = self.artifacts_dir / "runtime.onnx"
        energy_onnx_path = self.artifacts_dir / "energy.onnx"
        runtime_sklearn_path = self.artifacts_dir / "runtime_sklearn.joblib"
        energy_sklearn_path = self.artifacts_dir / "energy_sklearn.joblib"
        
        self.backend = None
        self.use_onnx = False
        self.runtime_session = None
        self.energy_session = None
        self.runtime_model = None
        self.energy_model = None
        
        # Try ONNX first (preferred)
        if runtime_onnx_path.exists() and energy_onnx_path.exists() and ONNX_AVAILABLE:
            try:
                # Create ONNX sessions with CPU provider only (deterministic)
                self.runtime_session = ort.InferenceSession(
                    str(runtime_onnx_path), 
                    providers=['CPUExecutionProvider']
                )
                self.energy_session = ort.InferenceSession(
                    str(energy_onnx_path), 
                    providers=['CPUExecutionProvider']
                )
                self.use_onnx = True
                self.backend = "onnx"
                logger.info("✅ Using ONNX models for prediction (preferred)")
            except Exception as e:
                logger.warning(f"ONNX loading failed: {e}, falling back to sklearn")
        
        # Fallback to sklearn models
        if not self.use_onnx:
            if not runtime_sklearn_path.exists():
                raise FileNotFoundError(f"Runtime sklearn model not found: {runtime_sklearn_path}")
            if not energy_sklearn_path.exists():
                raise FileNotFoundError(f"Energy sklearn model not found: {energy_sklearn_path}")
            
            self.runtime_model = joblib.load(runtime_sklearn_path)
            self.energy_model = joblib.load(energy_sklearn_path)
            self.backend = "sklearn"
            logger.info("⚠️  Using sklearn models for prediction (fallback)")
        
        logger.info(f"ONNX predictor initialized from {artifacts_dir}")
    
    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Predict runtime and energy for jobs.
        
        Args:
            df_features: DataFrame with job features matching feature_spec
            
        Returns:
            DataFrame with columns 'runtime_ms_pred' and 'energy_j_pred'
            aligned to input DataFrame index
        """
        if df_features.empty:
            return pd.DataFrame(
                columns=['runtime_ms_pred', 'energy_j_pred'],
                index=df_features.index
            )
        
        # Validate feature order and presence
        missing_features = set(self.expected_features) - set(df_features.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and reorder features to match expected order
        feature_df = df_features[self.expected_features].copy()
        
        # Preprocess features
        try:
            X_processed = self.preprocessor.transform(feature_df)
        except Exception as e:
            raise RuntimeError(f"Feature preprocessing failed: {e}")
        
        # Run inference (ONNX or sklearn)
        try:
            if self.use_onnx:
                # Convert to float32 for ONNX
                X_onnx = X_processed.astype(np.float32)
                
                runtime_input = {self.runtime_session.get_inputs()[0].name: X_onnx}
                runtime_pred = self.runtime_session.run(None, runtime_input)[0]
                
                energy_input = {self.energy_session.get_inputs()[0].name: X_onnx}
                energy_pred = self.energy_session.run(None, energy_input)[0]
            else:
                # Use sklearn models
                runtime_pred = self.runtime_model.predict(X_processed)
                energy_pred = self.energy_model.predict(X_processed)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")
        
        # Create result DataFrame
        result = pd.DataFrame({
            'runtime_ms_pred': runtime_pred.flatten(),
            'energy_j_pred': energy_pred.flatten()
        }, index=df_features.index)
        
        return result


def pack_features(df_jobs: pd.DataFrame) -> pd.DataFrame:
    """Pack job DataFrame into features format for prediction.
    
    Args:
        df_jobs: DataFrame with job information
        
    Returns:
        DataFrame with features in correct format and dtypes
    """
    # Load feature spec for dtypes
    feature_spec_path = Path(__file__).parent.parent / "radix_core" / "features_v1.json"
    with open(feature_spec_path, 'r') as f:
        feature_spec = json.load(f)
    
    expected_features = feature_spec["features"]
    expected_dtypes = feature_spec["dtypes"]
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=df_jobs.index)
    
    # Map and convert features
    feature_mapping = {
        'est_input_mb': 'est_input_mb',
        'cpu_req': 'cpu_cores', 
        'mem_req_gb': 'mem_gb',
        'gpu_req': 'gpu_count',
        'gpu_mem_gb': 'gpu_mem_gb',
        'batch_size': 'batch_size',
        'seq_len': 'seq_len', 
        'flops_est_tflops': 'flops_est_tflops',
        'prev_runtime_ms': 'prev_runtime_ms',
        'model_family': 'model_family'
    }
    
    for feature_name in expected_features:
        if feature_name in feature_mapping:
            source_col = feature_mapping[feature_name]
            if source_col in df_jobs.columns:
                result[feature_name] = df_jobs[source_col]
            else:
                # Provide sensible defaults
                if feature_name == 'est_input_mb':
                    result[feature_name] = 512.0
                elif feature_name == 'cpu_req':
                    result[feature_name] = 4
                elif feature_name == 'mem_req_gb':
                    result[feature_name] = 16.0
                elif feature_name == 'gpu_req':
                    result[feature_name] = 1
                elif feature_name == 'gpu_mem_gb':
                    result[feature_name] = 40.0
                elif feature_name == 'batch_size':
                    result[feature_name] = 32
                elif feature_name == 'seq_len':
                    result[feature_name] = 512
                elif feature_name == 'flops_est_tflops':
                    result[feature_name] = 2.5
                elif feature_name == 'prev_runtime_ms':
                    result[feature_name] = 300000
                elif feature_name == 'model_family':
                    result[feature_name] = 'bert-large'
        else:
            # Default values for missing mappings
            result[feature_name] = 0.0 if expected_dtypes[feature_name] != 'object' else 'unknown'
    
    # Convert dtypes
    for feature_name, dtype in expected_dtypes.items():
        if feature_name in result.columns:
            if dtype == 'object':
                result[feature_name] = result[feature_name].astype(str)
            elif dtype == 'int64':
                result[feature_name] = pd.to_numeric(result[feature_name], errors='coerce').fillna(0).astype('int64')
            elif dtype == 'float64':
                result[feature_name] = pd.to_numeric(result[feature_name], errors='coerce').fillna(0.0).astype('float64')
    
    return result


@dataclass
class GPUCandidate:
    """Represents a GPU candidate with scoring information."""
    gpu_type: str
    score: float
    terms: Dict[str, Any]
    reasoning: str = ""


@dataclass
class ScoringRequest:
    """Request for GPU scoring."""
    job_type: str
    features: Dict[str, Any]
    candidate_gpu_types: List[str]
    colocated_job_types: List[str] = None


class InfoScoringClient:
    """Client for requesting GPU rankings from scheduler-agent."""

    def __init__(self,
                 scheduler_url: str = "http://localhost:8080",
                 timeout_seconds: float = 5.0,
                 max_retries: int = 3,
                 backoff_factor: float = 1.5,
                 random_seed: Optional[int] = None):
        self.scheduler_url = scheduler_url.rstrip('/')
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Set random seed for deterministic tie-breaking
        if random_seed is not None:
            random.seed(random_seed)

        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _validate_request(self, request: ScoringRequest) -> None:
        """Validate scoring request format."""
        if not request.job_type:
            raise ValueError("job_type cannot be empty")

        if not request.candidate_gpu_types:
            raise ValueError("candidate_gpu_types cannot be empty")

        if not isinstance(request.features, dict):
            raise ValueError("features must be a dictionary")

    async def _call_scorer_with_retry(self, request: ScoringRequest) -> Optional[Dict[str, Any]]:
        """Call the scorer service with retry logic."""
        request_data = {
            "job_type": request.job_type,
            "features": request.features,
            "candidate_gpu_types": request.candidate_gpu_types,
            "colocated_job_types": request.colocated_job_types or []
        }

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.scheduler_url}/v1/score",
                    json=request_data
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Scorer returned {response.status_code}: {response.text}")

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    backoff_time = self.backoff_factor ** attempt
                    logger.warning(f"Scorer call failed (attempt {attempt + 1}), retrying in {backoff_time}s: {e}")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Scorer call failed after {self.max_retries} attempts: {e}")

        return None

    def _create_fallback_ranking(self, request: ScoringRequest) -> List[GPUCandidate]:
        """Create fallback ranking when scorer is unavailable."""
        logger.info("Using fallback ranking (random shuffle)")

        # Shuffle candidates for fallback
        candidates = request.candidate_gpu_types.copy()
        random.shuffle(candidates)

        return [
            GPUCandidate(
                gpu_type=gpu_type,
                score=50.0 + random.uniform(-10, 10),  # Random score around 50
                terms={"fallback": True, "gpu_type": gpu_type},
                reasoning="Fallback ranking (scorer unavailable)"
            )
            for gpu_type in candidates
        ]

    def _parse_scorer_response(self, response: Dict[str, Any],
                              request: ScoringRequest) -> List[GPUCandidate]:
        """Parse scorer response into ranked candidates."""
        try:
            # Extract primary recommendation
            primary_gpu = response.get("gpu_selector", {}).get("nodeSelector", {}).get("gpu.nvidia.com/class")
            priority_score = response.get("priority_score", 50.0)
            terms = response.get("terms", {})

            # Create ranking with primary choice first
            candidates = []

            if primary_gpu and primary_gpu in request.candidate_gpu_types:
                candidates.append(GPUCandidate(
                    gpu_type=primary_gpu,
                    score=priority_score,
                    terms=terms,
                    reasoning=f"Primary choice (score: {priority_score:.2f})"
                ))

            # Add remaining candidates with decreasing scores
            remaining = [gpu for gpu in request.candidate_gpu_types if gpu != primary_gpu]
            for i, gpu_type in enumerate(remaining):
                # Decrease score for non-primary choices
                adjusted_score = max(0, priority_score - (i + 1) * 10)
                candidates.append(GPUCandidate(
                    gpu_type=gpu_type,
                    score=adjusted_score,
                    terms={"gpu_type": gpu_type, "rank": i + 2},
                    reasoning=f"Alternative choice (rank: {i + 2})"
                ))

            return candidates

        except Exception as e:
            logger.error(f"Error parsing scorer response: {e}")
            return self._create_fallback_ranking(request)

    async def rank_candidates(self,
                            job_meta: Dict[str, Any],
                            candidate_gpu_types: List[str],
                            colocated_job_types: List[str] = None) -> List[GPUCandidate]:
        """
        Rank GPU candidates for a job using information-theoretic scoring.

        Args:
            job_meta: Job metadata including type and features
            candidate_gpu_types: List of available GPU types
            colocated_job_types: List of job types already on target nodes

        Returns:
            List of GPUCandidate objects sorted by score (highest first)
        """
        # Extract job information
        job_type = job_meta.get("job_type", "unknown")
        features = job_meta.get("features", {})

        # Create scoring request
        request = ScoringRequest(
            job_type=job_type,
            features=features,
            candidate_gpu_types=candidate_gpu_types,
            colocated_job_types=colocated_job_types or []
        )

        # Validate request
        try:
            self._validate_request(request)
        except ValueError as e:
            logger.error(f"Invalid scoring request: {e}")
            return self._create_fallback_ranking(request)

        # Call scorer service
        response = await self._call_scorer_with_retry(request)

        if response is None:
            return self._create_fallback_ranking(request)

        # Parse and return ranked candidates
        candidates = self._parse_scorer_response(response, request)

        # Sort by score (highest first) with deterministic tie-breaking
        candidates.sort(key=lambda c: (-c.score, c.gpu_type))

        logger.info(f"Ranked {len(candidates)} GPU candidates for job {job_type}")
        return candidates


def create_scoring_client(scheduler_url: str = None, **kwargs) -> InfoScoringClient:
    """Factory function to create a scoring client with default configuration."""
    if scheduler_url is None:
        # Try to detect if running in Kubernetes
        import os
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            scheduler_url = "http://scheduler-agent:8080"
        else:
            scheduler_url = "http://localhost:8080"

    return InfoScoringClient(scheduler_url=scheduler_url, **kwargs)


# Synchronous wrapper for backward compatibility
class SyncInfoScoringClient:
    """Synchronous wrapper for InfoScoringClient."""

    def __init__(self, **kwargs):
        self.async_client = InfoScoringClient(**kwargs)

    def rank_candidates(self, job_meta: Dict[str, Any],
                       candidate_gpu_types: List[str],
                       colocated_job_types: List[str] = None) -> List[GPUCandidate]:
        """Synchronous version of rank_candidates."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.async_client.rank_candidates(job_meta, candidate_gpu_types, colocated_job_types)
        )

    def close(self):
        """Close the underlying async client."""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.async_client.close())
        except RuntimeError:
            pass
