"""
Simplified predictive model integration for radix-bench.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)


class SimplePredictor:
    """Simplified predictor using sklearn models."""
    
    def __init__(self, artifacts_dir: Path):
        """Initialize predictor with artifacts directory."""
        self.artifacts_dir = Path(artifacts_dir)
        
        # Load feature specification
        feature_spec_path = self.artifacts_dir / "feature_spec.json"
        if not feature_spec_path.exists():
            raise FileNotFoundError(f"Feature spec not found: {feature_spec_path}")
        
        with open(feature_spec_path, 'r') as f:
            self.feature_spec = json.load(f)
        
        self.expected_features = self.feature_spec["features"]
        
        # Load preprocessing pipeline
        preprocess_path = self.artifacts_dir / "preprocess.joblib"
        if not preprocess_path.exists():
            raise FileNotFoundError(f"Preprocessing pipeline not found: {preprocess_path}")
        
        self.preprocessor = joblib.load(preprocess_path)
        
        # Load sklearn models
        runtime_sklearn_path = self.artifacts_dir / "runtime_sklearn.joblib"
        energy_sklearn_path = self.artifacts_dir / "energy_sklearn.joblib"
        
        if not runtime_sklearn_path.exists():
            raise FileNotFoundError(f"Runtime sklearn model not found: {runtime_sklearn_path}")
        if not energy_sklearn_path.exists():
            raise FileNotFoundError(f"Energy sklearn model not found: {energy_sklearn_path}")
        
        self.runtime_model = joblib.load(runtime_sklearn_path)
        self.energy_model = joblib.load(energy_sklearn_path)
        self.backend = "sklearn"
        
        logger.info(f"SimplePredictor initialized from {artifacts_dir} (backend: sklearn)")
    
    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Predict runtime and energy for jobs."""
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
        
        # Run sklearn inference
        try:
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


def pack_job_features(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Pack job DataFrame into features format for prediction."""
    # Simple feature mapping for jobs
    result = pd.DataFrame(index=jobs_df.index)
    
    # Map job attributes to features
    result['est_input_mb'] = jobs_df.get('est_input_mb', 512.0)
    result['cpu_req'] = jobs_df.get('cpu_cores', 4)
    result['mem_req_gb'] = jobs_df.get('mem_gb', jobs_df.get('memory_gb', 16.0))
    result['gpu_req'] = 1  # Assume single GPU jobs
    result['gpu_mem_gb'] = jobs_df.get('gpu_mem_gb', jobs_df.get('memory_gb', 40.0))
    result['batch_size'] = jobs_df.get('batch_size', 32)
    result['seq_len'] = jobs_df.get('seq_len', 512)
    result['flops_est_tflops'] = jobs_df.get('flops_est_tflops', 2.5)
    result['prev_runtime_ms'] = jobs_df.get('prev_runtime_ms', jobs_df.get('runtime_estimate', 300) * 1000)
    result['model_family'] = jobs_df.get('model_family', 'bert-large')
    
    return result
