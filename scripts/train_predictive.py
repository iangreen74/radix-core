#!/usr/bin/env python3
"""
Train ONNX predictive models for job runtime and energy consumption.
Pure local training with deterministic RandomForest models.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ONNX conversion imports
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def verify_onnx_parity(preprocessor, sklearn_model, onnx_model, sample_features, tolerance=1e-6):
    """Verify ONNX model produces similar results to sklearn model."""
    try:
        import onnxruntime as ort
        
        # Get sklearn predictions
        X_processed = preprocessor.transform(sample_features)
        sklearn_pred = sklearn_model.predict(X_processed)
        
        # Get ONNX predictions
        session = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        onnx_pred = session.run(None, {input_name: X_processed.astype(np.float32)})[0]
        
        # Check MAE
        mae = np.mean(np.abs(sklearn_pred - onnx_pred.flatten()))
        return mae < tolerance
        
    except Exception as e:
        print(f"Parity check failed: {e}")
        return False


def load_feature_spec(spec_path: Path) -> dict:
    """Load feature specification from engine."""
    with open(spec_path, 'r') as f:
        return json.load(f)
def generate_synthetic_traces(n_jobs=200, seed=1337):
    """Generate synthetic training traces if no file provided."""
    np.random.seed(seed)
    
    model_families = ['bert-large', 'gpt-3', 'resnet-50', 'llama-7b', 'stable-diffusion']
    
    data = []
    for i in range(n_jobs):
        model_family = np.random.choice(model_families)
        
        # Generate correlated features
        if model_family in ['bert-large', 'gpt-3', 'llama-7b']:
            # NLP models
            seq_len = np.random.randint(128, 2048)
            batch_size = np.random.randint(8, 64)
            gpu_mem_gb = np.random.choice([40, 80])
            flops_est = np.random.uniform(1.0, 5.0)
        else:
            # Vision models
            seq_len = np.random.randint(224, 512)
            batch_size = np.random.randint(16, 128) 
            gpu_mem_gb = np.random.choice([24, 40])
            flops_est = np.random.uniform(0.5, 3.0)
        
        # Base runtime influenced by model complexity
        base_runtime = (
            batch_size * seq_len * flops_est * 50 +
            np.random.normal(0, 10000)
        )
        runtime_ms = max(10000, int(base_runtime))
        
        # Energy correlated with runtime and GPU memory
        energy_j = runtime_ms * gpu_mem_gb * 0.8 + np.random.normal(0, 50000)
        energy_j = max(10000, energy_j)
        
        data.append({
            'job_id': f'job_{i:04d}',
            'est_input_mb': np.random.uniform(100, 2000),
            'cpu_req': np.random.randint(2, 16),
            'mem_req_gb': np.random.uniform(8, 128),
            'gpu_req': 1,
            'gpu_mem_gb': gpu_mem_gb,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'flops_est_tflops': flops_est,
            'prev_runtime_ms': int(runtime_ms * np.random.uniform(0.8, 1.2)),
            'model_family': model_family,
            'label_runtime_ms': runtime_ms,
            'label_energy_j': energy_j
        })
    
    return pd.DataFrame(data)


def build_preprocessor(df, feature_spec):
    """Build preprocessing pipeline."""
    features = feature_spec['features']
    
    # Identify numeric and categorical features
    numeric_features = []
    categorical_features = []
    
    for feature in features:
        if feature_spec['dtypes'][feature] == 'object':
            categorical_features.append(feature)
        else:
            numeric_features.append(feature)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def train_models(df, feature_spec, seed=1337):
    """Train RandomForest models for runtime and energy prediction."""
    features = feature_spec['features']
    targets = feature_spec['targets']
    
    # Map column names from CSV to feature spec
    column_mapping = {
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
    
    # Create feature DataFrame with proper mapping
    X = pd.DataFrame(index=df.index)
    for feature_name in features:
        if feature_name in column_mapping:
            source_col = column_mapping[feature_name]
            if source_col in df.columns:
                X[feature_name] = df[source_col]
            else:
                # Use defaults for missing columns
                if feature_name == 'est_input_mb':
                    X[feature_name] = 512.0
                elif feature_name == 'cpu_req':
                    X[feature_name] = 4
                elif feature_name == 'mem_req_gb':
                    X[feature_name] = 16.0
                elif feature_name == 'gpu_req':
                    X[feature_name] = 1
                elif feature_name == 'gpu_mem_gb':
                    X[feature_name] = 40.0
                elif feature_name == 'batch_size':
                    X[feature_name] = 32
                elif feature_name == 'seq_len':
                    X[feature_name] = 512
                elif feature_name == 'flops_est_tflops':
                    X[feature_name] = 2.5
                elif feature_name == 'prev_runtime_ms':
                    X[feature_name] = 300000
                elif feature_name == 'model_family':
                    X[feature_name] = 'bert-large'
        else:
            # Default for unmapped features
            X[feature_name] = 0.0 if feature_spec['dtypes'][feature_name] != 'object' else 'unknown'
    
    # Create synthetic targets if not present
    if targets[0] in df.columns:
        y_runtime = df[targets[0]]
    else:
        # Generate from runtime_estimate
        y_runtime = df['runtime_estimate'] * 1000  # Convert to ms
        
    if targets[1] in df.columns:
        y_energy = df[targets[1]]
    else:
        # Generate synthetic energy based on runtime and GPU memory
        y_energy = y_runtime * X['gpu_mem_gb'] * 0.8
    
    # Build preprocessor
    preprocessor = build_preprocessor(df, feature_spec)
    
    # Fit preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Train models
    runtime_model = RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        max_depth=10
    )
    
    energy_model = RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        max_depth=10
    )
    
    # Split data for validation
    X_train, X_val, y_runtime_train, y_runtime_val, y_energy_train, y_energy_val = train_test_split(
        X_processed, y_runtime, y_energy, test_size=0.2, random_state=seed
    )
    
    # Fit models
    runtime_model.fit(X_train, y_runtime_train)
    energy_model.fit(X_train, y_energy_train)
    
    # Validate models
    runtime_pred = runtime_model.predict(X_val)
    energy_pred = energy_model.predict(X_val)
    
    # Calculate metrics
    metrics = {
        'runtime': {
            'mae': float(mean_absolute_error(y_runtime_val, runtime_pred)),
            'mse': float(mean_squared_error(y_runtime_val, runtime_pred)),
            'r2': float(r2_score(y_runtime_val, runtime_pred))
        },
        'energy': {
            'mae': float(mean_absolute_error(y_energy_val, energy_pred)),
            'mse': float(mean_squared_error(y_energy_val, energy_pred)),
            'r2': float(r2_score(y_energy_val, energy_pred))
        }
    }
    
    return preprocessor, runtime_model, energy_model, metrics


def convert_to_onnx(preprocessor, model, feature_spec, model_name):
    """Convert sklearn model to ONNX format."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX conversion requires: pip install skl2onnx onnxruntime")
    
    features = feature_spec['features']
    
    # Create initial types for ONNX conversion
    initial_types = []
    for i, feature in enumerate(features):
        if feature_spec['dtypes'][feature] == 'object':
            initial_types.append((f'input_{i}', StringTensorType([None, 1])))
        else:
            initial_types.append((f'input_{i}', FloatTensorType([None, 1])))
    
    # For simplicity, convert the full pipeline (preprocessor + model)
    from sklearn.pipeline import Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Convert to ONNX with simplified input (all float)
    # Note: This is a simplified approach - in production you'd handle mixed types properly
    initial_type = [('float_input', FloatTensorType([None, len(features)]))]
    
    try:
        onnx_model = convert_sklearn(
            full_pipeline,
            initial_types=initial_type,
            target_opset=11
        )
        return onnx_model
    except Exception as e:
        print(f"Warning: ONNX conversion failed for {model_name}: {e}")
        return None


def verify_onnx_parity(preprocessor, sklearn_model, onnx_model, test_data, tolerance=1e-6):
    """Verify ONNX model produces same results as sklearn."""
    if onnx_model is None:
        return False
    
    # Prepare test data
    X_processed = preprocessor.transform(test_data)
    
    # Sklearn prediction
    sklearn_pred = sklearn_model.predict(X_processed)
    
    # ONNX prediction
    try:
        ort_session = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = ort_session.get_inputs()[0].name
        onnx_pred = ort_session.run(None, {input_name: X_processed.astype(np.float32)})[0]
        
        # Check parity
        mae_diff = mean_absolute_error(sklearn_pred, onnx_pred.flatten())
        return mae_diff < tolerance
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def update_run_log(artifacts_dir, metrics, duration_s):
    """Update docs/RUN_LOG.json with training results."""
    docs_dir = Path(__file__).parent.parent / "docs"
    run_log_path = docs_dir / "RUN_LOG.json"
    
    # Load existing log or create new
    if run_log_path.exists():
        with open(run_log_path, 'r') as f:
            run_log = json.load(f)
    else:
        run_log = {}
    
    # Add predictive training entry
    if 'predictive_training' not in run_log:
        run_log['predictive_training'] = []
    
    entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'artifacts_dir': str(artifacts_dir),
        'duration_seconds': duration_s,
        'metrics': metrics,
        'status': 'completed'
    }
    
    run_log['predictive_training'].append(entry)
    
    # Write back
    with open(run_log_path, 'w') as f:
        json.dump(run_log, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train ONNX predictive models')
    parser.add_argument('--traces', type=str, help='Path to training traces CSV')
    parser.add_argument('--out', type=str, default='artifacts/predictive', help='Output directory')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    
    args = parser.parse_args()
    
    start_time = datetime.utcnow()
    
    # Set deterministic environment
    np.random.seed(args.seed)
    
    # Load feature specification
    spec_path = Path(__file__).parent.parent / "engine" / "radix_core" / "features_v1.json"
    feature_spec = load_feature_spec(spec_path)
    print(f"Loaded feature spec: {len(feature_spec['features'])} features")
    
    # Load or generate training data
    if args.traces and Path(args.traces).exists():
        print(f"Loading traces from {args.traces}")
        df = pd.read_csv(args.traces)
    else:
        print("Generating synthetic training traces")
        df = generate_synthetic_traces(seed=args.seed)
    
    print(f"Training data: {len(df)} jobs")
    
    # Create output directory with timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path(args.out + f"__{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    print("Training models...")
    preprocessor, runtime_model, energy_model, metrics = train_models(df, feature_spec, args.seed)
    
    # Save preprocessor
    preprocessor_path = out_dir / "preprocess.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor: {preprocessor_path}")
    
    # Save sklearn models
    runtime_sklearn_path = out_dir / "runtime_sklearn.joblib"
    energy_sklearn_path = out_dir / "energy_sklearn.joblib"
    joblib.dump(runtime_model, runtime_sklearn_path)
    joblib.dump(energy_model, energy_sklearn_path)
    print(f"Saved sklearn models: {runtime_sklearn_path}, {energy_sklearn_path}")
    
    # Save feature spec
    spec_path = out_dir / "feature_spec.json"
    with open(spec_path, 'w') as f:
        json.dump(feature_spec, f, indent=2)
    
    # Convert to ONNX if available
    if ONNX_AVAILABLE:
        print("Converting to ONNX...")
        
        # Create sample for ONNX conversion (32 rows)
        sample_features = df[feature_spec['features']].head(32)
        
        try:
            # Convert preprocessor + models to ONNX
            from sklearn.pipeline import Pipeline
            
            # Create pipelines for ONNX conversion
            runtime_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', runtime_model)
            ])
            
            energy_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', energy_model)
            ])
            
            # Convert to ONNX with proper input types
            from skl2onnx.common.data_types import FloatTensorType
            
            # Determine input shape (number of features after preprocessing)
            X_sample_processed = preprocessor.transform(sample_features)
            n_features = X_sample_processed.shape[1]
            
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert runtime model
            runtime_onnx = convert_sklearn(
                runtime_pipeline,
                initial_types=initial_type,
                target_opset=11
            )
            
            # Convert energy model
            energy_onnx = convert_sklearn(
                energy_pipeline,
                initial_types=initial_type,
                target_opset=11
            )
            
            # Save ONNX models
            runtime_onnx_path = out_dir / "runtime.onnx"
            energy_onnx_path = out_dir / "energy.onnx"
            
            with open(runtime_onnx_path, 'wb') as f:
                f.write(runtime_onnx.SerializeToString())
            
            with open(energy_onnx_path, 'wb') as f:
                f.write(energy_onnx.SerializeToString())
            
            print(f"Exported ONNX models: {runtime_onnx_path}, {energy_onnx_path}")
            
            # Verify ONNX parity
            print("Verifying ONNX parity...")
            parity_check = verify_onnx_parity(
                preprocessor, runtime_model, runtime_onnx, sample_features
            )
            print(f"ONNX parity check: {'✅ PASS' if parity_check else '❌ FAIL'}")
            
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
            print("Falling back to sklearn-only export")
    else:
        print("ONNX dependencies not available, sklearn-only export")
    
    # Save training metrics
    metrics_path = out_dir / "train_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Calculate duration
    duration_s = (datetime.utcnow() - start_time).total_seconds()
    
    # Update run log
    update_run_log(out_dir, metrics, duration_s)
    
    print(f"\nTraining completed in {duration_s:.1f}s")
    print(f"Artifacts saved to: {out_dir}")
    print(f"Runtime R²: {metrics['runtime']['r2']:.3f}")
    print(f"Energy R²: {metrics['energy']['r2']:.3f}")


if __name__ == '__main__':
    main()
