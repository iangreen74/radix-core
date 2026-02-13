#!/usr/bin/env python3
"""
Predictor v1 Baseline Model Trainer

Trains a baseline gradient boosting model to predict throughput
from context (workload, instance_type) and action (batch_size).

This serves as the foundation for replacing the free-energy proxy
controller with a learned generative model.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for training.
    
    Args:
        df: Raw DataFrame from RIM-1 observations
    
    Returns:
        DataFrame with engineered features
    """
    features = df.copy()
    
    # Ensure batch_size is numeric
    features['batch_size'] = features['batch_size'].astype(float)
    
    # Add interaction features (may contain NaNs if gpu_count is NaN)
    features['batch_x_gpu_count'] = features['batch_size'] * features['gpu_count']
    features['batch_per_gpu'] = features['batch_size'] / features['gpu_count']
    
    return features


def build_preprocessing_pipeline(X: pd.DataFrame, random_state: int = 42):
    """
    Build preprocessing pipeline with imputation and encoding.
    
    Args:
        X: Feature DataFrame
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (preprocessor, numeric_cols, categorical_cols)
    """
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_cols)}", file=sys.stderr)
    print(f"Categorical features: {len(categorical_cols)}", file=sys.stderr)
    
    # Numeric transformer: impute with median
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Categorical transformer: impute with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_cols, categorical_cols


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> tuple:
    """
    Train gradient boosting model with preprocessing pipeline.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_params: Optional model hyperparameters
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (pipeline, metrics_dict, missing_value_report)
    """
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'loss': 'squared_error',
            'random_state': random_state
        }
    
    # Report missing values
    print("\n=== Missing Value Analysis ===", file=sys.stderr)
    total_nans = X_train.isna().sum().sum()
    print(f"Total NaN values: {total_nans}", file=sys.stderr)
    
    if total_nans > 0:
        per_col_nans = X_train.isna().sum().sort_values(ascending=False)
        print("\nTop columns with missing values:", file=sys.stderr)
        for col, count in per_col_nans.head(10).items():
            if count > 0:
                pct = 100 * count / len(X_train)
                print(f"  {col}: {count} ({pct:.1f}%)", file=sys.stderr)
    
    missing_value_report = {
        'total_nans': int(total_nans),
        'columns_with_nans': {col: int(count) for col, count in X_train.isna().sum().items() if count > 0}
    }
    
    # Build preprocessing pipeline
    preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(X_train, random_state)
    
    # Build full pipeline
    print("\nTraining gradient boosting model with preprocessing pipeline...", file=sys.stderr)
    print(f"  Training samples: {len(X_train)}", file=sys.stderr)
    print(f"  Test samples: {len(X_test)}", file=sys.stderr)
    print(f"  Raw features: {X_train.shape[1]}", file=sys.stderr)
    
    model = GradientBoostingRegressor(**model_params)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Metrics
    metrics = {
        'train': {
            'mse': float(mean_squared_error(y_train, y_train_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'mae': float(mean_absolute_error(y_train, y_train_pred)),
            'r2': float(r2_score(y_train, y_train_pred))
        },
        'test': {
            'mse': float(mean_squared_error(y_test, y_test_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'mae': float(mean_absolute_error(y_test, y_test_pred)),
            'r2': float(r2_score(y_test, y_test_pred))
        }
    }
    
    # Feature importance (from the model in the pipeline)
    # Use robust feature name extraction
    try:
        # Try to get feature names from fitted preprocessor (sklearn >= 1.0)
        preprocessor = pipeline.named_steps['preprocessor']
        feature_names = list(preprocessor.get_feature_names_out())
    except (AttributeError, Exception) as e:
        # Fallback: construct names manually, but handle dropped columns
        print(f"Warning: Could not extract feature names automatically: {e}", file=sys.stderr)
        print("Using fallback feature naming", file=sys.stderr)
        
        try:
            # Get actual transformed feature count
            n_features = len(model.feature_importances_)
            
            # Build names for numeric features that weren't dropped
            feature_names = []
            for col in numeric_cols:
                if X_train[col].notna().sum() > 0:  # Only if column has data
                    feature_names.append(col)
            
            # Add categorical feature names if encoder exists
            if len(categorical_cols) > 0:
                try:
                    cat_transformer = preprocessor.named_transformers_.get('cat')
                    if cat_transformer and hasattr(cat_transformer.named_steps.get('onehot'), 'get_feature_names_out'):
                        # Get categorical columns that have data
                        valid_cat_cols = [col for col in categorical_cols if X_train[col].notna().sum() > 0]
                        if valid_cat_cols:
                            cat_names = cat_transformer.named_steps['onehot'].get_feature_names_out(valid_cat_cols)
                            feature_names.extend(cat_names)
                except Exception:
                    pass
            
            # If we still don't have enough names, pad with generic names
            while len(feature_names) < n_features:
                feature_names.append(f'feature_{len(feature_names)}')
            
            # Truncate if we have too many
            feature_names = feature_names[:n_features]
        except Exception as e2:
            # Ultimate fallback: generic names
            print(f"Warning: Fallback naming also failed: {e2}", file=sys.stderr)
            n_features = len(model.feature_importances_)
            feature_names = [f'feature_{i}' for i in range(n_features)]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Training Metrics ===", file=sys.stderr)
    print(f"Train RMSE: {metrics['train']['rmse']:.2f}", file=sys.stderr)
    print(f"Train MAE: {metrics['train']['mae']:.2f}", file=sys.stderr)
    print(f"Train R²: {metrics['train']['r2']:.4f}", file=sys.stderr)
    
    print("\n=== Test Metrics ===", file=sys.stderr)
    print(f"Test RMSE: {metrics['test']['rmse']:.2f}", file=sys.stderr)
    print(f"Test MAE: {metrics['test']['mae']:.2f}", file=sys.stderr)
    print(f"Test R²: {metrics['test']['r2']:.4f}", file=sys.stderr)
    
    print("\n=== Top 10 Features ===", file=sys.stderr)
    print(feature_importance.head(10).to_string(index=False), file=sys.stderr)
    
    metrics['feature_importance'] = feature_importance.to_dict('records')
    metrics['missing_values'] = missing_value_report
    
    return pipeline, metrics


def estimate_uncertainty(
    pipeline: Pipeline,
    X: pd.DataFrame,
    n_samples: int = 100
) -> np.ndarray:
    """
    Estimate prediction uncertainty using staged predictions.
    
    Args:
        pipeline: Trained pipeline with model
        X: Features
        n_samples: Number of staged predictions to use
    
    Returns:
        Array of uncertainty estimates (std dev)
    """
    # Transform features through preprocessing
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Use staged predictions to estimate uncertainty
    model = pipeline.named_steps['model']
    staged_preds = np.array(list(model.staged_predict(X_transformed)))
    
    # Take last n_samples stages
    if len(staged_preds) > n_samples:
        staged_preds = staged_preds[-n_samples:]
    
    # Compute std dev across stages as uncertainty proxy
    uncertainty = np.std(staged_preds, axis=0)
    
    return uncertainty


def main():
    parser = argparse.ArgumentParser(
        description="Train Predictor v1 baseline model"
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to RIM-1 dataset (parquet or CSV)'
    )
    parser.add_argument(
        '--output-dir',
        default='predictor_output',
        help='Output directory for model and metrics (default: predictor_output)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
    )
    parser.add_argument(
        '--target',
        default='throughput_mean',
        choices=['throughput_mean', 'duration_mean', 'estimated_cost_usd'],
        help='Target variable to predict (default: throughput_mean)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of boosting stages (default: 100)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='Max tree depth (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed set to: {args.seed}", file=sys.stderr)
    
    # Load data
    print(f"Loading data from: {args.data}", file=sys.stderr)
    data_path = Path(args.data)
    
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        print(f"Error: Unsupported file format: {data_path.suffix}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(df)} observations", file=sys.stderr)
    
    # Filter out rows with missing target
    df = df.dropna(subset=[args.target])
    print(f"After filtering: {len(df)} observations", file=sys.stderr)
    
    # Prepare features
    features = prepare_features(df)
    
    # Select feature columns (exclude target and metadata)
    exclude_cols = [
        'observation_id', 'timestamp', 'environment',
        'throughput_mean', 'throughput_std_dev',
        'duration_mean', 'duration_std_dev',
        'estimated_cost_usd', 'estimated_gpu_hours',
        'hourly_rate_usd', 'samples',
        'policy_mode', 'controller_algorithm',
        'controller_lambda', 'controller_selected_batch'
    ]
    
    feature_cols = [col for col in features.columns if col not in exclude_cols]
    
    X = features[feature_cols]
    y = features[args.target]
    
    print(f"Features: {len(feature_cols)}", file=sys.stderr)
    print(f"Target: {args.target}", file=sys.stderr)
    
    # Dataset quality gates
    print("\n=== Dataset Quality Check ===", file=sys.stderr)
    n_obs = len(X)
    print(f"Total observations: {n_obs}", file=sys.stderr)
    
    # Check for all-missing columns
    per_col_nonnull = X.notna().sum()
    all_missing_cols = per_col_nonnull[per_col_nonnull == 0].index.tolist()
    remaining_cols = per_col_nonnull[per_col_nonnull > 0].index.tolist()
    
    if all_missing_cols:
        print(f"\nColumns with 100% missing values: {len(all_missing_cols)}", file=sys.stderr)
        for col in all_missing_cols[:10]:
            print(f"  - {col}", file=sys.stderr)
    
    print(f"Columns with observed values: {len(remaining_cols)}", file=sys.stderr)
    
    # Define quality gates
    # Note: With row explosion, we now have rows_total >= observations
    MIN_ROWS = 60  # Minimum training rows (after explosion)
    MIN_OBSERVATIONS = 5  # Minimum observations (before explosion)
    MIN_FEATURES_WITH_DATA = 2
    
    insufficient_data = False
    reasons = []
    
    # Check both row count and observation count
    if n_obs < MIN_ROWS:
        insufficient_data = True
        reasons.append(f"Only {n_obs} training rows (minimum: {MIN_ROWS})")
    
    # For backwards compatibility, also check if we have enough unique observations
    # (This matters less with row explosion, but still useful)
    if 'observation_id' in X.columns:
        n_unique_obs = X['observation_id'].nunique()
        if n_unique_obs < MIN_OBSERVATIONS:
            insufficient_data = True
            reasons.append(f"Only {n_unique_obs} unique observations (minimum: {MIN_OBSERVATIONS})")
    
    if len(remaining_cols) < MIN_FEATURES_WITH_DATA:
        insufficient_data = True
        reasons.append(f"Only {len(remaining_cols)} features with data (minimum: {MIN_FEATURES_WITH_DATA})")
    
    if insufficient_data:
        print("\n⚠️  INSUFFICIENT DATA FOR TRAINING", file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
        
        # Calculate row source statistics if row_source column exists
        row_source_counts = {}
        n_unique_obs = n_obs
        avg_rows_per_obs = 1.0
        
        if 'row_source' in X.columns:
            row_source_counts = X['row_source'].value_counts().to_dict()
        
        if 'observation_id' in X.columns:
            n_unique_obs = X['observation_id'].nunique()
            if n_unique_obs > 0:
                avg_rows_per_obs = n_obs / n_unique_obs
        
        # Generate insufficient data report
        report = {
            'status': 'insufficient_data',
            'rows_total': n_obs,
            'observations_total': n_unique_obs,
            'avg_rows_per_observation': round(avg_rows_per_obs, 2),
            'row_source_counts': row_source_counts,
            'row_generation_mode': 'explode_sweep' if row_source_counts else 'single',
            'features_total': len(feature_cols),
            'features_with_data': len(remaining_cols),
            'all_missing_columns': all_missing_cols,
            'reasons': reasons,
            'quality_gates': {
                'min_rows': MIN_ROWS,
                'min_observations': MIN_OBSERVATIONS,
                'min_features_with_data': MIN_FEATURES_WITH_DATA
            },
            'recommended_action': 'Run GPU Benchmark Orchestrator with exploration enabled to generate diverse RIM-1 observations. With row explosion, 3-5 orchestrator runs should provide sufficient training data.'
        }
        
        # Save report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'predictor_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved insufficient data report to: {report_path}", file=sys.stderr)
        
        # Generate markdown report
        md_report = f"""# Predictor v1 Training Report

## Status: Insufficient Data

**Training rows:** {n_obs} (minimum required: {MIN_ROWS})

**Unique observations:** {n_unique_obs} (minimum required: {MIN_OBSERVATIONS})

**Average rows per observation:** {avg_rows_per_obs:.2f}

**Row generation mode:** {report['row_generation_mode']}

**Features with data:** {len(remaining_cols)} / {len(feature_cols)} (minimum required: {MIN_FEATURES_WITH_DATA})

## Issues

"""
        for reason in reasons:
            md_report += f"- {reason}\n"
        
        if all_missing_cols:
            md_report += f"\n## Columns with 100% Missing Values ({len(all_missing_cols)})\n\n"
            for col in all_missing_cols[:20]:
                md_report += f"- `{col}`\n"
            if len(all_missing_cols) > 20:
                md_report += f"\n... and {len(all_missing_cols) - 20} more\n"
        
        md_report += """\n## Recommended Action

Run the GPU Benchmark Orchestrator workflow with exploration enabled to generate diverse RIM-1 observations:

1. Go to Actions → GPU Benchmark Orchestrator v0.4 (DEV)
2. Run workflow with:
   - exploration_mode: epsilon_greedy or round_robin
   - sweep_batches: 64,128,256 (or your target batch sizes)
3. Wait for orchestrator to complete multiple runs
4. Re-run Predictor v1 Training workflow

## Next Steps

- Accumulate at least 20 observations with complete feature coverage
- Ensure observations include context (workload, instance_type, gpu_count)
- Ensure observations include action (batch_size)
- Ensure observations include outcome (throughput, duration)
"""
        
        md_path = output_dir / 'predictor_report.md'
        with open(md_path, 'w') as f:
            f.write(md_report)
        print(f"Saved markdown report to: {md_path}", file=sys.stderr)
        
        print("\n✓ Reports generated. Exiting without training.", file=sys.stderr)
        sys.exit(0)
    
    print("✓ Dataset quality checks passed", file=sys.stderr)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    
    # Train model
    model_params = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'loss': 'squared_error',
        'random_state': args.seed
    }
    
    pipeline, metrics = train_model(X_train, y_train, X_test, y_test, model_params, args.seed)
    
    # Save pipeline and metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\nSaved pipeline to: {model_path}", file=sys.stderr)
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}", file=sys.stderr)
    
    # Save feature names for inference
    feature_names_path = output_dir / 'feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"Saved feature names to: {feature_names_path}", file=sys.stderr)
    
    # Generate predictions with uncertainty on test set
    print("\n=== Generating Test Predictions ===", file=sys.stderr)
    test_preds = pipeline.predict(X_test)
    test_uncertainty = estimate_uncertainty(pipeline, X_test)
    
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': test_preds,
        'uncertainty': test_uncertainty,
        'error': y_test.values - test_preds,
        'abs_error': np.abs(y_test.values - test_preds)
    })
    
    predictions_path = output_dir / 'test_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved test predictions to: {predictions_path}", file=sys.stderr)
    
    print("\n=== Prediction Summary ===", file=sys.stderr)
    print(f"Mean uncertainty: {predictions_df['uncertainty'].mean():.2f}", file=sys.stderr)
    print(f"Max uncertainty: {predictions_df['uncertainty'].max():.2f}", file=sys.stderr)
    print(f"Mean absolute error: {predictions_df['abs_error'].mean():.2f}", file=sys.stderr)
    
    print("\n✓ Training complete", file=sys.stderr)


if __name__ == '__main__':
    main()
