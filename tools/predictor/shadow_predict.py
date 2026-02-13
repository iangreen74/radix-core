#!/usr/bin/env python3
"""
Shadow predictor: compute predictions without actuating.
Used during suite report generation to log predictor recommendations alongside controller decisions.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import joblib
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed (joblib, pandas)", file=sys.stderr)
    sys.exit(1)


def load_model(model_path: str):
    """Load trained predictor pipeline."""
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        print(f"Error loading pipeline: {e}", file=sys.stderr)
        return None


def prepare_features(batch_sizes: list, workload: str = "resnet50", instance_type: str = "g5.2xlarge", 
                     gpu_count: int = 1, gpu_memory_gb: float = 24.0):
    """Prepare features for prediction matching training schema."""
    features = []
    for batch_size in batch_sizes:
        features.append({
            "batch_size": batch_size,
            "workload": workload,
            "instance_type": instance_type,
            "gpu_count": gpu_count,
            "gpu_memory_gb": gpu_memory_gb,
            "batch_x_gpu_count": batch_size * gpu_count,
            "batch_per_gpu": batch_size / gpu_count,
        })
    return pd.DataFrame(features)


def predict_with_uncertainty(pipeline, features_df):
    """Generate predictions with uncertainty estimates."""
    try:
        # For pipeline with gradient boosting, use staged predictions for uncertainty
        predictions = []
        
        # Check if this is a pipeline
        if hasattr(pipeline, 'named_steps'):
            # Extract preprocessor and model from pipeline
            preprocessor = pipeline.named_steps.get('preprocessor')
            model = pipeline.named_steps.get('model')
            
            if preprocessor and model and hasattr(model, 'estimators_'):
                # Transform features through preprocessing
                features_transformed = preprocessor.transform(features_df)
                
                # Gradient boosting model - use staged predictions
                staged_preds = list(model.staged_predict(features_transformed))
                mean_pred = staged_preds[-1]  # Final prediction
                
                # Estimate uncertainty from variance across stages
                if len(staged_preds) > 10:
                    recent_stages = staged_preds[-10:]
                    uncertainty = [pd.Series([stage[i] for stage in recent_stages]).std() 
                                  for i in range(len(features_df))]
                else:
                    uncertainty = [0.0] * len(features_df)
            else:
                # Pipeline without gradient boosting - use direct prediction
                mean_pred = pipeline.predict(features_df)
                uncertainty = [0.0] * len(features_df)
        elif hasattr(pipeline, 'estimators_'):
            # Legacy: raw gradient boosting model (not a pipeline)
            staged_preds = list(pipeline.staged_predict(features_df))
            mean_pred = staged_preds[-1]
            
            if len(staged_preds) > 10:
                recent_stages = staged_preds[-10:]
                uncertainty = [pd.Series([stage[i] for stage in recent_stages]).std() 
                              for i in range(len(features_df))]
            else:
                uncertainty = [0.0] * len(features_df)
        else:
            # Fallback for other models
            mean_pred = pipeline.predict(features_df)
            uncertainty = [0.0] * len(features_df)
        
        for i, row in features_df.iterrows():
            predictions.append({
                "batch_size": int(row["batch_size"]),
                "predicted_throughput": float(mean_pred[i]),
                "uncertainty": float(uncertainty[i])
            })
        
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        return None


def select_optimal_batch(predictions, lambda_param: float = 0.2):
    """Select optimal batch using free energy-like criterion."""
    scores = []
    for pred in predictions:
        # Lower is better: minimize negative throughput + lambda * uncertainty
        score = -pred["predicted_throughput"] + lambda_param * pred["uncertainty"]
        scores.append({
            "batch_size": pred["batch_size"],
            "score": score,
            "predicted_throughput": pred["predicted_throughput"],
            "uncertainty": pred["uncertainty"]
        })
    
    # Select batch with minimum score
    optimal = min(scores, key=lambda x: x["score"])
    return optimal["batch_size"], scores


def main():
    parser = argparse.ArgumentParser(description="Shadow predictor for suite report")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--batches", required=True, help="Comma-separated batch sizes to evaluate")
    parser.add_argument("--workload", default="resnet50", help="Workload name")
    parser.add_argument("--instance-type", default="g5.2xlarge", help="Instance type")
    parser.add_argument("--lambda", dest="lambda_param", type=float, default=0.2, 
                       help="Lambda parameter for uncertainty weighting")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(b.strip()) for b in args.batches.split(",")]
    
    print(f"Loading model from {args.model}")
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    print(f"Preparing features for batches: {batch_sizes}")
    features_df = prepare_features(batch_sizes, args.workload, args.instance_type)
    
    print("Generating predictions...")
    predictions = predict_with_uncertainty(model, features_df)
    if predictions is None:
        sys.exit(1)
    
    print(f"Selecting optimal batch (lambda={args.lambda_param})...")
    optimal_batch, scores = select_optimal_batch(predictions, args.lambda_param)
    
    # Build decision output
    decision = {
        "predictor_version": "v1_baseline",
        "model_type": "gradient_boosting",
        "lambda": args.lambda_param,
        "candidate_batches": batch_sizes,
        "predictions": predictions,
        "scores": scores,
        "selected_batch": optimal_batch,
        "shadow_mode": True,
        "note": "Predictor recommendation computed in shadow mode - NOT used for actuation"
    }
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(decision, f, indent=2)
    
    print(f"✓ Predictor decision written to {args.output}")
    print(f"  Selected batch: {optimal_batch}")
    print(f"  Predictions: {len(predictions)} batches evaluated")


if __name__ == "__main__":
    main()
