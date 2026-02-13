#!/usr/bin/env python3
"""
RIM-1 Data Loader for Predictor v1

Loads RIM-1 observations from S3, filters, and materializes to local format
for training predictive models.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import boto3
import pandas as pd


def normalize_prefix(prefix: str, environment: str) -> str:
    """
    Normalize S3 prefix to avoid duplication of environment suffix.
    
    Args:
        prefix: Base S3 prefix (e.g., 'rim1/observations' or 'rim1/observations/dev')
        environment: Environment name (e.g., 'dev')
    
    Returns:
        Normalized prefix with environment appended only if not already present
    
    Examples:
        normalize_prefix('rim1/observations', 'dev') -> 'rim1/observations/dev'
        normalize_prefix('rim1/observations/dev', 'dev') -> 'rim1/observations/dev'
        normalize_prefix('rim1/observations/dev/', 'dev') -> 'rim1/observations/dev'
    """
    # Strip trailing slashes
    prefix = prefix.rstrip('/')
    
    # Check if prefix already ends with /<environment>
    if prefix.endswith(f'/{environment}'):
        return prefix
    
    # Append environment
    return f'{prefix}/{environment}'


def list_rim1_observations(
    bucket: str,
    prefix: str,
    environment: str,
    days: int,
    region: str = 'us-west-2'
) -> List[str]:
    """
    List RIM-1 observation keys from S3.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'rim1/observations')
        environment: Environment filter (dev/staging/prod)
        days: Number of days to look back
        region: AWS region
    
    Returns:
        List of S3 keys
    """
    s3 = boto3.client('s3', region_name=region)
    
    # Normalize prefix to avoid duplication (e.g., rim1/observations/dev/dev)
    effective_prefix = normalize_prefix(prefix, environment)
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    print(f"Scanning S3 for RIM-1 observations...", file=sys.stderr)
    print(f"  Bucket: {bucket}", file=sys.stderr)
    print(f"  Effective prefix: {effective_prefix}", file=sys.stderr)
    print(f"  Date range: {start_date.date()} to {end_date.date()}", file=sys.stderr)
    
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    
    # Scan date-partitioned structure: prefix/env/YYYY/MM/DD/
    for year in range(start_date.year, end_date.year + 1):
        for month in range(1, 13):
            # Skip months outside date range
            if year == start_date.year and month < start_date.month:
                continue
            if year == end_date.year and month > end_date.month:
                continue
            
            month_prefix = f"{effective_prefix}/{year:04d}/{month:02d}/"
            
            for page in paginator.paginate(Bucket=bucket, Prefix=month_prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.json'):
                        keys.append(key)
    
    print(f"Found {len(keys)} observation files", file=sys.stderr)
    return keys


def load_observation(bucket: str, key: str, region: str = 'us-west-2') -> Optional[Dict[str, Any]]:
    """
    Load a single RIM-1 observation from S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 key
        region: AWS region
    
    Returns:
        Observation dict or None if load fails
    """
    s3 = boto3.client('s3', region_name=region)
    
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"Warning: Failed to load {key}: {e}", file=sys.stderr)
        return None


def observation_to_rows(obs: Dict[str, Any], explode_sweep: bool = True, include_policy_rows: bool = True) -> List[Dict[str, Any]]:
    """
    Convert RIM-1 observation to multiple training rows.
    
    Explodes sweep candidates into separate rows for richer training data.
    
    RIM-1 schema v0.1.0 uses:
    - state.workload, state.instance_type, state.gpu_count, state.gpu_type
    - action.selected_batch_size (primary), action.actuated_batch_size (fallback)
    - evidence.batch_sweep.candidates (for explosion)
    
    RIM-1 schema v0.1.0 uses:
    - state.workload, state.instance_type, state.gpu_count, state.gpu_type
    - action.selected_batch_size (primary), action.actuated_batch_size (fallback)
    
    Args:
        obs: RIM-1 observation dict
        explode_sweep: If True, create separate rows for each sweep candidate
        include_policy_rows: If True, include policy_off and policy_on as separate rows
    
    Returns:
        List of flat dicts suitable for DataFrame rows
    """
    rows = []
    # Extract base state fields (shared across all rows from this observation)
    state = obs.get('state', {})
    context = obs.get('context', {})  # Legacy fallback
    
    workload = state.get('workload') or context.get('workload')
    instance_type = state.get('instance_type') or context.get('instance_type')
    gpu_count = state.get('gpu_count') or context.get('gpu_count')
    gpu_type = state.get('gpu_type')
    
    # Derive gpu_memory_gb from gpu_type if not explicitly provided
    gpu_memory_gb = state.get('gpu_memory_gb') or context.get('gpu_memory_gb')
    if gpu_memory_gb is None and gpu_type:
        # Common GPU memory mappings
        gpu_memory_map = {
            'A10G': 24.0,
            'A100': 40.0,
            'V100': 16.0,
            'T4': 16.0
        }
        gpu_memory_gb = gpu_memory_map.get(gpu_type)
    
    # Build base feature dict (shared state)
    base_features = {
        'observation_id': obs.get('observation_id'),
        'timestamp': obs.get('timestamp'),
        'environment': obs.get('environment'),
        'workload': workload,
        'instance_type': instance_type,
        'gpu_count': gpu_count,
        'gpu_memory_gb': gpu_memory_gb,
        'gpu_type': gpu_type,
    }
    
    # Extract provenance
    evidence = obs.get('evidence', {})
    provenance = {
        'orchestrator_run_id': evidence.get('github_run_id'),
        'suite_run_id': evidence.get('suite_run_id'),
    }
    
    # Try to explode sweep candidates
    batch_sweep = evidence.get('batch_sweep', {})
    candidates = batch_sweep.get('candidates', [])
    
    if explode_sweep and candidates:
        # Create one row per sweep candidate
        for candidate in candidates:
            row = base_features.copy()
            row['batch_size'] = candidate.get('batch_size')
            row['throughput_mean'] = candidate.get('throughput_mean')
            row['throughput_std_dev'] = candidate.get('throughput_std')
            row['duration_mean'] = candidate.get('duration_mean')
            row['duration_std_dev'] = candidate.get('duration_std')
            row['samples'] = candidate.get('samples')
            row['row_source'] = 'sweep'
            row['sweep_run_id'] = batch_sweep.get('sweep_run_id')
            row['orchestrator_run_id'] = provenance.get('orchestrator_run_id')
            row['suite_run_id'] = provenance.get('suite_run_id')
            
            # Compute derived features
            if row['batch_size'] is not None and gpu_count is not None and gpu_count > 0:
                row['batch_x_gpu_count'] = row['batch_size'] * gpu_count
                row['batch_per_gpu'] = row['batch_size'] / gpu_count
            else:
                row['batch_x_gpu_count'] = None
                row['batch_per_gpu'] = None
            
            rows.append(row)
    
    # Add policy_off and policy_on rows if requested
    outcome = obs.get('outcome', {})
    if include_policy_rows:
        # Policy-off row (baseline)
        comparison = outcome.get('comparison', {})
        baseline_throughput = comparison.get('baseline_throughput')
        if baseline_throughput:
            action = obs.get('action', {})
            baseline_batch = state.get('baseline_batch_size', 64)
            
            row = base_features.copy()
            row['batch_size'] = baseline_batch
            row['throughput_mean'] = baseline_throughput
            row['throughput_std_dev'] = None
            row['duration_mean'] = None
            row['duration_std_dev'] = None
            row['samples'] = None
            row['row_source'] = 'policy_off'
            row['orchestrator_run_id'] = provenance.get('orchestrator_run_id')
            row['suite_run_id'] = provenance.get('suite_run_id')
            
            if baseline_batch and gpu_count and gpu_count > 0:
                row['batch_x_gpu_count'] = baseline_batch * gpu_count
                row['batch_per_gpu'] = baseline_batch / gpu_count
            else:
                row['batch_x_gpu_count'] = None
                row['batch_per_gpu'] = None
            
            rows.append(row)
        
        # Policy-on row (selected batch)
        throughput = outcome.get('throughput', {})
        policy_throughput = throughput.get('mean')
        if policy_throughput:
            action = obs.get('action', {})
            selected_batch = (
                action.get('actuated_batch_size') or
                action.get('batch_size') or
                action.get('selected_batch_size')
            )
            
            row = base_features.copy()
            row['batch_size'] = selected_batch
            row['throughput_mean'] = policy_throughput
            row['throughput_std_dev'] = throughput.get('std_dev')
            row['duration_mean'] = outcome.get('duration', {}).get('mean')
            row['duration_std_dev'] = outcome.get('duration', {}).get('std_dev')
            row['samples'] = throughput.get('samples')
            row['row_source'] = 'policy_on'
            row['orchestrator_run_id'] = provenance.get('orchestrator_run_id')
            row['suite_run_id'] = provenance.get('suite_run_id')
            
            if selected_batch and gpu_count and gpu_count > 0:
                row['batch_x_gpu_count'] = selected_batch * gpu_count
                row['batch_per_gpu'] = selected_batch / gpu_count
            else:
                row['batch_x_gpu_count'] = None
                row['batch_per_gpu'] = None
            
            rows.append(row)
    
    # Fallback: if no rows generated, create single row from observation (legacy behavior)
    if not rows:
        action = obs.get('action', {})
        batch_size = (
            action.get('actuated_batch_size') or
            action.get('batch_size') or
            action.get('selected_batch_size')
        )
        
        if batch_size is None:
            print(f"Warning: batch_size missing in observation {obs.get('observation_id', 'unknown')}", file=sys.stderr)
        if workload is None:
            print(f"Warning: workload missing in observation {obs.get('observation_id', 'unknown')}", file=sys.stderr)
        
        row = base_features.copy()
        row['batch_size'] = batch_size
        row['policy_mode'] = action.get('policy')
        row['throughput_mean'] = obs.get('outcome', {}).get('throughput', {}).get('mean')
        row['throughput_std_dev'] = obs.get('outcome', {}).get('throughput', {}).get('std_dev')
        row['duration_mean'] = obs.get('outcome', {}).get('duration', {}).get('mean')
        row['duration_std_dev'] = obs.get('outcome', {}).get('duration', {}).get('std_dev')
        row['samples'] = obs.get('outcome', {}).get('samples')
        row['hourly_rate_usd'] = obs.get('outcome', {}).get('cost', {}).get('hourly_rate_usd')
        row['estimated_gpu_hours'] = obs.get('outcome', {}).get('cost', {}).get('estimated_gpu_hours')
        row['estimated_cost_usd'] = obs.get('outcome', {}).get('cost', {}).get('estimated_cost_usd')
        row['row_source'] = 'single'
        row['orchestrator_run_id'] = provenance.get('orchestrator_run_id')
        row['suite_run_id'] = provenance.get('suite_run_id')
        
        # Compute derived features
        if batch_size is not None and gpu_count is not None and gpu_count > 0:
            row['batch_x_gpu_count'] = batch_size * gpu_count
            row['batch_per_gpu'] = batch_size / gpu_count
        else:
            row['batch_x_gpu_count'] = None
            row['batch_per_gpu'] = None
        
        rows.append(row)
    
    return rows


def load_rim1_dataset(
    bucket: str,
    prefix: str,
    environment: str,
    days: int,
    region: str = 'us-west-2',
    output_path: Optional[str] = None,
    explode_sweep: bool = True,
    include_policy_rows: bool = True
) -> pd.DataFrame:
    """
    Load RIM-1 observations and materialize to DataFrame.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix
        environment: Environment filter
        days: Number of days to look back
        region: AWS region
        output_path: Optional path to save parquet file
        explode_sweep: If True, explode sweep candidates into separate rows
        include_policy_rows: If True, include policy_off and policy_on rows
    
    Returns:
        DataFrame with observations (potentially exploded into multiple rows per observation)
    """
    keys = list_rim1_observations(bucket, prefix, environment, days, region)
    
    if not keys:
        print("No observations found", file=sys.stderr)
        return pd.DataFrame()
    
    print(f"Loading {len(keys)} observations...", file=sys.stderr)
    print(f"Row generation mode: {'explode_sweep' if explode_sweep else 'single'}", file=sys.stderr)
    
    rows = []
    observations_loaded = 0
    for i, key in enumerate(keys):
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{len(keys)} observations...", file=sys.stderr)
        
        obs = load_observation(bucket, key, region)
        if obs:
            obs_rows = observation_to_rows(obs, explode_sweep=explode_sweep, include_policy_rows=include_policy_rows)
            rows.extend(obs_rows)
            observations_loaded += 1
    
    df = pd.DataFrame(rows)
    
    print(f"Loaded {observations_loaded} observations -> {len(df)} training rows", file=sys.stderr)
    if observations_loaded > 0:
        print(f"Average rows per observation: {len(df) / observations_loaded:.2f}", file=sys.stderr)
    
    if len(df) == 0:
        print("WARNING: No observations loaded. Dataset is empty.", file=sys.stderr)
        return df
    
    print(f"Columns: {list(df.columns)}", file=sys.stderr)
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}", file=sys.stderr)
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        print(f"Saved to: {output_file}", file=sys.stderr)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Load RIM-1 observations from S3 for predictor training"
    )
    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket containing RIM-1 observations'
    )
    parser.add_argument(
        '--prefix',
        default='rim1/observations',
        help='S3 prefix for observations (default: rim1/observations)'
    )
    parser.add_argument(
        '--environment',
        default='dev',
        choices=['dev', 'staging', 'prod'],
        help='Environment filter (default: dev)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to look back (default: 30)'
    )
    parser.add_argument(
        '--region',
        default='us-west-2',
        help='AWS region (default: us-west-2)'
    )
    parser.add_argument(
        '--output',
        help='Output path for parquet file (optional)'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Also save as CSV for inspection'
    )
    parser.add_argument(
        '--explode-sweep',
        action='store_true',
        default=True,
        help='Explode sweep candidates into separate training rows (default: True)'
    )
    parser.add_argument(
        '--no-explode-sweep',
        dest='explode_sweep',
        action='store_false',
        help='Disable sweep explosion (single row per observation)'
    )
    parser.add_argument(
        '--include-policy-rows',
        action='store_true',
        default=True,
        help='Include policy_off and policy_on as separate rows (default: True)'
    )
    parser.add_argument(
        '--no-include-policy-rows',
        dest='include_policy_rows',
        action='store_false',
        help='Disable policy row generation'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    df = load_rim1_dataset(
        bucket=args.bucket,
        prefix=args.prefix,
        environment=args.environment,
        days=args.days,
        region=args.region,
        output_path=args.output,
        explode_sweep=args.explode_sweep,
        include_policy_rows=args.include_policy_rows
    )
    
    # Save CSV if requested
    if args.csv and args.output:
        csv_path = Path(args.output).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}", file=sys.stderr)
    
    # Print summary stats
    print("\n=== Dataset Summary ===", file=sys.stderr)
    print(f"Total observations: {len(df)}", file=sys.stderr)
    
    if len(df) == 0:
        print("No observations found. Cannot compute statistics.", file=sys.stderr)
        print("\nRecommended action: Run GPU Benchmark Orchestrator to generate RIM-1 observations", file=sys.stderr)
        return
    
    print(f"Unique workloads: {df['workload'].nunique()}", file=sys.stderr)
    print(f"Unique instance types: {df['instance_type'].nunique()}", file=sys.stderr)
    print(f"Unique batch sizes: {sorted(df['batch_size'].unique())}", file=sys.stderr)
    print(f"Mean throughput: {df['throughput_mean'].mean():.2f}", file=sys.stderr)
    print(f"Mean cost: ${df['estimated_cost_usd'].mean():.4f}", file=sys.stderr)


def test_observation_to_row():
    """Test observation_to_row with synthetic RIM-1 observation."""
    synthetic_obs = {
        "schema_version": "0.1.0",
        "observation_id": "test-123",
        "timestamp": "2025-12-30T15:00:00Z",
        "environment": "dev",
        "state": {
            "workload": "resnet50",
            "instance_type": "g5.2xlarge",
            "gpu_count": 1,
            "gpu_type": "A10G",
            "aws_region": "us-west-2"
        },
        "action": {
            "policy": "radix_adaptive",
            "selected_batch_size": 128
        },
        "outcome": {
            "throughput": {
                "mean": 450.5,
                "std_dev": 12.3,
                "unit": "images_per_sec",
                "samples": 10
            },
            "duration": {
                "mean": 120.5,
                "std_dev": 5.2,
                "unit": "seconds",
                "samples": 10
            },
            "cost": {
                "currency": "USD",
                "hourly_rate_usd": 1.212,
                "estimated_gpu_hours": 0.0335,
                "estimated_cost_usd": 0.0406
            }
        },
        "evidence": {
            "suite_run_id": "test-suite-001",
            "github_run_id": "123456"
        }
    }
    
    row = observation_to_row(synthetic_obs)
    
    # Validate critical fields are extracted
    assert row['workload'] == 'resnet50', f"Expected workload='resnet50', got {row['workload']}"
    assert row['instance_type'] == 'g5.2xlarge', f"Expected instance_type='g5.2xlarge', got {row['instance_type']}"
    assert row['gpu_count'] == 1, f"Expected gpu_count=1, got {row['gpu_count']}"
    assert row['gpu_type'] == 'A10G', f"Expected gpu_type='A10G', got {row['gpu_type']}"
    assert row['batch_size'] == 128, f"Expected batch_size=128, got {row['batch_size']}"
    assert row['throughput_mean'] == 450.5, f"Expected throughput_mean=450.5, got {row['throughput_mean']}"
    
    # Validate derived features
    assert row['batch_x_gpu_count'] == 128, f"Expected batch_x_gpu_count=128, got {row['batch_x_gpu_count']}"
    assert row['batch_per_gpu'] == 128.0, f"Expected batch_per_gpu=128.0, got {row['batch_per_gpu']}"
    
    # Validate gpu_memory_gb derived from gpu_type
    assert row['gpu_memory_gb'] == 24.0, f"Expected gpu_memory_gb=24.0 (derived from A10G), got {row['gpu_memory_gb']}"
    
    print("✓ test_observation_to_row passed", file=sys.stderr)
    return True


if __name__ == '__main__':
    # Run self-test if --test flag provided
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running self-test...", file=sys.stderr)
        test_observation_to_row()
        print("All tests passed!", file=sys.stderr)
        sys.exit(0)
    
    main()
