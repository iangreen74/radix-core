"""
Cluster Policy Management for GPU Governance.
Stores and retrieves per-cluster GPU job limits and policies.
"""
import os
from typing import Dict, Any
import boto3
from botocore.exceptions import ClientError

# DynamoDB client
_dynamodb = boto3.resource('dynamodb')
_policy_table_name = os.environ.get('CLUSTER_POLICY_TABLE', '')
_policy_table = None

# Default GPU job limits
DEFAULT_GPU_JOBS_PER_USER = 1
DEFAULT_GPU_JOBS_TOTAL = 100  # Cluster-wide limit (not enforced in Phase 1)


def _get_table():
    """Lazy-load the policy table."""
    global _policy_table
    if _policy_table is None and _policy_table_name:
        _policy_table = _dynamodb.Table(_policy_table_name)
    return _policy_table


def get_cluster_policy(cluster_id: str) -> Dict[str, Any]:
    """
    Get GPU policy for a cluster.
    
    Args:
        cluster_id: Cluster identifier
        
    Returns:
        Policy dict with:
            - cluster_id: str
            - max_gpu_jobs_per_user: int (default: 1)
            - max_gpu_jobs_total: int (default: 100, not enforced yet)
    """
    table = _get_table()
    if not table:
        # Return defaults if table not configured
        return {
            'cluster_id': cluster_id,
            'max_gpu_jobs_per_user': DEFAULT_GPU_JOBS_PER_USER,
            'max_gpu_jobs_total': DEFAULT_GPU_JOBS_TOTAL,
        }
    
    try:
        resp = table.get_item(Key={'cluster_id': cluster_id})
        item = resp.get('Item', {})
        
        return {
            'cluster_id': cluster_id,
            'max_gpu_jobs_per_user': int(item.get('max_gpu_jobs_per_user', DEFAULT_GPU_JOBS_PER_USER)),
            'max_gpu_jobs_total': int(item.get('max_gpu_jobs_total', DEFAULT_GPU_JOBS_TOTAL)),
        }
    except ClientError as e:
        print(f"Error getting cluster policy for {cluster_id}: {e}")
        # Return defaults on error
        return {
            'cluster_id': cluster_id,
            'max_gpu_jobs_per_user': DEFAULT_GPU_JOBS_PER_USER,
            'max_gpu_jobs_total': DEFAULT_GPU_JOBS_TOTAL,
        }


def set_cluster_policy(cluster_id: str, max_gpu_jobs_per_user: int = None, 
                       max_gpu_jobs_total: int = None) -> Dict[str, Any]:
    """
    Set GPU policy for a cluster.
    
    Args:
        cluster_id: Cluster identifier
        max_gpu_jobs_per_user: Max concurrent GPU jobs per user (optional)
        max_gpu_jobs_total: Max total GPU jobs on cluster (optional)
        
    Returns:
        Updated policy dict
    """
    table = _get_table()
    if not table:
        raise RuntimeError("Cluster policy table not configured")
    
    # Build update item
    item = {'cluster_id': cluster_id}
    
    if max_gpu_jobs_per_user is not None:
        if max_gpu_jobs_per_user < 0:
            raise ValueError("max_gpu_jobs_per_user must be >= 0")
        item['max_gpu_jobs_per_user'] = max_gpu_jobs_per_user
    
    if max_gpu_jobs_total is not None:
        if max_gpu_jobs_total < 0:
            raise ValueError("max_gpu_jobs_total must be >= 0")
        item['max_gpu_jobs_total'] = max_gpu_jobs_total
    
    # If no updates provided, just return current policy
    if len(item) == 1:
        return get_cluster_policy(cluster_id)
    
    try:
        table.put_item(Item=item)
        return get_cluster_policy(cluster_id)
    except ClientError as e:
        print(f"Error setting cluster policy for {cluster_id}: {e}")
        raise
