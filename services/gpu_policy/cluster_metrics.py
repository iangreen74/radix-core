"""
Cluster Metrics Module.
Stores and retrieves cluster metrics from agents (GPU utilization, job counts, etc.).
"""
import os
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

# DynamoDB client
_dynamodb = boto3.resource('dynamodb')
_table_name = os.environ.get('CLUSTER_METRICS_TABLE_NAME', '')
_table = None


def _get_table():
    """Lazy-load the metrics table."""
    global _table
    if _table is None and _table_name:
        _table = _dynamodb.Table(_table_name)
    return _table


def write_cluster_metric(metric: Dict) -> None:
    """
    Write a cluster metric data point.
    
    Args:
        metric: Dict containing:
            - cluster_id (required)
            - timestamp (required, ISO 8601 string)
            - gpu_utilization_percent (optional)
            - gpu_memory_used_mb (optional)
            - running_jobs (optional)
            - pending_jobs (optional)
    """
    table = _get_table()
    if not table:
        print("Warning: CLUSTER_METRICS_TABLE_NAME not configured")
        return
    
    if not metric.get('cluster_id') or not metric.get('timestamp'):
        print("Warning: cluster_id and timestamp required for metrics")
        return
    
    try:
        table.put_item(Item=metric)
    except ClientError as e:
        print(f"Error writing cluster metric: {e}")


def get_cluster_metrics(cluster_id: str, limit: int = 100) -> List[Dict]:
    """
    Get recent cluster metrics for a cluster.
    
    Args:
        cluster_id: Cluster identifier
        limit: Maximum number of metrics to return (default: 100)
        
    Returns:
        List of metric dicts in chronological order (oldest to newest)
    """
    table = _get_table()
    if not table:
        print("Warning: CLUSTER_METRICS_TABLE_NAME not configured")
        return []
    
    try:
        resp = table.query(
            KeyConditionExpression='cluster_id = :cid',
            ExpressionAttributeValues={':cid': cluster_id},
            ScanIndexForward=False,  # Newest first
            Limit=limit,
        )
        
        items = resp.get('Items', [])
        
        # Return in chronological order (oldest to newest)
        return list(reversed(items))
    
    except ClientError as e:
        print(f"Error querying cluster metrics: {e}")
        return []
