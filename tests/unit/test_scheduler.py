"""
Unit tests for Radix Core 1.1a Scheduler

Tests the deterministic cost-based scheduler for cluster selection.
"""

import unittest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Import scheduler module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/cloud-api'))

from radix_core.scheduler import select_cluster, NoEligibleClustersError, _score_by_cost


class TestScheduler(unittest.TestCase):
    """Test suite for scheduler module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tenant_id = "user-test123"
        self.now = datetime.now(timezone.utc)
        
        # Create test clusters
        self.clusters = [
            {
                'cluster_id': 'cluster-a',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.50')}
            },
            {
                'cluster_id': 'cluster-b',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 8},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.75')}
            },
            {
                'cluster_id': 'cluster-c',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 2},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.40')}
            }
        ]
    
    def test_select_cheapest_cluster(self):
        """Test that scheduler selects the cheapest eligible cluster"""
        job_spec = {'orchestration': {'num_gpus': 1}}
        
        result = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=self.clusters,
            policy='cost_optimized'
        )
        
        # cluster-c has lowest cost per GPU ($0.40)
        self.assertEqual(result['cluster_id'], 'cluster-c')
        self.assertEqual(result['policy'], 'cost_optimized')
        self.assertEqual(result['version'], '1.1a-cost-v1')
        self.assertIn('score', result)
        self.assertIn('candidates', result)
        self.assertIn('scheduled_at', result)
    
    def test_multi_gpu_cost_calculation(self):
        """Test that cost is calculated correctly for multi-GPU jobs"""
        job_spec = {'orchestration': {'num_gpus': 4}}
        
        result = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=self.clusters,
            policy='cost_optimized'
        )
        
        # cluster-a: 4 GPUs * $0.50 = $2.00
        # cluster-b: 4 GPUs * $0.75 = $3.00
        # cluster-c: only 2 GPUs, ineligible
        # cluster-a should win
        self.assertEqual(result['cluster_id'], 'cluster-a')
    
    def test_insufficient_gpu_capacity(self):
        """Test that clusters without enough GPUs are filtered out"""
        job_spec = {'orchestration': {'num_gpus': 10}}
        
        with self.assertRaises(NoEligibleClustersError) as ctx:
            select_cluster(
                tenant_id=self.tenant_id,
                job_spec=job_spec,
                clusters=self.clusters,
                policy='cost_optimized'
            )
        
        self.assertIn('No eligible clusters', str(ctx.exception))
    
    def test_inactive_cluster_filtered(self):
        """Test that inactive clusters are filtered out"""
        clusters = [
            {
                'cluster_id': 'cluster-inactive',
                'tenant_id': self.tenant_id,
                'status': 'inactive',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.10')}
            }
        ]
        
        job_spec = {'orchestration': {'num_gpus': 1}}
        
        with self.assertRaises(NoEligibleClustersError):
            select_cluster(
                tenant_id=self.tenant_id,
                job_spec=job_spec,
                clusters=clusters,
                policy='cost_optimized'
            )
    
    def test_stale_heartbeat_filtered(self):
        """Test that clusters with stale heartbeats are filtered out"""
        stale_time = self.now - timedelta(seconds=200)
        
        clusters = [
            {
                'cluster_id': 'cluster-stale',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': stale_time.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.10')}
            }
        ]
        
        job_spec = {'orchestration': {'num_gpus': 1}}
        
        with self.assertRaises(NoEligibleClustersError):
            select_cluster(
                tenant_id=self.tenant_id,
                job_spec=job_spec,
                clusters=clusters,
                policy='cost_optimized',
                freshness_threshold_seconds=120
            )
    
    def test_deterministic_tie_breaking(self):
        """Test that ties are broken deterministically by cluster_id"""
        # Two clusters with same cost
        clusters = [
            {
                'cluster_id': 'cluster-z',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.50')}
            },
            {
                'cluster_id': 'cluster-a',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.50')}
            }
        ]
        
        job_spec = {'orchestration': {'num_gpus': 1}}
        
        result = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=clusters,
            policy='cost_optimized'
        )
        
        # cluster-a should win (alphabetically first)
        self.assertEqual(result['cluster_id'], 'cluster-a')
        
        # Run multiple times to ensure determinism
        for _ in range(5):
            result2 = select_cluster(
                tenant_id=self.tenant_id,
                job_spec=job_spec,
                clusters=clusters,
                policy='cost_optimized'
            )
            self.assertEqual(result2['cluster_id'], 'cluster-a')
    
    def test_missing_cost_fallback(self):
        """Test that clusters without cost data are deprioritized but still selectable"""
        clusters = [
            {
                'cluster_id': 'cluster-no-cost',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {}  # No cost data
            },
            {
                'cluster_id': 'cluster-with-cost',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {'cost_per_gpu_hour_usd': Decimal('0.50')}
            }
        ]
        
        job_spec = {'orchestration': {'num_gpus': 1}}
        
        result = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=clusters,
            policy='cost_optimized'
        )
        
        # cluster-with-cost should win
        self.assertEqual(result['cluster_id'], 'cluster-with-cost')
    
    def test_only_no_cost_clusters(self):
        """Test that scheduler works when no clusters have cost data"""
        clusters = [
            {
                'cluster_id': 'cluster-b',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {}
            },
            {
                'cluster_id': 'cluster-a',
                'tenant_id': self.tenant_id,
                'status': 'active',
                'last_heartbeat_at': self.now.isoformat(),
                'capabilities': {'gpu_count': 4},
                'pricing': {}
            }
        ]
        
        job_spec = {'orchestration': {'num_gpus': 1}}
        
        result = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=clusters,
            policy='cost_optimized'
        )
        
        # Should select deterministically (fallback uses cluster_id hash)
        self.assertIn(result['cluster_id'], ['cluster-a', 'cluster-b'])
        
        # Should be deterministic across runs
        result2 = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=clusters,
            policy='cost_optimized'
        )
        self.assertEqual(result['cluster_id'], result2['cluster_id'])
    
    def test_candidates_list(self):
        """Test that all eligible clusters are listed in candidates"""
        job_spec = {'orchestration': {'num_gpus': 1}}
        
        result = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=self.clusters,
            policy='cost_optimized'
        )
        
        # All 3 clusters should be candidates
        self.assertEqual(len(result['candidates']), 3)
        self.assertIn('cluster-a', result['candidates'])
        self.assertIn('cluster-b', result['candidates'])
        self.assertIn('cluster-c', result['candidates'])
    
    def test_score_by_cost_function(self):
        """Test the _score_by_cost helper function directly"""
        scored = _score_by_cost(self.clusters, num_gpus=2)
        
        # Should return list of dicts with cluster_id and score
        self.assertEqual(len(scored), 3)
        
        # Check scores are calculated correctly
        scores_by_id = {s['cluster_id']: s['score'] for s in scored}
        self.assertEqual(scores_by_id['cluster-a'], Decimal('0.50') * 2)
        self.assertEqual(scores_by_id['cluster-b'], Decimal('0.75') * 2)
        self.assertEqual(scores_by_id['cluster-c'], Decimal('0.40') * 2)
    
    def test_legacy_num_gpus_support(self):
        """Test that legacy num_gpus field still works"""
        job_spec = {'num_gpus': 1}  # Legacy format
        
        result = select_cluster(
            tenant_id=self.tenant_id,
            job_spec=job_spec,
            clusters=self.clusters,
            policy='cost_optimized'
        )
        
        # Should still work and select cheapest cluster
        self.assertEqual(result['cluster_id'], 'cluster-c')


if __name__ == '__main__':
    unittest.main()
