"""Test mutating admission webhook functionality."""

import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.webhook import app, WebhookConfig, MutatingWebhook


class TestMutatingWebhook:
    """Test the mutating admission webhook."""

    def setup_method(self):
        """Set up test environment."""
        self.config = WebhookConfig(
            scheduler_url="http://mock-scheduler:8080",
            enable_mutation=True
        )
        self.webhook = MutatingWebhook(self.config)
        self.client = TestClient(app)

    def test_health_endpoints(self):
        """Test health and readiness endpoints."""
        # Health check
        response = self.client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Readiness check
        response = self.client.get("/readyz")
        assert response.status_code == 200
        # Should work even if scheduler is down (disconnected)

    def test_extract_job_features(self):
        """Test job feature extraction from Pod spec."""
        pod_spec = {
            "containers": [
                {
                    "name": "trainer",
                    "resources": {
                        "requests": {
                            "nvidia.com/gpu": "1",
                            "memory": "8Gi"
                        }
                    }
                }
            ]
        }

        annotations = {
            "app.kubernetes.io/job-type": "train-bert",
            "gpu.mem.gi": "40",
            "ml.batch_size": "32",
            "scheduler.radix.ai/tenant": "team-a"
        }

        features = self.webhook.extract_job_features(pod_spec, annotations)

        assert features["job_type"] == "train-bert"
        assert features["gpu_mem_gb"] == 40.0
        assert features["batch_size"] == 32
        assert features["tenant"] == "team-a"

    def test_extract_job_features_fallback(self):
        """Test job feature extraction with missing annotations."""
        pod_spec = {"containers": []}
        annotations = {}

        features = self.webhook.extract_job_features(pod_spec, annotations)

        assert features["job_type"] == "unknown"
        assert features["tenant"] == "default"

    def test_priority_class_mapping(self):
        """Test priority class mapping from scores."""
        # Test each bucket
        assert self.webhook.get_priority_class(95) == "gpu-ultra"
        assert self.webhook.get_priority_class(75) == "gpu-high"
        assert self.webhook.get_priority_class(55) == "gpu-medium"
        assert self.webhook.get_priority_class(25) == "gpu-default"

        # Test boundary conditions
        assert self.webhook.get_priority_class(90) == "gpu-ultra"
        assert self.webhook.get_priority_class(89) == "gpu-high"
        assert self.webhook.get_priority_class(0) == "gpu-default"

    def test_json_patch_creation(self):
        """Test JSON patch creation for Pod mutation."""
        pod = {
            "metadata": {
                "name": "test-pod",
                "annotations": {}
            },
            "spec": {
                "containers": [{"name": "test"}]
            }
        }

        score_response = {
            "priority_score": 85,
            "gpu_selector": {
                "nodeSelector": {"gpu.nvidia.com/class": "A100-80GB"},
                "tolerations": [{
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule"
                }]
            },
            "avoid_co_locate_with": ["train-llama"],
            "terms": {
                "chosen_gpu": "A100-80GB",
                "mu": 5.0,
                "sigma": 1.2,
                "ig": 0.3,
                "penalty": 0.1,
                "cost": 5.8
            }
        }

        patches = self.webhook.create_json_patch(pod, score_response)

        # Should contain priority class patch
        priority_patches = [p for p in patches if p["path"] == "/spec/priorityClassName"]
        assert len(priority_patches) == 1
        assert priority_patches[0]["value"] == "gpu-high"

        # Should contain nodeSelector patch
        node_selector_patches = [p for p in patches if "/spec/nodeSelector" in p["path"]]
        assert len(node_selector_patches) > 0

        # Should contain tolerations patch
        tolerations_patches = [p for p in patches if p["path"] == "/spec/tolerations"]
        assert len(tolerations_patches) == 1

        # Should contain anti-affinity patch
        affinity_patches = [p for p in patches if "/spec/affinity" in p["path"]]
        assert len(affinity_patches) > 0

        # Should contain debug annotations
        annotation_patches = [p for p in patches if "/metadata/annotations/" in p["path"]]
        assert len(annotation_patches) >= 3  # priority-score, chosen-gpu, decision-timestamp

    @patch('app.webhook.httpx.AsyncClient.post')
    async def test_call_scheduler_success(self, mock_post):
        """Test successful scheduler API call."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "priority_score": 75,
            "gpu_selector": {"nodeSelector": {"gpu.nvidia.com/class": "A100-80GB"}},
            "avoid_co_locate_with": [],
            "terms": {"chosen_gpu": "A100-80GB"}
        }
        mock_post.return_value = mock_response

        result = await self.webhook.call_scheduler(
            job_type="test-job",
            features={"gpu_mem_gb": 40},
            candidates=["A100-80GB"],
            colocated=[]
        )

        assert result is not None
        assert result["priority_score"] == 75
        mock_post.assert_called_once()

    @patch('app.webhook.httpx.AsyncClient.post')
    async def test_call_scheduler_failure(self, mock_post):
        """Test scheduler API call failure handling."""
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        result = await self.webhook.call_scheduler(
            job_type="test-job",
            features={},
            candidates=["A100-80GB"],
            colocated=[]
        )

        assert result is None

    @patch('app.webhook.httpx.AsyncClient.post')
    async def test_call_scheduler_timeout(self, mock_post):
        """Test scheduler API call timeout handling."""
        mock_post.side_effect = Exception("Timeout")

        result = await self.webhook.call_scheduler(
            job_type="test-job",
            features={},
            candidates=["A100-80GB"],
            colocated=[]
        )

        assert result is None

    @patch.object(MutatingWebhook, 'call_scheduler')
    async def test_mutate_pod_success(self, mock_call_scheduler):
        """Test successful Pod mutation."""
        mock_call_scheduler.return_value = {
            "priority_score": 80,
            "gpu_selector": {
                "nodeSelector": {"gpu.nvidia.com/class": "A100-80GB"},
                "tolerations": []
            },
            "avoid_co_locate_with": [],
            "terms": {"chosen_gpu": "A100-80GB"}
        }

        admission_request = {
            "uid": "test-uid-123",
            "object": {
                "metadata": {
                    "name": "test-pod",
                    "annotations": {
                        "app.kubernetes.io/job-type": "train-bert"
                    }
                },
                "spec": {
                    "containers": [{"name": "trainer"}]
                }
            }
        }

        response = await self.webhook.mutate_pod(admission_request)

        assert response.uid == "test-uid-123"
        assert response.allowed is True
        assert response.patch is not None
        assert response.patchType == "JSONPatch"

        # Decode and verify patch
        import base64
        patch_json = base64.b64decode(response.patch).decode()
        patches = json.loads(patch_json)
        assert len(patches) > 0

    async def test_mutate_pod_unknown_job(self):
        """Test Pod mutation with unknown job type."""
        admission_request = {
            "uid": "test-uid-456",
            "object": {
                "metadata": {
                    "name": "test-pod",
                    "annotations": {}  # No job type annotation
                },
                "spec": {
                    "containers": [{"name": "trainer"}]
                }
            }
        }

        response = await self.webhook.mutate_pod(admission_request)

        assert response.uid == "test-uid-456"
        assert response.allowed is True
        assert response.patch is None  # No mutation for unknown jobs

    async def test_mutate_pod_mutation_disabled(self):
        """Test Pod mutation when mutations are disabled."""
        config_disabled = WebhookConfig(enable_mutation=False)
        webhook_disabled = MutatingWebhook(config_disabled)

        admission_request = {
            "uid": "test-uid-789",
            "object": {
                "metadata": {
                    "name": "test-pod",
                    "annotations": {
                        "app.kubernetes.io/job-type": "train-bert"
                    }
                },
                "spec": {
                    "containers": [{"name": "trainer"}]
                }
            }
        }

        response = await webhook_disabled.mutate_pod(admission_request)

        assert response.uid == "test-uid-789"
        assert response.allowed is True
        assert response.patch is None  # No mutation when disabled

    def test_mutate_endpoint_success(self):
        """Test the /mutate endpoint with valid admission review."""
        admission_review = {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "request": {
                "uid": "test-uid",
                "object": {
                    "metadata": {
                        "name": "test-pod",
                        "annotations": {"app.kubernetes.io/job-type": "train-bert"}
                    },
                    "spec": {"containers": [{"name": "trainer"}]}
                }
            }
        }

        with patch.object(MutatingWebhook, 'call_scheduler', return_value=None):
            response = self.client.post("/mutate", json=admission_review)

        assert response.status_code == 200
        result = response.json()
        assert result["apiVersion"] == "admission.k8s.io/v1"
        assert result["kind"] == "AdmissionReview"
        assert result["response"]["uid"] == "test-uid"
        assert result["response"]["allowed"] is True

    def test_mutate_endpoint_error_handling(self):
        """Test the /mutate endpoint error handling."""
        # Invalid admission review
        invalid_review = {"invalid": "data"}

        response = self.client.post("/mutate", json=invalid_review)

        # Should still return 200 but with allowed=True (fail open)
        assert response.status_code == 200
        result = response.json()
        assert result["response"]["allowed"] is True

    def test_candidate_gpu_types(self):
        """Test GPU type candidate generation."""
        candidates = self.webhook.get_candidate_gpu_types()

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert "A100-80GB" in candidates
        assert "H100-80GB" in candidates

    def test_colocated_job_extraction(self):
        """Test colocated job type extraction."""
        pod_spec = {"containers": []}

        # Simplified implementation returns empty list
        colocated = self.webhook.extract_colocated_jobs(pod_spec)
        assert isinstance(colocated, list)

    async def test_webhook_cleanup(self):
        """Test webhook cleanup on shutdown."""
        await self.webhook.close()
        # Should not raise any exceptions
