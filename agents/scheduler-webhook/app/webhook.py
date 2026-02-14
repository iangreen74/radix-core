"""Mutating admission webhook for information-theoretic GPU scheduler."""

import json
import logging
import base64
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field


@dataclass
class WebhookConfig:
    """Configuration for the webhook."""
    scheduler_url: str = "http://scheduler-agent:8080"
    timeout_seconds: float = 5.0
    enable_mutation: bool = True
    priority_buckets: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.priority_buckets is None:
            self.priority_buckets = [
                {"minScore": 90, "priorityClassName": "gpu-ultra"},
                {"minScore": 70, "priorityClassName": "gpu-high"},
                {"minScore": 50, "priorityClassName": "gpu-medium"},
                {"minScore": 0, "priorityClassName": "gpu-default"}
            ]


class AdmissionReview(BaseModel):
    """Kubernetes AdmissionReview object."""
    apiVersion: str
    kind: str
    request: Dict[str, Any]


class AdmissionResponse(BaseModel):
    """Kubernetes AdmissionResponse object."""
    uid: str
    allowed: bool
    patch: Optional[str] = None
    patchType: Optional[str] = None


class MutatingWebhook:
    """Mutating admission webhook for GPU scheduling."""

    def __init__(self, config: WebhookConfig):
        self.config = config
        self.http_client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()

    def extract_job_features(self, pod_spec: Dict[str, Any],
                           annotations: Dict[str, str]) -> Dict[str, Any]:
        """Extract job features from Pod spec and annotations."""
        features = {}

        # Extract from annotations
        if "app.kubernetes.io/job-type" in annotations:
            features["job_type"] = annotations["app.kubernetes.io/job-type"]
        else:
            # Fallback to pod name or labels
            features["job_type"] = "unknown"

        if "gpu.mem.gi" in annotations:
            try:
                features["gpu_mem_gb"] = float(annotations["gpu.mem.gi"])
            except ValueError:
                pass

        if "ml.batch_size" in annotations:
            try:
                features["batch_size"] = int(annotations["ml.batch_size"])
            except ValueError:
                pass

        # Extract tenant from namespace or annotations
        features["tenant"] = annotations.get("scheduler.radix.ai/tenant", "default")

        # Estimate GPU memory from resource requests
        containers = pod_spec.get("containers", [])
        for container in containers:
            resources = container.get("resources", {})
            requests = resources.get("requests", {})

            # Check for NVIDIA GPU requests
            for key in requests:
                if "nvidia.com/gpu" in key or "gpu" in key.lower():
                    # Estimate memory needs
                    if "gpu_mem_gb" not in features:
                        features["gpu_mem_gb"] = 40.0  # Default assumption

        return features

    def get_candidate_gpu_types(self) -> List[str]:
        """Get available GPU types from cluster node labels."""
        try:
            from kubernetes import client as k8s_client, config as k8s_config
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()

            v1 = k8s_client.CoreV1Api()
            nodes = v1.list_node().items
            gpu_types = set()
            for node in nodes:
                labels = node.metadata.labels or {}
                # Check common GPU label conventions
                for key in ("gpu.nvidia.com/class", "nvidia.com/gpu.product"):
                    if key in labels:
                        gpu_types.add(labels[key])
            if gpu_types:
                return list(gpu_types)
        except Exception as e:
            logging.debug(f"Failed to query cluster GPU types: {e}")

        # Fallback to known GPU types
        return ["A100-80GB", "A100-40GB", "L4-24GB", "H100-80GB"]

    def extract_colocated_jobs(self, pod_spec: Dict[str, Any]) -> List[str]:
        """Query running GPU pods on candidate nodes to find colocated job types."""
        try:
            from kubernetes import client as k8s_client, config as k8s_config
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()

            v1 = k8s_client.CoreV1Api()
            # Get running GPU pods
            pods = v1.list_pod_for_all_namespaces(
                field_selector="status.phase=Running"
            ).items

            colocated = []
            for pod in pods:
                # Check if this pod uses GPUs
                for container in pod.spec.containers or []:
                    requests = (container.resources.requests or {}) if container.resources else {}
                    if any("nvidia.com/gpu" in k for k in requests):
                        annotations = pod.metadata.annotations or {}
                        job_type = annotations.get(
                            "app.kubernetes.io/job-type",
                            (pod.metadata.labels or {}).get("app.kubernetes.io/job-type", "")
                        )
                        if job_type:
                            colocated.append(job_type)
                        break
            return colocated

        except Exception as e:
            logging.debug(f"Failed to query colocated jobs: {e}")
            return []

    async def call_scheduler(self, job_type: str, features: Dict[str, Any],
                           candidates: List[str], colocated: List[str]) -> Optional[Dict[str, Any]]:
        """Call the scheduler service for scoring."""
        try:
            request_data = {
                "job_type": job_type,
                "features": features,
                "candidate_gpu_types": candidates,
                "colocated_job_types": colocated
            }

            response = await self.http_client.post(
                f"{self.config.scheduler_url}/v1/score",
                json=request_data
            )

            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"Scheduler returned {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logging.error(f"Error calling scheduler: {e}")
            return None

    def get_priority_class(self, score: float) -> str:
        """Map score to priority class name."""
        for bucket in sorted(self.config.priority_buckets, key=lambda x: x["minScore"], reverse=True):
            if score >= bucket["minScore"]:
                return bucket["priorityClassName"]

        return "gpu-default"  # Fallback

    def create_json_patch(self, pod: Dict[str, Any], score_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create JSON Patch operations to mutate the Pod."""
        patches = []

        # Add priority class
        priority_score = score_response.get("priority_score", 50)
        priority_class = self.get_priority_class(priority_score)

        patches.append({
            "op": "add",
            "path": "/spec/priorityClassName",
            "value": priority_class
        })

        # Add GPU selector (nodeSelector)
        gpu_selector = score_response.get("gpu_selector", {})
        node_selector = gpu_selector.get("nodeSelector", {})

        if node_selector:
            if "nodeSelector" in pod.get("spec", {}):
                # Merge with existing nodeSelector
                for key, value in node_selector.items():
                    patches.append({
                        "op": "add",
                        "path": f"/spec/nodeSelector/{key}",
                        "value": value
                    })
            else:
                patches.append({
                    "op": "add",
                    "path": "/spec/nodeSelector",
                    "value": node_selector
                })

        # Add tolerations
        tolerations = gpu_selector.get("tolerations", [])
        if tolerations:
            existing_tolerations = pod.get("spec", {}).get("tolerations", [])
            new_tolerations = existing_tolerations + tolerations

            patches.append({
                "op": "add",
                "path": "/spec/tolerations",
                "value": new_tolerations
            })

        # Add pod anti-affinity if needed
        avoid_list = score_response.get("avoid_co_locate_with", [])
        if avoid_list:
            anti_affinity = {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": []
                }
            }

            for job_type in avoid_list:
                term = {
                    "weight": 100,
                    "podAffinityTerm": {
                        "labelSelector": {
                            "matchExpressions": [{
                                "key": "app.kubernetes.io/job-type",
                                "operator": "In",
                                "values": [job_type]
                            }]
                        },
                        "topologyKey": "kubernetes.io/hostname"
                    }
                }
                anti_affinity["podAntiAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"].append(term)

            if pod.get("spec", {}).get("affinity"):
                patches.append({
                    "op": "add",
                    "path": "/spec/affinity/podAntiAffinity",
                    "value": anti_affinity["podAntiAffinity"]
                })
            else:
                patches.append({
                    "op": "add",
                    "path": "/spec/affinity",
                    "value": anti_affinity
                })

        # Add scheduler annotations for debugging
        annotations = pod.get("metadata", {}).get("annotations", {})
        debug_info = {
            "scheduler.radix.ai/priority-score": str(priority_score),
            "scheduler.radix.ai/chosen-gpu": score_response.get("terms", {}).get("chosen_gpu", "unknown"),
            "scheduler.radix.ai/decision-timestamp": str(int(asyncio.get_event_loop().time()))
        }

        for key, value in debug_info.items():
            patches.append({
                "op": "add",
                "path": f"/metadata/annotations/{key.replace('/', '~1')}",
                "value": value
            })

        return patches

    async def mutate_pod(self, admission_request: Dict[str, Any]) -> AdmissionResponse:
        """Mutate a Pod based on scheduler recommendations."""
        uid = admission_request["uid"]
        pod = admission_request["object"]

        # Default response (allow without mutation)
        response = AdmissionResponse(uid=uid, allowed=True)

        try:
            # Check if this is a GPU-requesting pod
            spec = pod.get("spec", {})
            metadata = pod.get("metadata", {})
            annotations = metadata.get("annotations", {})

            # Extract features
            features = self.extract_job_features(spec, annotations)
            job_type = features.get("job_type", "unknown")

            # Skip if not a GPU job or if mutations disabled
            if not self.config.enable_mutation or job_type == "unknown":
                return response

            # Get GPU candidates and colocated jobs
            candidates = self.get_candidate_gpu_types()
            colocated = self.extract_colocated_jobs(spec)

            # Call scheduler
            score_response = await self.call_scheduler(job_type, features, candidates, colocated)

            if score_response:
                # Create JSON patch
                patches = self.create_json_patch(pod, score_response)

                if patches:
                    patch_json = json.dumps(patches)
                    patch_b64 = base64.b64encode(patch_json.encode()).decode()

                    response.patch = patch_b64
                    response.patchType = "JSONPatch"

                    logging.info(f"Mutated pod {metadata.get('name', 'unknown')} with {len(patches)} patches")

        except Exception as e:
            logging.error(f"Error mutating pod: {e}")
            # Still allow the pod to be created

        return response


# FastAPI app
config = WebhookConfig()
webhook = MutatingWebhook(config)

app = FastAPI(
    title="GPU Scheduler Webhook",
    description="Mutating admission webhook for information-theoretic GPU scheduling",
    version="1.0.0"
)


@app.post("/mutate")
async def mutate(admission_review: AdmissionReview):
    """Handle mutating admission webhook requests."""
    try:
        # Process the mutation
        response = await webhook.mutate_pod(admission_review.request)

        # Return AdmissionReview with response
        return {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": response.dict()
        }

    except Exception as e:
        logging.error(f"Webhook error: {e}")
        # Return allowing admission without mutation
        return {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": admission_review.request.get("uid", ""),
                "allowed": True
            }
        }


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/readyz")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Test scheduler connectivity
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{config.scheduler_url}/healthz")
            if response.status_code == 200:
                return {"status": "ready", "scheduler": "connected"}
            else:
                return {"status": "ready", "scheduler": "disconnected"}
    except Exception:
        return {"status": "ready", "scheduler": "disconnected"}


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await webhook.close()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8443)
