#!/usr/bin/env python3
"""
Radix Kubernetes Agent v1
Polls control plane for jobs, executes them, and reports metrics.
"""
import os
import time
import json
import subprocess
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    exit(1)

# Configuration from environment
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.vaultscaler.com")
CLUSTER_ID = os.environ["CLUSTER_ID"]
AGENT_API_KEY = os.environ["AGENT_API_KEY"]
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))
METRICS_INTERVAL_SECONDS = int(os.environ.get("METRICS_INTERVAL_SECONDS", "30"))

print(f"Radix Agent v1 starting for cluster: {CLUSTER_ID}")
print(f"API Base URL: {API_BASE_URL}")
print(f"Poll interval: {POLL_INTERVAL_SECONDS}s, Metrics interval: {METRICS_INTERVAL_SECONDS}s")


def _headers() -> Dict[str, str]:
    """Return headers for API requests."""
    return {
        "X-Radix-Agent-Key": AGENT_API_KEY,
        "Content-Type": "application/json"
    }


def poll_next_job() -> Optional[Dict]:
    """Poll control plane for next pending job."""
    url = f"{API_BASE_URL}/v1/agent/jobs/next"
    params = {"cluster_id": CLUSTER_ID}
    
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=10)
        
        if resp.status_code == 204:
            # No pending jobs
            return None
        
        resp.raise_for_status()
        return resp.json()
    
    except requests.RequestException as e:
        print(f"Error polling for jobs: {e}")
        return None


def complete_job(job_id: str, tenant_id: str, status: str, exit_code: int, logs: str, output_payload: Dict = None) -> None:
    """Report job completion to control plane."""
    url = f"{API_BASE_URL}/v1/agent/jobs/{job_id}/complete"
    
    payload = {
        "tenant_id": tenant_id,
        "cluster_id": CLUSTER_ID,
        "status": status,
        "exit_code": exit_code,
        "logs": logs[:10000] if logs else "",  # Limit log size
        "output_payload": output_payload or {}
    }
    
    try:
        resp = requests.post(url, headers=_headers(), data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        print(f"Job {job_id} marked as {status}")
    
    except requests.RequestException as e:
        print(f"Error completing job {job_id}: {e}")


def collect_gpu_metrics() -> Optional[float]:
    """Collect GPU utilization using nvidia-smi (best-effort)."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=3
        ).decode("utf-8", "ignore")
        
        values = [int(x.strip()) for x in output.splitlines() if x.strip()]
        
        if values:
            return sum(values) / len(values)
    
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def push_metrics() -> None:
    """Push cluster metrics to control plane."""
    url = f"{API_BASE_URL}/v1/agent/metrics"
    
    gpu_util = collect_gpu_metrics()
    
    metric = {
        "cluster_id": CLUSTER_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_utilization_percent": gpu_util,
        "running_jobs": None,  # TODO: Track from K8s
        "pending_jobs": None,  # TODO: Track from K8s
    }
    
    try:
        resp = requests.post(url, headers=_headers(), data=json.dumps(metric), timeout=10)
        
        if resp.ok:
            print(f"Metrics pushed: GPU={gpu_util}%")
        else:
            print(f"Metrics push failed: {resp.status_code}")
    
    except requests.RequestException as e:
        print(f"Error pushing metrics: {e}")


def run_job(job: Dict) -> Tuple[str, int, str]:
    """
    Execute a job (simple shell command for Phase 1).
    
    Returns:
        (status, exit_code, logs)
    """
    payload = job.get("payload", {})
    command = payload.get("command")
    
    if not command:
        return ("failed", 1, "No command specified in job payload")
    
    print(f"Executing job {job['job_id']}: {command}")
    
    try:
        # Execute command with timeout
        timeout = payload.get("timeout", 600)
        
        output = subprocess.check_output(
            command,
            shell=True,
            stderr=subprocess.STDOUT,
            timeout=timeout
        )
        
        logs = output.decode("utf-8", "ignore")
        print(f"Job completed successfully")
        return ("completed", 0, logs)
    
    except subprocess.CalledProcessError as e:
        logs = e.output.decode("utf-8", "ignore") if e.output else str(e)
        print(f"Job failed with exit code {e.returncode}")
        return ("failed", e.returncode, logs)
    
    except subprocess.TimeoutExpired as e:
        logs = f"Job timed out after {e.timeout}s"
        print(logs)
        return ("failed", 124, logs)
    
    except Exception as e:
        logs = f"Job execution error: {str(e)}"
        print(logs)
        return ("failed", 1, logs)


def main_loop() -> None:
    """Main agent loop: poll for jobs and push metrics."""
    last_metrics_time = 0.0
    
    print("Agent loop started")
    
    while True:
        now = time.time()
        
        # Push metrics periodically
        if now - last_metrics_time >= METRICS_INTERVAL_SECONDS:
            push_metrics()
            last_metrics_time = now
        
        # Poll for jobs
        job = poll_next_job()
        
        if job:
            job_id = job["job_id"]
            tenant_id = job["tenant_id"]
            
            print(f"Received job: {job_id}")
            
            # Execute job
            status, exit_code, logs = run_job(job)
            
            # Report completion
            complete_job(job_id, tenant_id, status, exit_code, logs)
        
        # Wait before next poll
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nAgent stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)
