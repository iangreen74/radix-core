#!/usr/bin/env python3
"""
Radix Cluster Agent - Main Entry Point

This agent runs on external GPU clusters and executes containerized workloads
submitted via the Radix control plane.

Execution Mode:
- docker: Run jobs as Docker containers (requires Docker socket access)

Future modes (not yet implemented):
- k8s: Submit jobs to Kubernetes cluster
- slurm: Submit jobs to Slurm scheduler
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Agent version
AGENT_VERSION = '1.0.0'


class ClusterAgent:
    """Radix Cluster Agent - executes jobs from control plane."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent.
        
        Args:
            config: Agent configuration dict
        """
        self.api_base = config['api_base']
        self.cluster_id = config['cluster_id']
        self.tenant_id = config.get('tenant_id', 'demo-tenant')
        self.poll_interval = config.get('poll_interval', 30)
        self.jobs_workdir = Path(config.get('jobs_workdir', '/tmp/radix-jobs'))
        self.execution_mode = config.get('execution_mode', 'docker')
        self.token_file = Path(config.get('token_file', '/var/lib/radix/cluster_token'))
        
        # Handle token: one-time exchange, env var, or persisted file
        one_time_token = config.get('one_time_token')
        cluster_token = config.get('cluster_token')
        
        if cluster_token:
            # Use provided long-lived token (existing behavior)
            self.cluster_token = cluster_token
            logger.info("Using cluster token from environment variable")
        elif one_time_token:
            # Exchange one-time token for long-lived token
            logger.info("One-time token provided, exchanging for long-lived token...")
            self.cluster_token = self._exchange_onboarding_token(one_time_token)
            # Persist token to disk
            self._persist_token(self.cluster_token)
            logger.info("Successfully exchanged and persisted cluster token")
        elif self.token_file.exists():
            # Load token from persisted file
            logger.info(f"Loading cluster token from {self.token_file}")
            self.cluster_token = self._load_token()
        else:
            raise ValueError("No cluster token available. Provide RADIX_CLUSTER_TOKEN or RADIX_ONE_TIME_TOKEN, or ensure token file exists.")
        
        # Create jobs workdir
        self.jobs_workdir.mkdir(parents=True, exist_ok=True)
        
        # Collect capabilities
        self.capabilities = self._collect_capabilities()
        
        logger.info(f"Radix Cluster Agent v{AGENT_VERSION}")
        logger.info(f"Cluster ID: {self.cluster_id}")
        logger.info(f"API Base: {self.api_base}")
        logger.info(f"Execution Mode: {self.execution_mode}")
        logger.info(f"Token present: {bool(self.cluster_token)}")
        logger.info(f"Capabilities: {self.capabilities}")
    
    def _collect_capabilities(self) -> Dict[str, Any]:
        """
        Collect cluster capabilities.
        
        Returns:
            Dict with cpu_count, memory_gb, gpus, etc.
        """
        capabilities = {}
        
        # CPU count
        try:
            import psutil
            capabilities['cpu_count'] = psutil.cpu_count(logical=True)
            capabilities['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            capabilities['cpu_count'] = os.cpu_count() or 1
            capabilities['memory_gb'] = 0
        
        # GPU detection via nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            memory = parts[1].strip()
                            gpus.append({'name': name, 'memory': memory})
                capabilities['gpus'] = gpus
                capabilities['gpu_count'] = len(gpus)
            else:
                capabilities['gpus'] = []
                capabilities['gpu_count'] = 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            capabilities['gpus'] = []
            capabilities['gpu_count'] = 0
        
        return capabilities
    
    def _exchange_onboarding_token(self, one_time_token: str) -> str:
        """
        Exchange one-time onboarding token for long-lived cluster token.
        
        Args:
            one_time_token: One-time token from cluster registration
        
        Returns:
            Long-lived cluster token
        
        Raises:
            RuntimeError: If exchange fails
        """
        url = f"{self.api_base}/v1/clusters/{self.cluster_id}/auth/exchange"
        headers = {
            'X-Radix-One-Time-Token': one_time_token,
            'X-Radix-Tenant-Id': self.tenant_id,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(url, headers=headers, timeout=30)
            
            # Compatibility fallback: retry with header if 401 with specific message
            if response.status_code == 401:
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', '')
                    if 'Missing X-Radix-One-Time-Token' in error_message:
                        logger.warning("Backend reported missing header, retrying with explicit header")
                        # Header already set, but retry once in case of transient issue
                        response = requests.post(url, headers=headers, timeout=30)
                except (ValueError, KeyError):
                    pass  # Not JSON or no message field
            
            response.raise_for_status()
            data = response.json()
            cluster_token = data.get('cluster_token')
            if not cluster_token:
                raise RuntimeError("Exchange response missing cluster_token")
            
            logger.info("Successfully exchanged onboarding token for cluster token")
            return cluster_token
            
        except requests.exceptions.HTTPError as e:
            # Enhanced error reporting
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = f": {error_data.get('message', e.response.text)}"
            except:
                error_detail = f": {e.response.text if hasattr(e.response, 'text') else str(e)}"
            
            logger.error(f"Failed to exchange onboarding token (HTTP {e.response.status_code}){error_detail}")
            raise RuntimeError(f"Token exchange failed (HTTP {e.response.status_code}){error_detail}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to exchange onboarding token: {e}")
            raise RuntimeError(f"Token exchange failed: {e}")
    
    def _persist_token(self, token: str):
        """
        Persist cluster token to disk for future restarts.
        
        Args:
            token: Cluster token to persist
        """
        try:
            # Create directory if needed
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write token with restrictive permissions
            self.token_file.write_text(token)
            self.token_file.chmod(0o600)  # Owner read/write only
            
            logger.info(f"Persisted cluster token to {self.token_file}")
        except Exception as e:
            logger.error(f"Failed to persist token: {e}")
            # Don't raise - agent can still run with in-memory token
    
    def _load_token(self) -> str:
        """
        Load cluster token from persisted file.
        
        Returns:
            Cluster token
        
        Raises:
            RuntimeError: If token file cannot be read
        """
        try:
            token = self.token_file.read_text().strip()
            if not token:
                raise RuntimeError("Token file is empty")
            return token
        except Exception as e:
            logger.error(f"Failed to load token from {self.token_file}: {e}")
            raise RuntimeError(f"Cannot load token: {e}")
    
    def _build_auth_headers(self) -> Dict[str, str]:
        """
        Build authentication headers for cluster API requests.
        
        Returns:
            Dict with X-Radix-Cluster-Token, X-Radix-Tenant-Id, Content-Type
        """
        return {
            'X-Radix-Cluster-Token': self.cluster_token,
            'X-Radix-Tenant-Id': self.tenant_id,
            'Content-Type': 'application/json'
        }
    
    def _request_with_logging(self, session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with detailed error logging.
        
        Args:
            session: Requests session
            method: HTTP method
            url: Full URL
            **kwargs: Additional request arguments
        
        Returns:
            Response object
        
        Raises:
            requests.HTTPError: On HTTP errors (after logging details)
            Exception: On any other request errors (after logging details)
        """
        try:
            response = session.request(method, url, **kwargs)
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                status = "N/A"
                body = "No response body"
                if e.response is not None:
                    status = e.response.status_code
                    try:
                        text = e.response.text
                        body = text[:1000] if text else body
                    except Exception:
                        pass
                logger.error(
                    "HTTP %s %s failed with status %s: %s",
                    method, url, status, body,
                )
                logger.error(
                    "HTTPError type=%s, message=%s",
                    type(e).__name__,
                    str(e),
                )
                logger.error("Traceback:\n%s", traceback.format_exc())
                raise
            return response
        except Exception as e:
            logger.error(
                "Request %s %s raised %s: %s",
                method, url, type(e).__name__, str(e),
            )
            logger.error("Traceback:\n%s", traceback.format_exc())
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _api_request(self, method: str, path: str, **kwargs) -> requests.Response:
        """
        Make API request with retry logic.
        
        Args:
            method: HTTP method
            path: API path
            **kwargs: Additional request arguments
        
        Returns:
            Response object
        """
        url = f"{self.api_base}{path}"
        headers = kwargs.pop('headers', {})
        headers.update(self._build_auth_headers())
        
        session = requests.Session()
        return self._request_with_logging(session, method, url, headers=headers, timeout=30, **kwargs)
    
    def send_heartbeat(self):
        """Send heartbeat to control plane."""
        try:
            logger.info(f"Sending heartbeat to {self.api_base}/v1/clusters/{self.cluster_id}/heartbeat")
            resp = self._api_request(
                'POST',
                f'/v1/clusters/{self.cluster_id}/heartbeat',
                json={
                    'status': 'active',
                    'capabilities': self.capabilities,
                    'agent_version': AGENT_VERSION
                }
            )
            logger.info("Heartbeat succeeded with status %s", resp.status_code)
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    def poll_job(self) -> Optional[Dict[str, Any]]:
        """
        Poll for next job.
        
        Returns:
            Job dict or None if no jobs available
        """
        try:
            logger.info(f"Polling jobs from {self.api_base}/v1/clusters/{self.cluster_id}/jobs/poll")
            response = self._api_request(
                'POST',
                f'/v1/clusters/{self.cluster_id}/jobs/poll',
                json={'max_jobs': 1}
            )
            data = response.json()
            jobs = data.get('jobs', [])
            if jobs:
                job = jobs[0]
                logger.info(f"Received job {job.get('job_id')} (kind={job.get('job_kind')})")
                return job
            else:
                logger.info("No jobs available")
                return None
        except Exception as e:
            logger.error(f"Failed to poll jobs: {e}")
            return None
    
    def execute_job(self, job: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[str]]:
        """
        Execute a job.
        
        Args:
            job: Job specification
        
        Returns:
            Tuple of (status, output_payload, metrics, error_message)
        """
        job_id = job['job_id']
        job_kind = job.get('job_kind', 'generic')
        logger.info(f"Executing job {job_id} (kind={job_kind})")
        
        # Route to specialized handlers for specific job kinds
        if job_kind == 'resnet50_benchmark':
            return self._execute_resnet50_benchmark_job(job)
        elif self.execution_mode == 'docker':
            return self._execute_docker_job(job)
        else:
            error_msg = f"Execution mode '{self.execution_mode}' not implemented"
            logger.error(error_msg)
            return 'failed', {}, {}, error_msg
    
    def _execute_resnet50_benchmark_job(self, job: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[str]]:
        """
        Execute ResNet50 benchmark job.
        
        Args:
            job: Job specification with benchmark parameters
        
        Returns:
            Tuple of (status, output_payload, metrics, error_message)
        """
        job_id = job['job_id']
        params = job.get('params', {})
        epochs = params.get('epochs', 1)
        batch_size = params.get('batch_size', 64)
        
        # Default ResNet50 benchmark image
        image = job.get('image', 'iangreen74/radix-resnet50-benchmark:latest')
        
        # Get orchestration spec
        orchestration = job.get('orchestration', {})
        num_gpus = orchestration.get('num_gpus', 1)
        launcher = orchestration.get('launcher', 'torchrun')
        
        logger.info(f"Running ResNet50 benchmark job {job_id}")
        logger.info(f"Image: {image}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        logger.info(f"Orchestration: num_gpus={num_gpus}, launcher={launcher}")
        
        # Build base command
        base_command = [
            'python', 'train_resnet50.py',
            '--epochs', str(epochs),
            '--batch-size', str(batch_size)
        ]
        
        # Wrap with torchrun for multi-GPU
        if num_gpus > 1:
            command = [
                'torchrun',
                f'--nproc_per_node={num_gpus}',
                '--standalone'
            ] + base_command
            logger.info(f"Multi-GPU execution: wrapping with torchrun (nproc_per_node={num_gpus})")
        else:
            command = base_command
        
        # Build docker run command
        docker_cmd = [
            'docker', 'run',
            '--rm',
            '-w', '/app'
        ]
        
        # Add GPU support if available
        if self.capabilities.get('gpu_count', 0) > 0:
            docker_cmd.extend(['--gpus', 'all'])
            logger.info("GPU support enabled for benchmark")
        else:
            logger.warning("No GPU available for benchmark")
        
        # Add image and command
        docker_cmd.append(image)
        docker_cmd.extend(command)
        
        logger.info(f"Running: {' '.join(docker_cmd)}")
        
        try:
            # Execute container
            start_time = time.time()
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout for benchmarks
            )
            runtime_seconds = time.time() - start_time
            
            # Extract throughput from stdout (look for "images/sec" pattern)
            throughput = None
            for line in result.stdout.split('\n'):
                if 'images/sec' in line.lower():
                    try:
                        # Try to extract number before "images/sec"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'images/sec' in part.lower() and i > 0:
                                throughput = float(parts[i-1])
                                break
                    except (ValueError, IndexError):
                        pass
            
            # Build metrics (use duration_seconds for dashboard compatibility)
            metrics = {
                'duration_seconds': round(runtime_seconds, 2),
                'throughput_images_per_sec': round(throughput, 2) if throughput else None,
                'exit_code': result.returncode
            }
            
            if throughput:
                logger.info(f"Extracted throughput: {throughput:.2f} images/sec")
            else:
                logger.warning("Could not extract throughput from benchmark output")
            
            # Build output payload
            output_payload = {
                'stdout': result.stdout[-5000:] if result.stdout else '',  # Last 5000 chars
                'stderr': result.stderr[-2000:] if result.stderr else ''
            }
            
            # Determine status
            if result.returncode == 0:
                status = 'succeeded'
                error_message = None
                logger.info(f"ResNet50 benchmark job {job_id} completed successfully in {runtime_seconds:.2f}s")
            else:
                status = 'failed'
                error_message = f"Benchmark exited with code {result.returncode}"
                if result.stderr:
                    error_message += f": {result.stderr[:500]}"
                logger.error(f"ResNet50 benchmark job {job_id} failed: {error_message}")
            
            return status, output_payload, metrics, error_message
        
        except subprocess.TimeoutExpired:
            logger.error(f"ResNet50 benchmark job {job_id} timed out after 2 hours")
            return 'failed', {}, {}, 'Benchmark execution timed out after 2 hours'
        
        except Exception as e:
            logger.error(f"ResNet50 benchmark job {job_id} failed: {e}")
            return 'failed', {}, {}, str(e)
    
    def _execute_docker_job(self, job: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[str]]:
        """
        Execute job as Docker container.
        
        Args:
            job: Job specification
        
        Returns:
            Tuple of (status, output_payload, metrics, error_message)
        """
        job_id = job['job_id']
        image = job.get('image')
        command = job.get('command', [])
        args = job.get('args', [])
        env = job.get('env', {})
        input_payload = job.get('input_payload', {})
        
        # Get orchestration spec
        orchestration = job.get('orchestration', {})
        num_gpus = orchestration.get('num_gpus', 1)
        launcher = orchestration.get('launcher', 'torchrun')
        
        if not image:
            return 'failed', {}, {}, 'No image specified'
        
        logger.info(f"Orchestration: num_gpus={num_gpus}, launcher={launcher}")
        
        # Create job directory
        job_dir = self.jobs_workdir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write input payload
            input_file = job_dir / 'input.json'
            with open(input_file, 'w') as f:
                json.dump(input_payload, f)
            
            # Build docker run command
            docker_cmd = [
                'docker', 'run',
                '--rm',
                '-v', f'{job_dir}:/workspace',
                '-w', '/workspace'
            ]
            
            # Add GPU support if available
            if self.capabilities.get('gpu_count', 0) > 0:
                docker_cmd.extend(['--gpus', 'all'])
            
            # Add environment variables
            for key, value in env.items():
                docker_cmd.extend(['-e', f'{key}={value}'])
            
            # Add image
            docker_cmd.append(image)
            
            # Wrap command with torchrun for multi-GPU if needed
            if num_gpus > 1 and command:
                # Wrap with torchrun
                wrapped_command = [
                    'torchrun',
                    f'--nproc_per_node={num_gpus}',
                    '--standalone'
                ] + command
                docker_cmd.extend(wrapped_command)
                logger.info(f"Multi-GPU execution: wrapping with torchrun (nproc_per_node={num_gpus})")
            else:
                # Add command and args as-is
                if command:
                    docker_cmd.extend(command)
            
            if args:
                docker_cmd.extend(args)
            
            logger.info(f"Running: {' '.join(docker_cmd)}")
            
            # Execute container
            start_time = time.time()
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            runtime_seconds = time.time() - start_time
            
            # Determine output payload
            output_file = job_dir / 'output.json'
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        output_payload = json.load(f)
                except json.JSONDecodeError:
                    output_payload = {'error': 'Invalid output.json'}
            else:
                # Try to parse stdout as JSON
                try:
                    output_payload = json.loads(result.stdout)
                except json.JSONDecodeError:
                    # Wrap stdout/stderr
                    output_payload = {
                        'stdout': result.stdout[:10000],  # Limit size
                        'stderr': result.stderr[:10000]
                    }
            
            # Build metrics
            metrics = {
                'runtime_seconds': round(runtime_seconds, 2),
                'exit_code': result.returncode
            }
            
            # Determine status
            if result.returncode == 0:
                status = 'succeeded'
                error_message = None
            else:
                status = 'failed'
                error_message = f"Container exited with code {result.returncode}"
                if result.stderr:
                    error_message += f": {result.stderr[:500]}"
            
            logger.info(f"Job {job_id} completed with status {status}")
            
            return status, output_payload, metrics, error_message
        
        except subprocess.TimeoutExpired:
            logger.error(f"Job {job_id} timed out")
            return 'failed', {}, {}, 'Job execution timed out after 1 hour'
        
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            return 'failed', {}, {}, str(e)
        
        finally:
            # Cleanup job directory
            try:
                shutil.rmtree(job_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup job directory: {e}")
    
    def complete_job(self, job_id: str, status: str, output_payload: Dict[str, Any],
                    metrics: Dict[str, Any], error_message: Optional[str]):
        """
        Mark job as complete.
        
        Args:
            job_id: Job identifier
            status: Job status (succeeded, failed)
            output_payload: Output data
            metrics: Job metrics
            error_message: Optional error message
        """
        try:
            self._api_request(
                'POST',
                f'/v1/clusters/{self.cluster_id}/jobs/{job_id}/complete',
                json={
                    'tenant_id': self.tenant_id,
                    'status': status,
                    'output_payload': output_payload,
                    'metrics': metrics,
                    'error_message': error_message
                }
            )
            logger.info(f"Job {job_id} marked as {status}")
        except Exception as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
    
    def run(self):
        """Main agent loop."""
        logger.info("Starting agent main loop")
        
        # Send initial heartbeat
        self.send_heartbeat()
        
        last_heartbeat = time.time()
        
        while True:
            try:
                # Send heartbeat every 5 minutes
                if time.time() - last_heartbeat > 60:
                    self.send_heartbeat()
                    last_heartbeat = time.time()
                
                # Poll for job
                job = self.poll_job()
                
                if job:
                    job_id = job['job_id']
                    logger.info(f"Received job {job_id}")
                    
                    # Execute job
                    status, output_payload, metrics, error_message = self.execute_job(job)
                    
                    # Complete job
                    self.complete_job(job_id, status, output_payload, metrics, error_message)
                else:
                    # No job available, sleep
                    time.sleep(self.poll_interval)
            
            except KeyboardInterrupt:
                logger.info("Agent stopped by user")
                break
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(self.poll_interval)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Radix Cluster Agent')
    parser.add_argument('--api-base', help='API base URL', default=os.environ.get('RADIX_API_BASE'))
    parser.add_argument('--cluster-id', help='Cluster ID', default=os.environ.get('RADIX_CLUSTER_ID'))
    parser.add_argument('--cluster-token', help='Long-lived cluster token', default=os.environ.get('RADIX_CLUSTER_TOKEN'))
    parser.add_argument('--one-time-token', help='One-time onboarding token', default=os.environ.get('RADIX_ONE_TIME_TOKEN'))
    parser.add_argument('--tenant-id', help='Tenant ID', default=os.environ.get('RADIX_TENANT_ID', 'demo-tenant'))
    parser.add_argument('--poll-interval', type=int, help='Poll interval in seconds', 
                       default=int(os.environ.get('RADIX_POLL_INTERVAL_SECONDS', '30')))
    parser.add_argument('--jobs-workdir', help='Jobs working directory',
                       default=os.environ.get('RADIX_JOBS_WORKDIR', '/tmp/radix-jobs'))
    parser.add_argument('--execution-mode', help='Execution mode (docker, k8s, slurm)',
                       default=os.environ.get('RADIX_EXECUTION_MODE', 'docker'))
    parser.add_argument('--token-file', help='Token persistence file',
                       default=os.environ.get('RADIX_TOKEN_FILE', '/var/lib/radix/cluster_token'))
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.api_base:
        logger.error("RADIX_API_BASE is required")
        sys.exit(1)
    if not args.cluster_id:
        logger.error("RADIX_CLUSTER_ID is required")
        sys.exit(1)
    
    # Token validation: at least one of cluster_token, one_time_token, or token_file must be available
    token_file_path = Path(args.token_file)
    if not args.cluster_token and not args.one_time_token and not token_file_path.exists():
        logger.error("No authentication available. Provide RADIX_CLUSTER_TOKEN, RADIX_ONE_TIME_TOKEN, or ensure token file exists at %s", args.token_file)
        sys.exit(1)
    
    # Build config
    config = {
        'api_base': args.api_base,
        'cluster_id': args.cluster_id,
        'cluster_token': args.cluster_token,
        'one_time_token': args.one_time_token,
        'tenant_id': args.tenant_id,
        'poll_interval': args.poll_interval,
        'jobs_workdir': args.jobs_workdir,
        'execution_mode': args.execution_mode,
        'token_file': args.token_file
    }
    
    # Create and run agent
    agent = ClusterAgent(config)
    agent.run()


if __name__ == '__main__':
    main()
