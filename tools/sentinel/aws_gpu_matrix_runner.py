#!/usr/bin/env python3
"""
AWS GPU Matrix Runner for Sentinel

Provisions NVIDIA + AMD GPU instances, runs full onboarding + smoke tests,
generates evidence pack, and promotes digest if all tests pass.
"""

import os
import sys
import time
import json
import yaml
import boto3
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from aws_ssm import SSMHelper
from reporting.evidence_pack import EvidencePack, create_matrix_result
from auth import get_jwt_token, get_api_base


def load_invariants() -> Dict[str, Any]:
    """Load ops/invariants.yaml configuration."""
    repo_root = Path(__file__).parent.parent.parent
    invariants_path = repo_root / 'ops' / 'invariants.yaml'
    
    with open(invariants_path, 'r') as f:
        return yaml.safe_load(f)


def get_matrix_config(environment: str = 'dev') -> Dict[str, Any]:
    """Get GPU matrix configuration from invariants.
    
    Args:
        environment: Environment name
    
    Returns:
        Matrix configuration dictionary
    """
    invariants = load_invariants()
    env_config = invariants['environments'].get(environment, {})
    agent_promo = env_config.get('agent_promotion', {})
    
    return {
        'ssm_param': agent_promo.get('ssm_approved_digest_param', '/radix/dev/agent_digest_approved'),
        'ssm_instance_profile': agent_promo.get('ssm_instance_profile', 'RadixSentinelSSMProfile'),
        'region': env_config.get('region', 'us-west-2'),
        'nvidia': agent_promo.get('gpu_matrix', {}).get('nvidia', {}),
        'amd': agent_promo.get('gpu_matrix', {}).get('amd', {}),
        'timeouts': agent_promo.get('timeouts', {}),
        'cleanup': agent_promo.get('cleanup', {})
    }


class GPUMatrixRunner:
    """Runs GPU matrix tests and generates evidence pack."""
    
    def __init__(
        self,
        run_id: str,
        git_sha: str,
        candidate_digest: str,
        environment: str = 'dev'
    ):
        """Initialize GPU matrix runner.
        
        Args:
            run_id: Unique run identifier
            git_sha: Git commit SHA
            candidate_digest: Candidate agent image digest
            environment: Environment name (dev or prod)
        """
        self.run_id = run_id
        self.git_sha = git_sha
        self.candidate_digest = candidate_digest
        self.environment = environment
        
        # Load configuration from invariants
        self.config = get_matrix_config(environment)
        self.region = self.config['region']
        
        # Get API base and JWT token automatically
        self.api_base = get_api_base(environment)
        self.jwt_token = get_jwt_token(environment=environment)
        
        print(f"🔧 Initialized GPU Matrix Runner")
        print(f"   Environment: {environment}")
        print(f"   Region: {self.region}")
        print(f"   API Base: {self.api_base}")
        
        self.ec2 = boto3.client('ec2', region_name=self.region)
        self.ssm_helper = SSMHelper(region=self.region)
        
        # Get current approved digest before tests
        self.approved_digest_before = self.ssm_helper.get_parameter(
            self.config['ssm_param']
        )
        
        self.evidence_pack = EvidencePack(run_id, git_sha, candidate_digest, self.approved_digest_before)
        
        self.instances: List[str] = []
    
    def run_matrix(self) -> bool:
        """Run full GPU matrix test.
        
        Returns:
            True if all tests pass
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting GPU Matrix Test")
        print(f"   Run ID: {self.run_id}")
        print(f"   Git SHA: {self.git_sha}")
        print(f"   Candidate: {self.candidate_digest}")
        print(f"{'='*60}\n")
        
        try:
            # Cleanup old instances first
            self.ssm_helper.cleanup_old_instances()
            
            # Run NVIDIA test
            print("\n" + "="*60)
            print("🟢 NVIDIA GPU Test")
            print("="*60)
            nvidia_config = self.config['nvidia']
            nvidia_result = self.run_vendor_test(
                'nvidia',
                nvidia_config['instance_type'],
                nvidia_config['ami_id']
            )
            self.evidence_pack.add_matrix_result(nvidia_result)
            
            # Run AMD test
            print("\n" + "="*60)
            print("🔴 AMD GPU Test")
            print("="*60)
            amd_config = self.config['amd']
            amd_result = self.run_vendor_test(
                'amd',
                amd_config['instance_type'],
                amd_config['ami_id']
            )
            self.evidence_pack.add_matrix_result(amd_result)
            
            # Check if all passed
            all_passed = (
                nvidia_result.get('status') == 'pass' and
                amd_result.get('status') == 'pass'
            )
            
            print(f"\n{'='*60}")
            print(f"📊 Matrix Test Results")
            print(f"   NVIDIA: {'✅ PASS' if nvidia_result.get('status') == 'pass' else '❌ FAIL'}")
            print(f"   AMD: {'✅ PASS' if amd_result.get('status') == 'pass' else '❌ FAIL'}")
            print(f"   Overall: {'✅ ALL PASS' if all_passed else '❌ FAILED'}")
            print(f"{'='*60}\n")
            
            return all_passed
            
        finally:
            # Always cleanup instances
            self.cleanup_instances()
    
    def run_vendor_test(
        self,
        vendor: str,
        instance_type: str,
        ami_id: str
    ) -> Dict[str, Any]:
        """Run test for specific GPU vendor.
        
        Args:
            vendor: GPU vendor (nvidia or amd)
            instance_type: EC2 instance type
            ami_id: AMI ID
        
        Returns:
            Matrix result dictionary
        """
        instance_id = None
        cluster_id = None
        
        try:
            # Launch instance
            print(f"🚀 Launching {vendor.upper()} instance: {instance_type}")
            instance_id = self.launch_instance(vendor, instance_type, ami_id)
            
            if not instance_id:
                return create_matrix_result(
                    vendor, instance_type, 'N/A', ami_id, self.region,
                    status='fail',
                    errors='Failed to launch instance'
                )
            
            # Wait for SSM online
            print(f"⏳ Waiting for SSM agent...")
            if not self.ssm_helper.wait_for_ssm_online(instance_id, SSM_ONLINE_TIMEOUT):
                return create_matrix_result(
                    vendor, instance_type, instance_id, ami_id, self.region,
                    status='fail',
                    errors='SSM agent did not come online'
                )
            
            # Setup instance
            print(f"🔧 Setting up instance...")
            setup_result = self.setup_instance(instance_id, vendor)
            if setup_result['exit_code'] != 0:
                return create_matrix_result(
                    vendor, instance_type, instance_id, ami_id, self.region,
                    status='fail',
                    errors=f"Instance setup failed: {setup_result['stderr']}"
                )
            
            # Get system info
            sys_info = self.get_system_info(instance_id, vendor)
            
            # Create cluster and get onboarding token
            print(f"🔑 Creating cluster and requesting onboarding token...")
            cluster_id, one_time_token = self.create_cluster_and_token(vendor)
            
            if not cluster_id or not one_time_token:
                return create_matrix_result(
                    vendor, instance_type, instance_id, ami_id, self.region,
                    status='fail',
                    errors='Failed to create cluster or get token',
                    **sys_info
                )
            
            # Run onboarding
            print(f"🎯 Running onboarding installer...")
            onboarding_start = time.time()
            onboarding_result = self.run_onboarding(instance_id, cluster_id, one_time_token)
            onboarding_duration = time.time() - onboarding_start
            
            if onboarding_result['exit_code'] != 0:
                return create_matrix_result(
                    vendor, instance_type, instance_id, ami_id, self.region,
                    status='fail',
                    cluster_id=cluster_id,
                    onboarding_duration_seconds=onboarding_duration,
                    errors=f"Onboarding failed: {onboarding_result['stderr']}",
                    **sys_info
                )
            
            # Wait for cluster to become active
            print(f"⏳ Waiting for cluster to become active...")
            active_start = time.time()
            if not self.wait_for_cluster_active(cluster_id):
                return create_matrix_result(
                    vendor, instance_type, instance_id, ami_id, self.region,
                    status='fail',
                    cluster_id=cluster_id,
                    onboarding_duration_seconds=onboarding_duration,
                    time_to_active_seconds=time.time() - active_start,
                    token_exchange_status='unknown',
                    cluster_active_status='timeout',
                    errors='Cluster did not become active',
                    remediation='Check agent logs for token exchange errors. Verify network connectivity.',
                    **sys_info
                )
            
            time_to_active = time.time() - active_start
            
            # Submit smoke job
            print(f"🔥 Submitting GPU smoke job...")
            job_start = time.time()
            job_id, job_status = self.submit_and_wait_for_job(cluster_id)
            job_duration = time.time() - job_start
            
            if job_status != 'completed':
                return create_matrix_result(
                    vendor, instance_type, instance_id, ami_id, self.region,
                    status='fail',
                    cluster_id=cluster_id,
                    onboarding_duration_seconds=onboarding_duration,
                    time_to_active_seconds=time_to_active,
                    token_exchange_status='success',
                    cluster_active_status='active',
                    smoke_job_id=job_id,
                    smoke_job_status=job_status,
                    smoke_job_duration_seconds=job_duration,
                    errors=f'Job did not complete successfully: {job_status}',
                    remediation='Check job logs. Verify GPU is accessible to container runtime.',
                    **sys_info
                )
            
            # Success!
            print(f"✅ {vendor.upper()} test PASSED")
            return create_matrix_result(
                vendor, instance_type, instance_id, ami_id, self.region,
                status='pass',
                cluster_id=cluster_id,
                onboarding_duration_seconds=onboarding_duration,
                time_to_active_seconds=time_to_active,
                token_exchange_status='success',
                cluster_active_status='active',
                smoke_job_id=job_id,
                smoke_job_status=job_status,
                smoke_job_duration_seconds=job_duration,
                **sys_info
            )
            
        except Exception as e:
            print(f"❌ {vendor.upper()} test FAILED: {e}")
            import traceback
            return create_matrix_result(
                vendor, instance_type, instance_id or 'N/A', ami_id, self.region,
                status='fail',
                cluster_id=cluster_id,
                errors=f'Unexpected error: {str(e)}\n{traceback.format_exc()}'
            )
        
        finally:
            # Cleanup cluster
            if cluster_id:
                self.delete_cluster(cluster_id)
    
    def launch_instance(self, vendor: str, instance_type: str, ami_id: str) -> Optional[str]:
        """Launch EC2 instance.
        
        Args:
            vendor: GPU vendor
            instance_type: Instance type
            ami_id: AMI ID
        
        Returns:
            Instance ID or None
        """
        try:
            # Get IAM instance profile from config
            iam_instance_profile = self.config['ssm_instance_profile']
            
            response = self.ec2.run_instances(
                ImageId=ami_id,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                IamInstanceProfile={'Name': iam_instance_profile},
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'radix-sentinel-{vendor}-{self.run_id}'},
                            {'Key': 'radix:run-id', 'Value': self.run_id},
                            {'Key': 'radix:purpose', 'Value': 'gpu-matrix'},
                            {'Key': 'radix:vendor', 'Value': vendor},
                            {'Key': 'radix:ttl', 'Value': str(int(time.time()) + 14400)}  # 4 hours
                        ]
                    }
                ]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            self.instances.append(instance_id)
            print(f"✅ Launched instance: {instance_id}")
            
            # Wait for running state
            waiter = self.ec2.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            print(f"✅ Instance running: {instance_id}")
            
            return instance_id
            
        except Exception as e:
            print(f"❌ Failed to launch instance: {e}")
            return None
    
    def setup_instance(self, instance_id: str, vendor: str) -> Dict[str, Any]:
        """Setup instance with Docker and GPU runtime.
        
        Args:
            instance_id: Instance ID
            vendor: GPU vendor
        
        Returns:
            Command result
        """
        if vendor == 'nvidia':
            commands = [
                'set -e',
                'export DEBIAN_FRONTEND=noninteractive',
                'sudo apt-get update',
                'sudo apt-get install -y docker.io curl',
                'sudo systemctl start docker',
                'sudo systemctl enable docker',
                # Install nvidia-container-toolkit
                'curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg',
                'curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list',
                'sudo apt-get update',
                'sudo apt-get install -y nvidia-container-toolkit',
                'sudo systemctl restart docker',
                'sleep 5',
                # Validate
                'nvidia-smi',
                'docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi'
            ]
        else:  # AMD
            commands = [
                'set -e',
                'export DEBIAN_FRONTEND=noninteractive',
                'sudo apt-get update',
                'sudo apt-get install -y docker.io curl',
                'sudo systemctl start docker',
                'sudo systemctl enable docker',
                # Add user to video group
                'sudo usermod -aG video ssm-user || true',
                'sudo usermod -aG render ssm-user || true',
                # Validate
                'ls -la /dev/kfd /dev/dri'
            ]
        
        return self.ssm_helper.run_command(instance_id, commands, timeout=600)
    
    def get_system_info(self, instance_id: str, vendor: str) -> Dict[str, Any]:
        """Get system information from instance.
        
        Args:
            instance_id: Instance ID
            vendor: GPU vendor
        
        Returns:
            System info dictionary
        """
        if vendor == 'nvidia':
            gpu_cmd = 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'
        else:
            gpu_cmd = 'echo "AMD GPU" && cat /sys/class/drm/card*/device/product_name 2>/dev/null || echo "Unknown"'
        
        commands = [
            f'echo "GPU_INFO:"',
            gpu_cmd,
            'echo "DOCKER_INFO:"',
            'docker --version',
            'echo "OS_INFO:"',
            'cat /etc/os-release | grep PRETTY_NAME'
        ]
        
        result = self.ssm_helper.run_command(instance_id, commands, timeout=60)
        
        # Parse output
        output = result.get('stdout', '')
        info = {
            'gpu_model': 'Unknown',
            'driver_version': 'Unknown',
            'container_runtime_status': 'Unknown'
        }
        
        try:
            if 'GPU_INFO:' in output:
                gpu_line = output.split('GPU_INFO:')[1].split('DOCKER_INFO:')[0].strip()
                if vendor == 'nvidia' and ',' in gpu_line:
                    parts = gpu_line.split(',')
                    info['gpu_model'] = parts[0].strip()
                    info['driver_version'] = parts[1].strip()
                else:
                    info['gpu_model'] = gpu_line
            
            if 'DOCKER_INFO:' in output:
                docker_line = output.split('DOCKER_INFO:')[1].split('OS_INFO:')[0].strip()
                info['container_runtime_status'] = docker_line
        except:
            pass
        
        return info
    
    def create_cluster_and_token(self, vendor: str) -> Tuple[Optional[str], Optional[str]]:
        """Create cluster and request onboarding token.
        
        Args:
            vendor: GPU vendor
        
        Returns:
            Tuple of (cluster_id, one_time_token)
        """
        try:
            # Create cluster
            create_url = f"{self.api_base}/v1/clusters"
            create_payload = {
                'name': f'sentinel-{vendor}-{self.run_id[:8]}',
                'region': self.region
            }
            create_headers = {
                'Authorization': f'Bearer {self.jwt_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(create_url, json=create_payload, headers=create_headers, timeout=30)
            response.raise_for_status()
            cluster_data = response.json()
            cluster_id = cluster_data.get('cluster_id')
            
            # Request token
            token_url = f"{self.api_base}/v1/clusters/{cluster_id}/onboarding-token"
            token_response = requests.post(token_url, headers=create_headers, timeout=30)
            token_response.raise_for_status()
            token_data = token_response.json()
            one_time_token = token_data.get('one_time_token')
            
            return cluster_id, one_time_token
            
        except Exception as e:
            print(f"❌ Failed to create cluster/token: {e}")
            return None, None
    
    def run_onboarding(self, instance_id: str, cluster_id: str, one_time_token: str) -> Dict[str, Any]:
        """Run onboarding installer on instance.
        
        Args:
            instance_id: Instance ID
            cluster_id: Cluster ID
            one_time_token: One-time token (will be redacted in logs)
        
        Returns:
            Command result
        """
        # Use control-plane generated installer
        installer_url = f"{self.api_base}/v1/clusters/{cluster_id}/install.sh"
        
        commands = [
            f'curl -fsSL -H "X-Radix-One-Time-Token: {one_time_token}" "{installer_url}" | sudo bash'
        ]
        
        print(f"   Installer URL: {installer_url}")
        print(f"   Token: [REDACTED]")
        
        return self.ssm_helper.run_command(instance_id, commands, timeout=ONBOARDING_TIMEOUT)
    
    def wait_for_cluster_active(self, cluster_id: str, timeout: int = 120) -> bool:
        """Wait for cluster to become active.
        
        Args:
            cluster_id: Cluster ID
            timeout: Timeout in seconds
        
        Returns:
            True if active
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                url = f"{self.api_base}/v1/clusters/{cluster_id}"
                headers = {'Authorization': f'Bearer {self.jwt_token}'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                status = data.get('status')
                if status == 'active':
                    print(f"✅ Cluster active: {cluster_id}")
                    return True
                
                print(f"   Cluster status: {status}, waiting...")
                
            except Exception as e:
                print(f"   Error checking cluster: {e}")
            
            time.sleep(5)
        
        return False
    
    def submit_and_wait_for_job(self, cluster_id: str) -> Tuple[Optional[str], str]:
        """Submit GPU smoke job and wait for completion.
        
        Args:
            cluster_id: Cluster ID
        
        Returns:
            Tuple of (job_id, status)
        """
        try:
            # Submit simple GPU smoke job
            job_url = f"{self.api_base}/v1/jobs"
            job_payload = {
                'cluster_id': cluster_id,
                'image': 'nvidia/cuda:12.0.0-base-ubuntu22.04',
                'command': ['nvidia-smi'],
                'gpu_count': 1
            }
            job_headers = {
                'Authorization': f'Bearer {self.jwt_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(job_url, json=job_payload, headers=job_headers, timeout=30)
            response.raise_for_status()
            job_data = response.json()
            job_id = job_data.get('job_id')
            
            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < JOB_TIMEOUT:
                status_url = f"{self.api_base}/v1/jobs/{job_id}"
                status_response = requests.get(status_url, headers=job_headers, timeout=10)
                status_response.raise_for_status()
                status_data = status_response.json()
                
                job_status = status_data.get('status')
                if job_status in ['completed', 'failed', 'cancelled']:
                    print(f"   Job {job_status}: {job_id}")
                    return job_id, job_status
                
                print(f"   Job status: {job_status}, waiting...")
                time.sleep(5)
            
            return job_id, 'timeout'
            
        except Exception as e:
            print(f"❌ Job submission/wait failed: {e}")
            return None, 'error'
    
    def delete_cluster(self, cluster_id: str):
        """Delete cluster.
        
        Args:
            cluster_id: Cluster ID
        """
        try:
            url = f"{self.api_base}/v1/clusters/{cluster_id}"
            headers = {'Authorization': f'Bearer {self.jwt_token}'}
            requests.delete(url, headers=headers, timeout=30)
            print(f"🗑️  Deleted cluster: {cluster_id}")
        except:
            pass
    
    def cleanup_instances(self):
        """Cleanup all launched instances."""
        print(f"\n🧹 Cleaning up instances...")
        for instance_id in self.instances:
            self.ssm_helper.terminate_instance(instance_id)
    
    def promote_digest(self, ssm_parameter: str) -> bool:
        """Promote candidate digest to approved.
        
        Args:
            ssm_parameter: SSM parameter path
        
        Returns:
            True if successful
        """
        print(f"\n🎉 Promoting digest to approved...")
        print(f"   Parameter: {ssm_parameter}")
        print(f"   Digest: {self.candidate_digest}")
        
        success = self.ssm_helper.put_parameter(
            ssm_parameter,
            self.candidate_digest,
            description=f'Approved agent digest from Sentinel run {self.run_id} at {datetime.now(timezone.utc).isoformat()}'
        )
        
        if success:
            self.evidence_pack.set_promotion_result(
                promoted=True,
                reason='All matrix tests passed',
                ssm_parameter=ssm_parameter
            )
        else:
            self.evidence_pack.set_promotion_result(
                promoted=False,
                reason='Failed to update SSM parameter'
            )
        
        return success
    
    def save_evidence_pack(self, output_dir: str = '.'):
        """Save evidence pack.
        
        Args:
            output_dir: Output directory
        """
        self.evidence_pack.save(output_dir)


def main():
    """Main entry point."""
    # Get environment variables
    run_id = os.environ.get('GITHUB_RUN_ID', f'local-{int(time.time())}')
    git_sha = os.environ.get('GITHUB_SHA', 'unknown')
    candidate_digest = os.environ.get('RADIX_CANDIDATE_DIGEST')
    environment = os.environ.get('RADIX_ENVIRONMENT', 'dev')
    skip_promotion = os.environ.get('SKIP_PROMOTION', 'false').lower() == 'true'
    
    if not candidate_digest:
        print("❌ Missing required environment variable: RADIX_CANDIDATE_DIGEST")
        sys.exit(1)
    
    # Create runner (auth happens automatically)
    runner = GPUMatrixRunner(run_id, git_sha, candidate_digest, environment)
    
    try:
        # Run matrix tests
        all_passed = runner.run_matrix()
        
        # Promote if all passed
        if all_passed and not skip_promotion:
            runner.promote_digest(runner.config['ssm_param'])
        elif all_passed and skip_promotion:
            print("\n⏭️  Skipping promotion (dry run mode)")
            runner.evidence_pack.set_promotion_result(
                promoted=False,
                reason='Skipped (dry run mode)'
            )
        else:
            print("\n❌ Not promoting: tests failed")
            runner.evidence_pack.set_promotion_result(
                promoted=False,
                reason='Matrix tests failed'
            )
        
        # Save evidence pack
        runner.save_evidence_pack()
        
        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        runner.evidence_pack.set_promotion_result(
            promoted=False,
            reason=f'Fatal error: {str(e)}'
        )
        runner.save_evidence_pack()
        sys.exit(1)


if __name__ == '__main__':
    main()
