# CI GPU Benchmark Harness

This document explains how to use the automated GPU benchmark harness workflow to validate Radix cluster agent performance on AWS EC2 GPU instances.

## Overview

The `gpu-benchmark-ec2.yml` workflow provisions an ephemeral AWS EC2 GPU instance, deploys the Radix cluster agent, runs a ResNet50 benchmark, captures performance metrics, and generates an evidence report—all automatically via GitHub Actions.

**Key Features**:
- ✓ Fully automated end-to-end GPU benchmarking
- ✓ Ephemeral infrastructure (instance terminated after test)
- ✓ Secure credential handling (no secrets in logs)
- ✓ Evidence artifacts uploaded to GitHub
- ✓ Cloud-agnostic agent validation

## Prerequisites

### Required GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

#### 1. Cognito Authentication (for Cluster API)
- **Secret Names**: 
  - `RADIX_COGNITO_USER_POOL_ID` (e.g., `us-west-2_abc123`)
  - `RADIX_COGNITO_CLIENT_ID`
  - `RADIX_TEST_USERNAME` (CI test user email)
  - `RADIX_TEST_PASSWORD` (CI test user password)

**Purpose**: The workflow uses Cognito JWT authentication for cluster CRUD operations (create cluster, create onboarding token, etc.). This is the same authentication method used by the dashboard.

**Setup**: See [AUTH_CI_LOGIN.md](./AUTH_CI_LOGIN.md) for detailed instructions on creating a CI test user and configuring Cognito authentication.

**Note**: Cluster API endpoints require Cognito JWT tokens (not telemetry API keys).

#### 2. AWS Credentials (OIDC)
- **Secret Name**: `AWS_ROLE_TO_ASSUME_ARN`
- **Permissions Required**:
  - `ec2:RunInstances`
  - `ec2:TerminateInstances`
  - `ec2:DescribeInstances`
  - `ec2:CreateTags`
  - `iam:PassRole` (for instance profile)
  - `cognito-idp:AdminInitiateAuth` (for obtaining ID tokens)

**Recommendation**: Use OIDC role assumption (no static credentials). The workflow already uses `aws-actions/configure-aws-credentials@v4` with OIDC.

#### 3. AWS Infrastructure Configuration

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `BENCH_AMI_ID` | Deep Learning AMI ID in target region | `ami-0abcdef1234567890` |
| `BENCH_SUBNET_ID` | VPC subnet ID (must have internet access) | `subnet-0123456789abcdef0` |
| `BENCH_SECURITY_GROUP_ID` | Security group ID (allow egress) | `sg-0123456789abcdef0` |
| `BENCH_INSTANCE_PROFILE_NAME` | IAM instance profile with SSM permissions | `radix-benchmark-instance-profile` |

### AWS Infrastructure Setup

#### 1. Choose a Deep Learning AMI

Use an AWS Deep Learning AMI (DLAMI) that includes:
- NVIDIA drivers
- Docker pre-installed
- CUDA toolkit

**Recommended AMIs** (us-west-2):
- **Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 20.04)**: `ami-0c2f3a0f3b0f3a0f3` (example)
- **Deep Learning Base AMI (Ubuntu 20.04)**: Lighter weight option

Find the latest DLAMI:
```bash
aws ec2 describe-images \
  --region us-west-2 \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text
```

#### 2. Create VPC and Subnet

The subnet must have:
- Internet gateway attached (for agent to reach Radix API)
- Auto-assign public IP enabled (or use NAT gateway)

```bash
# Example: Use default VPC
aws ec2 describe-subnets \
  --region us-west-2 \
  --filters "Name=default-for-az,Values=true" \
  --query 'Subnets[0].SubnetId' \
  --output text
```

#### 3. Create Security Group

Allow outbound HTTPS (443) to Radix API:

```bash
aws ec2 create-security-group \
  --region us-west-2 \
  --group-name radix-benchmark-sg \
  --description "Security group for Radix benchmark instances"

aws ec2 authorize-security-group-egress \
  --region us-west-2 \
  --group-name radix-benchmark-sg \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0
```

**Note**: No inbound rules required (agent uses SSM for management).

#### 4. Create IAM Instance Profile

The instance profile must include `AmazonSSMManagedInstanceCore` for SSM access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Attach managed policy:
```bash
aws iam attach-role-policy \
  --role-name radix-benchmark-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
```

Create instance profile:
```bash
aws iam create-instance-profile \
  --instance-profile-name radix-benchmark-instance-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name radix-benchmark-instance-profile \
  --role-name radix-benchmark-role
```

## Running the Workflow

### 1. Navigate to GitHub Actions

Go to: `https://github.com/iangreen74/radix/actions/workflows/gpu-benchmark-ec2.yml`

### 2. Click "Run workflow"

Fill in the parameters (all have sensible defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| **aws_region** | `us-west-2` | AWS region for EC2 instance |
| **instance_type** | `g5.2xlarge` | GPU instance type |
| **ami_id** | (from secret) | Override AMI ID |
| **subnet_id** | (from secret) | Override subnet ID |
| **security_group_id** | (from secret) | Override security group ID |
| **instance_profile_name** | (from secret) | Override instance profile |
| **api_base** | `https://api.vaultscaler.com` | Radix API endpoint |
| **cluster_name** | `ci-gpu-bench` | Cluster name in dashboard |
| **epochs** | `1` | Training epochs |
| **batch_size** | `128` | Batch size |
| **ttl_minutes** | `45` | Max runtime before timeout |

### 3. Monitor Execution

The workflow will:
1. ✓ Create Radix cluster via API
2. ✓ Generate onboarding token
3. ✓ Launch EC2 GPU instance
4. ✓ Wait for instance to boot and agent to start
5. ✓ Wait for cluster to become active
6. ✓ Launch ResNet50 benchmark
7. ✓ Poll for job completion
8. ✓ Extract throughput and duration metrics
9. ✓ Generate evidence report
10. ✓ Upload artifact
11. ✓ Terminate instance and cleanup

**Expected Duration**: 10-20 minutes (depending on instance type and epochs)

### 4. View Results

#### Workflow Summary

Check the workflow run page for a summary table:

```
✅ GPU Benchmark Completed Successfully

Results
Instance Type: g5.2xlarge
Throughput: 343.2 images/sec
Duration: 85.6 seconds
Epochs: 1
Batch Size: 128
```

#### Evidence Artifact

Download the evidence report from the workflow artifacts:
- **Artifact Name**: `benchmark-evidence-{instance_type}-{run_number}`
- **Contents**: Markdown file with full benchmark details
- **Retention**: 90 days

#### Dashboard Verification

The cluster will appear briefly in the Radix Dashboard:
1. Navigate to **Clusters** section
2. Find cluster by name (e.g., `ci-gpu-bench`)
3. View status, heartbeats, and job history
4. Cluster is automatically cleaned up after workflow completes

## Cost Considerations

### Instance Pricing (us-west-2, as of 2024)

| Instance Type | GPU | VRAM | vCPU | RAM | Price/hour | 45min Cost |
|---------------|-----|------|------|-----|------------|------------|
| **g4dn.xlarge** | T4 | 16GB | 4 | 16GB | $0.526 | ~$0.39 |
| **g4dn.2xlarge** | T4 | 16GB | 8 | 32GB | $0.752 | ~$0.56 |
| **g5.xlarge** | A10G | 24GB | 4 | 16GB | $1.006 | ~$0.75 |
| **g5.2xlarge** | A10G | 24GB | 8 | 32GB | $1.212 | ~$0.91 |
| **g5.4xlarge** | A10G | 24GB | 16 | 64GB | $1.624 | ~$1.22 |

**Recommendation**: Use `g4dn.xlarge` for cost-effective testing, `g5.2xlarge` for production benchmarks.

### Cost Optimization Tips

1. **Reduce TTL**: Set `ttl_minutes` to minimum needed (e.g., 20 for single epoch)
2. **Use Smaller Instance**: Start with `g4dn.xlarge` for validation
3. **Reduce Epochs**: Use `epochs: 1` for quick tests
4. **Spot Instances**: Modify workflow to use spot instances (60-90% savings)
5. **Concurrency Limit**: Workflow already limits to 1 concurrent run

### Safety Features

- **Automatic Cleanup**: Instance always terminated via `if: always()` step
- **Timeout**: Workflow has 60-minute hard timeout
- **Concurrency Control**: Only 1 benchmark runs at a time
- **Resource Tags**: Instances tagged with `ManagedBy: github-actions`

## Troubleshooting

### "Failed to create cluster" or "Unauthorized"

**Cause**: Invalid Cognito credentials or API endpoint unreachable

**Solution**:
1. Verify all Cognito secrets are set correctly:
   - `RADIX_COGNITO_USER_POOL_ID`
   - `RADIX_COGNITO_CLIENT_ID`
   - `RADIX_TEST_USERNAME`
   - `RADIX_TEST_PASSWORD`
2. Ensure the CI test user exists and is confirmed in Cognito
3. Verify the Cognito app client has `ALLOW_ADMIN_USER_PASSWORD_AUTH` enabled
4. Check that the AWS role has `cognito-idp:AdminInitiateAuth` permission
5. Ensure `api_base` parameter is correct

### "Failed to launch instance"

**Cause**: Invalid AWS configuration or insufficient permissions

**Solution**:
1. Verify all `BENCH_*` secrets are set
2. Check IAM user has required EC2 permissions
3. Verify AMI ID exists in target region
4. Ensure subnet has internet access

### "Cluster did not become active within timeout"

**Cause**: Agent failed to start or authenticate

**Solution**:
1. Check instance has internet access (subnet routing)
2. Verify security group allows outbound HTTPS (443)
3. Check AMI has Docker pre-installed
4. Review instance user data logs via SSM:
   ```bash
   aws ssm start-session --target <instance-id>
   cat /var/log/cloud-init-output.log
   docker logs radix-agent
   ```

### "Job did not complete within TTL"

**Cause**: Benchmark taking longer than expected

**Solution**:
1. Increase `ttl_minutes` parameter
2. Reduce `epochs` or `batch_size`
3. Use faster instance type (e.g., g5.4xlarge)

### "Instance termination failed"

**Cause**: Transient AWS API error

**Solution**:
- Manually terminate instance via AWS Console
- Check for orphaned instances with tag `ManagedBy: github-actions`

## Advanced Usage

### Custom Benchmark Parameters

Test different configurations:

```yaml
# Quick validation (5-10 minutes)
epochs: 1
batch_size: 64
instance_type: g4dn.xlarge
ttl_minutes: 20

# Production benchmark (15-20 minutes)
epochs: 1
batch_size: 128
instance_type: g5.2xlarge
ttl_minutes: 30

# Extended test (30-45 minutes)
epochs: 3
batch_size: 256
instance_type: g5.4xlarge
ttl_minutes: 60
```

### Multiple Instance Types

Run benchmarks across different GPUs:

1. Run workflow with `instance_type: g4dn.xlarge`
2. Wait for completion
3. Run workflow with `instance_type: g5.2xlarge`
4. Compare evidence reports

### Automated Regression Testing

Add to CI pipeline:

```yaml
# .github/workflows/benchmark-regression.yml
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday 2am UTC

jobs:
  benchmark:
    uses: ./.github/workflows/gpu-benchmark-ec2.yml
    secrets: inherit
    with:
      instance_type: g5.2xlarge
      epochs: 1
      batch_size: 128
```

## Security Best Practices

1. **Cognito User Management**: Use dedicated CI test user with minimal permissions
2. **Password Rotation**: Rotate `RADIX_TEST_PASSWORD` periodically
3. **IAM Least Privilege**: Use minimal IAM permissions for AWS role
4. **Secret Masking**: Workflow automatically masks all sensitive values (ID tokens, passwords)
5. **No SSH Keys**: Use SSM for instance access (no SSH keys in workflow)
6. **Ephemeral Resources**: All resources cleaned up after test
7. **Audit Logs**: Review CloudTrail for EC2 API calls and Cognito authentication events

## Authentication Architecture

The GPU benchmark harness uses **two separate authentication mechanisms**:

1. **Cognito JWT (for Cluster API)**:
   - Endpoints: `/v1/clusters`, `/v1/clusters/{id}/onboarding-token`, `/v1/clusters/{id}/jobs`, etc.
   - Header: `Authorization: Bearer <id_token>`
   - Used by: Dashboard UI, CI workflows, integration tests
   - Obtained via: `scripts/get_ci_token.py` using `admin_initiate_auth`

2. **Telemetry API Keys (for Agent Telemetry)**:
   - Endpoints: `/v1/score`, `/v1/observe` (agent-to-API telemetry)
   - Header: `x-radix-api-key: <api_key>`
   - Used by: Cluster agents for sending metrics
   - Obtained via: Dashboard "Create API Key" (deprecated for cluster CRUD)

**Important**: The benchmark harness creates clusters via the Cluster API, which requires Cognito JWT authentication. Telemetry API keys are NOT used in this workflow.

**JWT Authorizer Configuration**: The API Gateway JWT authorizer accepts tokens from both the dashboard app client and the CI admin-auth app client (`6ehl9so0lctguq1r8tle0f4huc`), allowing CI workflows to authenticate independently of the dashboard.

## Related Documentation

- [CI Authentication Setup](./AUTH_CI_LOGIN.md) - **Required reading for Cognito setup**
- [Kubernetes Onboarding Guide](./KUBERNETES_ONBOARDING.md)
- [Benchmark Button Guide](./BENCHMARK_BUTTON.md)
- [API Reference](./API.md)
- [Agent Architecture](./AGENT_ARCHITECTURE.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/iangreen74/radix/issues
- Workflow Logs: Check GitHub Actions run logs for detailed output
- AWS Troubleshooting: Use SSM Session Manager to access instance
