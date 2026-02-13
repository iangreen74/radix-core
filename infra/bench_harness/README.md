# Radix Benchmark Harness Infrastructure

This Terraform module provisions the AWS infrastructure required for the Radix GPU Benchmark Harness.

## Overview

The module creates:
- **IAM Role & Instance Profile** with `AmazonSSMManagedInstanceCore` policy for SSM Session Manager access
- **Security Group** allowing outbound HTTPS (443) and HTTP (80)
- **VPC Endpoints** (conditional) for SSM connectivity in private subnets:
  - `com.amazonaws.{region}.ssm`
  - `com.amazonaws.{region}.ssmmessages`
  - `com.amazonaws.{region}.ec2messages`

### Subnet Detection

The module automatically detects whether the provided subnet is public or private based on the `map_public_ip_on_launch` attribute:
- **Public subnet**: `map_public_ip_on_launch = true` (auto-assigns public IPs)
- **Private subnet**: `map_public_ip_on_launch = false` → VPC endpoints are created for SSM access

This detection method is robust and works across all AWS accounts without requiring route table access.

## Prerequisites

- Terraform >= 1.0
- AWS CLI configured with appropriate credentials
- Existing VPC and subnet

## Usage

### 1. Create Configuration

```bash
cd infra/bench_harness
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your VPC and subnet IDs:

```hcl
vpc_id    = "vpc-0cd77a3a159a54896"
subnet_id = "subnet-0c993c1cdd523150d"
region    = "us-west-2"
```

### 2. Initialize and Apply

```bash
terraform init
terraform plan
terraform apply
```

### 3. Capture Outputs

After successful apply, capture the outputs:

```bash
terraform output bench_security_group_id
terraform output bench_instance_profile_name
```

## GitHub Secrets Mapping

The Terraform outputs map to GitHub Actions secrets used by the GPU benchmark workflow:

| Terraform Output | GitHub Secret | Description |
|-----------------|---------------|-------------|
| `bench_security_group_id` | `BENCH_SECURITY_GROUP_ID` | Security group ID for benchmark EC2 instances |
| `bench_instance_profile_name` | `BENCH_INSTANCE_PROFILE_NAME` | IAM instance profile name for benchmark EC2 instances |

### Setting GitHub Secrets

1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add the following secrets:

```bash
# Get values from Terraform outputs
BENCH_SECURITY_GROUP_ID=$(terraform output -raw bench_security_group_id)
BENCH_INSTANCE_PROFILE_NAME=$(terraform output -raw bench_instance_profile_name)
```

3. Create secrets in GitHub UI or via CLI:

```bash
gh secret set BENCH_SECURITY_GROUP_ID --body "$BENCH_SECURITY_GROUP_ID"
gh secret set BENCH_INSTANCE_PROFILE_NAME --body "$BENCH_INSTANCE_PROFILE_NAME"
```

## Outputs

| Output | Description |
|--------|-------------|
| `bench_security_group_id` | Security group ID for benchmark instances |
| `bench_instance_profile_name` | Instance profile name for benchmark instances |
| `bench_instance_profile_arn` | Instance profile ARN |
| `bench_iam_role_name` | IAM role name |
| `bench_iam_role_arn` | IAM role ARN |
| `is_public_subnet` | Whether the subnet is public (true) or private (false) |
| `vpc_endpoints_created` | Whether VPC endpoints were created (true for private subnets) |

## Architecture

### Public Subnet Configuration
```
┌─────────────────────────────────────┐
│ Benchmark EC2 Instance              │
│ - Instance Profile: bench-instance  │
│ - Security Group: bench-sg          │
│   - Egress: 443 (HTTPS)             │
│   - Egress: 80 (HTTP)               │
└──────────────┬──────────────────────┘
               │
               ▼
         Internet Gateway
               │
               ▼
    AWS Services (SSM, API, etc.)
```

### Private Subnet Configuration
```
┌─────────────────────────────────────┐
│ Benchmark EC2 Instance              │
│ - Instance Profile: bench-instance  │
│ - Security Group: bench-sg          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ VPC Endpoints (Interface)           │
│ - ssm.{region}.amazonaws.com        │
│ - ssmmessages.{region}.amazonaws.com│
│ - ec2messages.{region}.amazonaws.com│
│ - Private DNS enabled               │
└─────────────────────────────────────┘
```

## Cost Considerations

- **IAM Role/Instance Profile**: Free
- **Security Group**: Free
- **VPC Endpoints** (private subnet only): ~$0.01/hour per endpoint (~$0.03/hour total)
  - 3 endpoints × $0.01/hour × 730 hours/month = ~$21.90/month
  - Plus data processing charges (~$0.01/GB)

**Recommendation**: Use public subnets for benchmark harness to avoid VPC endpoint costs, unless private subnet is required for security/compliance.

## Troubleshooting

### SSM Session Manager Not Working

**Public Subnet:**
- Verify instance has public IP or NAT gateway route
- Check security group allows outbound HTTPS (443)
- Verify IAM instance profile is attached to EC2 instance

**Private Subnet:**
- Verify VPC endpoints were created: `terraform output vpc_endpoints_created`
- Check VPC endpoint security group allows inbound 443 from benchmark security group
- Verify private DNS is enabled on VPC endpoints
- Verify subnet has `map_public_ip_on_launch = false` (confirms private subnet detection)

### Subnet Detection Issues

If subnet is incorrectly detected as public/private:
```bash
# Check subnet map_public_ip_on_launch attribute
aws ec2 describe-subnets --subnet-ids <subnet-id> --query 'Subnets[0].MapPublicIpOnLaunch'

# Expected output:
# - true = public subnet (no VPC endpoints created)
# - false = private subnet (VPC endpoints created)
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete the IAM role, instance profile, security group, and VPC endpoints. Ensure no EC2 instances are using these resources.

## Support

For issues or questions:
- Check GitHub workflow logs: `.github/workflows/gpu-benchmark-ec2.yml`
- Review Terraform plan output before applying
- Verify AWS credentials have sufficient permissions (EC2, IAM, VPC)
