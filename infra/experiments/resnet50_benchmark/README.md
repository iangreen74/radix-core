# ResNet-50 Benchmark GPU Instance - Terraform Stack

This Terraform stack provisions a **single GPU EC2 instance** for running the ResNet-50 training benchmark. It uses the **default VPC** and maintains a minimal, isolated configuration.

## Purpose

- Provision a GPU-enabled EC2 instance for ResNet-50 throughput benchmarking
- Use AWS Deep Learning AMI with pre-installed NVIDIA drivers and CUDA toolkit
- Enable Docker with GPU support for running containerized benchmarks
- Support both SSH and AWS Systems Manager Session Manager access

## Prerequisites

1. **Terraform** installed (version >= 1.0)
   ```bash
   terraform version
   ```

2. **AWS credentials** configured via AWS CLI or environment variables
   ```bash
   aws configure
   # OR set environment variables:
   export AWS_ACCESS_KEY_ID="..."
   export AWS_SECRET_ACCESS_KEY="..."
   export AWS_DEFAULT_REGION="us-west-2"
   ```

3. **(Optional)** EC2 key pair created in AWS if you want SSH access
   ```bash
   aws ec2 create-key-pair --key-name radix-benchmark --query 'KeyMaterial' --output text > ~/.ssh/radix-benchmark.pem
   chmod 400 ~/.ssh/radix-benchmark.pem
   ```

## Resources Created

This stack creates:

- **EC2 Instance**: GPU instance (default: `g5.xlarge` with 1x A10G GPU)
- **Security Group**: Allows SSH inbound (configurable CIDR) and all outbound
- **IAM Role**: With SSM access, ECR pull permissions, and S3 read access to artifacts bucket
- **IAM Instance Profile**: Attached to the EC2 instance

Uses existing resources:
- **Default VPC** and default subnets

## Prerequisites

### Required: GPU AMI ID

This stack requires you to provide a **Deep Learning GPU AMI ID** for the target region.

**To find a suitable AMI:**

1. **Via AWS Console:**
   - Go to **EC2 → AMIs**
   - Set filters: **Public images**, **Owner = Amazon**
   - Search for: `Deep Learning AMI GPU PyTorch Ubuntu`
   - Copy the AMI ID from your target region (e.g., `ami-0123456789abcdef0`)

2. **Via AWS CLI:**
   ```bash
   aws ec2 describe-images \
     --owners amazon \
     --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*" \
     --query 'Images[*].[ImageId,Name,CreationDate]' \
     --output table \
     --region us-west-2
   ```

**Example AMI IDs (as of Dec 2024, verify current):**
- `us-west-2`: Check AWS Console for latest
- `us-east-1`: Check AWS Console for latest

## Usage

### 1. Initialize Terraform

```bash
cd infra/experiments/resnet50_benchmark
terraform init
```

### 2. Review the Plan

**⚠️ REQUIRED: You must specify `ami_id`**

```bash
terraform plan \
  -var="aws_region=us-west-2" \
  -var="instance_type=g5.xlarge" \
  -var="ami_id=ami-0123456789abcdef0"
```

**With optional variables:**
```bash
terraform plan \
  -var="aws_region=us-west-2" \
  -var="instance_type=g5.xlarge" \
  -var="ami_id=ami-0123456789abcdef0" \
  -var="ssh_key_name=radix-benchmark" \
  -var="ssh_allowed_cidr=203.0.113.0/24"
```

### 3. Apply the Configuration

```bash
terraform apply \
  -var="aws_region=us-west-2" \
  -var="instance_type=g5.xlarge" \
  -var="ami_id=ami-0123456789abcdef0"
```

Type `yes` when prompted to confirm.

### 4. Get Instance Information

After successful apply, Terraform outputs:

```
Outputs:

instance_id = "i-0123456789abcdef0"
public_ip = "54.123.45.67"
availability_zone = "us-west-2a"
security_group_id = "sg-0123456789abcdef0"
instance_type = "g5.xlarge"
ami_id = "ami-0123456789abcdef0"
ssh_command = "ssh -i ~/.ssh/radix-benchmark.pem ubuntu@54.123.45.67"
ssm_command = "aws ssm start-session --target i-0123456789abcdef0 --region us-west-2"
```

### 5. Connect to the Instance

**Option A: SSH (if key pair configured)**
```bash
ssh -i ~/.ssh/radix-benchmark.pem ubuntu@<public_ip>
```

**Option B: AWS Systems Manager Session Manager (no key pair needed)**
```bash
aws ssm start-session --target <instance_id> --region us-west-2
```

### 6. Verify GPU and Docker

Once connected:

```bash
# Check GPU
nvidia-smi

# Check Docker
docker --version
docker ps

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Next Steps

After the instance is running:

1. **Build and push benchmark image** (from your local machine):
   ```bash
   # From repo root
   scripts/experiments/resnet50/build_and_push_images.sh
   ```

2. **Bootstrap the Radix agent** (on the GPU instance):
   ```bash
   # Set environment variables
   export RADIX_API_BASE="https://api.vaultscaler.com"
   export RADIX_CLUSTER_ID="cluster-xxxx"
   export RADIX_CLUSTER_TOKEN="***"
   export RADIX_TENANT_ID="user-48a1e310-..."
   
   # Run bootstrap script
   AGENT_IMAGE="iangreen74/radix-cluster-agent:latest" \
   scripts/experiments/resnet50/bootstrap_agent.sh
   ```

3. **Register cluster** in Radix dashboard (if not already done)

4. **Run experiments** via Radix pipelines or direct job submission

## Instance Type Selection

### For Smoke Tests / Development
- **`g5.xlarge`**: 1x A10G GPU, 4 vCPUs, 16 GB RAM
  - Cost: ~$1.00/hour
  - Good for testing and small experiments

### For Full Benchmarks
- **`g5.12xlarge`**: 4x A10G GPUs, 48 vCPUs, 192 GB RAM
  - Cost: ~$5.67/hour
  - Recommended for the "21% throughput gain" case study

### Alternative GPU Options
- **`p3.2xlarge`**: 1x V100 GPU, 8 vCPUs, 61 GB RAM (~$3.06/hour)
- **`p3.8xlarge`**: 4x V100 GPUs, 32 vCPUs, 244 GB RAM (~$12.24/hour)

**To change instance type:**
```bash
terraform apply -var="instance_type=g5.12xlarge"
```

## Destroy Infrastructure

**⚠️ WARNING: This will terminate the instance and delete all data on it.**

```bash
terraform destroy \
  -var="aws_region=us-west-2" \
  -var="instance_type=g5.xlarge"
```

Type `yes` when prompted to confirm.

## Cost Management

**Important reminders:**

- GPU instances are **expensive** (g5.xlarge: ~$1/hour, g5.12xlarge: ~$5.67/hour)
- **Always destroy** the instance when not in use
- Consider using **EC2 Instance Scheduler** or manual stop/start for longer experiments
- Monitor costs in AWS Cost Explorer

**To stop (not destroy) the instance:**
```bash
aws ec2 stop-instances --instance-ids <instance_id> --region us-west-2
```

**To start a stopped instance:**
```bash
aws ec2 start-instances --instance-ids <instance_id> --region us-west-2
```

**Note:** Stopping saves compute costs but you still pay for EBS storage (~$0.10/GB/month).

## Security Considerations

### SSH Access
- Default `ssh_allowed_cidr` is `0.0.0.0/0` (open to the world)
- **Change this** for production:
  ```bash
  terraform apply -var="ssh_allowed_cidr=203.0.113.0/24"
  ```

### Secrets Management
- Never commit `terraform.tfstate` to git (contains sensitive data)
- Use environment variables for Radix cluster tokens
- Consider AWS Secrets Manager for production deployments

### IAM Permissions

The Terraform stack automatically grants the following permissions to the EC2 instance:

- **SSM Access**: AWS Systems Manager Session Manager (for remote shell access)
- **ECR Pull**: Amazon ECR (for pulling Docker images)
- **S3 Read**: Read-only access to `s3://radix-core-artifacts-418295677815-us-west-2/resnet50-lab/*` (for git-free deployment)

No manual IAM configuration is required. The instance can download source archives from S3 without additional setup.

## Troubleshooting

### "No default VPC found"
If your AWS account doesn't have a default VPC:
```bash
aws ec2 create-default-vpc --region us-west-2
```

### "Insufficient capacity" error
Try a different availability zone:
```bash
terraform apply -var="aws_region=us-west-1"
```

Or try a different instance type:
```bash
terraform apply -var="instance_type=g5.2xlarge"
```

### GPU not detected
1. Check AMI has NVIDIA drivers: `nvidia-smi`
2. Verify instance type has GPU: `lspci | grep -i nvidia`
3. Check NVIDIA container toolkit: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

### Docker permission denied
Add user to docker group and re-login:
```bash
sudo usermod -aG docker $USER
# Then logout and login again
```

## State Management

This stack uses a **remote S3 backend** for state management:

- **Bucket:** `vaultscaler-tf-state-418295677815-us-east-1`
- **Key:** `experiments/resnet50_benchmark/terraform.tfstate`
- **Region:** `us-east-1`
- **Encryption:** Enabled

**Benefits:**
- Terraform state is shared between local runs and GitHub Actions
- Multiple team members can collaborate safely
- State is encrypted at rest in S3

**Important:**
- The S3 bucket must exist before running `terraform init`
- To change the backend configuration, edit the `terraform { backend "s3" { ... } }` block in `main.tf`
- After changing backend config, run `terraform init -reconfigure`

## Relationship to Other Radix Infrastructure

This stack is **completely isolated** from:
- `infra/aws/` (EKS cluster infrastructure)
- `infra/cloud/` (SAM/Lambda backend)
- Any other Terraform modules

It can be created/destroyed independently without affecting other Radix components.

## Future Enhancements

- [ ] Add CloudWatch monitoring and alarms
- [ ] Add EBS volume snapshots for data persistence
- [ ] Add auto-shutdown after idle time
- [ ] Add support for spot instances (cost savings)
- [ ] Add multi-GPU instance support with NVLINK
- [ ] Migrate to remote state for team collaboration
