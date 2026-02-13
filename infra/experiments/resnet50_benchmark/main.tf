# Terraform configuration for ResNet-50 benchmark GPU instance
# This stack uses remote S3 backend for state management

terraform {
  required_version = ">= 1.5.0"
  
  backend "s3" {
    bucket  = "vaultscaler-tf-state-418295677815-us-east-1"
    key     = "experiments/resnet50_benchmark/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
    # DynamoDB table for state locking (optional, add if available):
    # dynamodb_table = "vaultscaler-tf-locks"
  }
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources for default VPC and subnets
# Look up the default VPC in the selected region
data "aws_vpc" "default" {
  default = true
}

# Get subnets for that VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security Group for benchmark instance
resource "aws_security_group" "benchmark_sg" {
  name        = "radix-resnet50-benchmark-sg"
  description = "Security group for ResNet-50 benchmark GPU instance"
  vpc_id      = data.aws_vpc.default.id

  # Inbound: SSH access
  # WARNING: Default allows SSH from anywhere. Change ssh_allowed_cidr variable for production.
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  # Outbound: Allow all (needed for Docker pulls, Radix API, etc.)
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "radix-resnet50-benchmark-sg"
    Project     = "radix"
    Environment = "experiments"
    Experiment  = "resnet50_benchmark"
  }
}

# IAM Role for EC2 instance
resource "aws_iam_role" "benchmark_role" {
  name = "radix-resnet50-benchmark-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "radix-resnet50-benchmark-role"
    Project     = "radix"
    Environment = "experiments"
    Experiment  = "resnet50_benchmark"
  }
}

# Attach AWS Systems Manager policy for SSM Session Manager access
resource "aws_iam_role_policy_attachment" "ssm_policy" {
  role       = aws_iam_role.benchmark_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Inline policy for ECR access (for pulling Docker images)
resource "aws_iam_role_policy" "ecr_policy" {
  name = "radix-resnet50-ecr-access"
  role = aws_iam_role.benchmark_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# Inline policy for S3 read access to source artifacts
resource "aws_iam_role_policy" "s3_artifacts_read" {
  name = "radix-resnet50-benchmark-s3-read"
  role = aws_iam_role.benchmark_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::radix-core-artifacts-418295677815-us-west-2",
          "arn:aws:s3:::radix-core-artifacts-418295677815-us-west-2/resnet50-lab/*"
        ]
      }
    ]
  })
}

# Instance profile for the EC2 instance
resource "aws_iam_instance_profile" "benchmark_instance_profile" {
  name = "radix-resnet50-benchmark-profile"
  role = aws_iam_role.benchmark_role.name

  tags = {
    Name        = "radix-resnet50-benchmark-profile"
    Project     = "radix"
    Environment = "experiments"
    Experiment  = "resnet50_benchmark"
  }
}

# EC2 instance for GPU benchmarking
resource "aws_instance" "gpu_benchmark" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.benchmark_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.benchmark_instance_profile.name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  # Enable public IP for SSH and SSM access
  associate_public_ip_address = true

  # User data: Install Docker and enable it
  # Note: DLAMI already has NVIDIA drivers and CUDA toolkit
  # We just need to ensure Docker is installed and nvidia-container-toolkit is set up
  user_data = <<-EOF
              #!/bin/bash
              set -e
              
              # Update system
              apt-get update
              
              # Install Docker if not present
              if ! command -v docker &> /dev/null; then
                apt-get install -y docker.io
              fi
              
              # Enable and start Docker
              systemctl enable docker
              systemctl start docker
              
              # Add ubuntu user to docker group
              usermod -aG docker ubuntu || true
              
              # Install NVIDIA Container Toolkit if not present
              if ! dpkg -l | grep -q nvidia-container-toolkit; then
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                apt-get update
                apt-get install -y nvidia-container-toolkit
                systemctl restart docker
              fi
              
              # Verify GPU access
              nvidia-smi || echo "WARNING: nvidia-smi failed"
              
              # Test Docker GPU access
              docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi || echo "WARNING: Docker GPU test failed"
              
              echo "Bootstrap complete. Instance ready for Radix agent."
              EOF

  tags = {
    Name        = "radix-resnet50-benchmark"
    Project     = "radix"
    Environment = "experiments"
    Experiment  = "resnet50_benchmark"
  }

  # Ensure we have a reasonable root volume for Docker images
  root_block_device {
    volume_size = 100 # GB
    volume_type = "gp3"
    encrypted   = true
  }
}
