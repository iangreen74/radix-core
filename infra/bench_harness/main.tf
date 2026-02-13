terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Data source to check if subnet is public or private
data "aws_subnet" "bench_subnet" {
  id = var.subnet_id
}

# Subnet detection based on map_public_ip_on_launch attribute
locals {
  # A subnet is public if it auto-assigns public IPs to instances
  is_public_subnet = data.aws_subnet.bench_subnet.map_public_ip_on_launch

  common_tags = merge(
    {
      Project              = "radix"
      Component            = "benchmark-harness"
      ManagedBy            = "terraform"
      "radix:bench_run_id" = var.run_id
    },
    var.tags
  )
}

# IAM Role for benchmark EC2 instances
resource "aws_iam_role" "bench_instance_role" {
  name = "${var.name_prefix}-${var.run_id}-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name_prefix}-${var.run_id}-instance-role"
    }
  )
}

# Attach AWS managed policy for SSM Session Manager
resource "aws_iam_role_policy_attachment" "bench_ssm_policy" {
  role       = aws_iam_role.bench_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Instance profile for EC2
resource "aws_iam_instance_profile" "bench_instance_profile" {
  name = "${var.name_prefix}-${var.run_id}-instance-profile"
  role = aws_iam_role.bench_instance_role.name

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name_prefix}-${var.run_id}-instance-profile"
    }
  )
}

# Security group for benchmark instances
resource "aws_security_group" "bench_sg" {
  name        = "${var.name_prefix}-${var.run_id}-sg"
  description = "Security group for Radix benchmark harness instances"
  vpc_id      = var.vpc_id

  # Outbound HTTPS for API calls, package downloads, SSM
  egress {
    description = "HTTPS outbound"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound HTTP for package repositories (optional, can be removed if not needed)
  egress {
    description = "HTTP outbound for package repos"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name_prefix}-${var.run_id}-sg"
    }
  )
}

# VPC Endpoints for private subnets (SSM Session Manager connectivity)
# Only created if subnet is private

resource "aws_security_group" "vpc_endpoints_sg" {
  count = local.is_public_subnet ? 0 : 1

  name        = "${var.name_prefix}-${var.run_id}-vpc-endpoints-sg"
  description = "Security group for VPC endpoints (SSM access from private subnet)"
  vpc_id      = var.vpc_id

  ingress {
    description     = "HTTPS from benchmark instances"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.bench_sg.id]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name_prefix}-${var.run_id}-vpc-endpoints-sg"
    }
  )
}

# SSM VPC Endpoint
resource "aws_vpc_endpoint" "ssm" {
  count = local.is_public_subnet ? 0 : 1

  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.region}.ssm"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [var.subnet_id]
  security_group_ids  = [aws_security_group.vpc_endpoints_sg[0].id]
  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name_prefix}-${var.run_id}-ssm-endpoint"
    }
  )
}

# SSM Messages VPC Endpoint
resource "aws_vpc_endpoint" "ssmmessages" {
  count = local.is_public_subnet ? 0 : 1

  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.region}.ssmmessages"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [var.subnet_id]
  security_group_ids  = [aws_security_group.vpc_endpoints_sg[0].id]
  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name_prefix}-${var.run_id}-ssmmessages-endpoint"
    }
  )
}

# EC2 Messages VPC Endpoint
resource "aws_vpc_endpoint" "ec2messages" {
  count = local.is_public_subnet ? 0 : 1

  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.region}.ec2messages"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [var.subnet_id]
  security_group_ids  = [aws_security_group.vpc_endpoints_sg[0].id]
  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name_prefix}-${var.run_id}-ec2messages-endpoint"
    }
  )
}

# IAM policy for SSM parameter access
resource "aws_iam_role_policy" "bench_ssm_params" {
  name = "${var.name_prefix}-${var.run_id}-ssm-params"
  role = aws_iam_role.bench_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters"
        ]
        Resource = "arn:aws:ssm:${var.region}:*:parameter/radix/bench/*"
      }
    ]
  })
}

# EC2 instance for benchmark harness
resource "aws_instance" "bench_instance" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.bench_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.bench_instance_profile.name

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    aws_region      = var.region
    radix_api_base  = var.radix_api_base
    tenant_id       = var.tenant_id
    cluster_id      = var.cluster_id
    ssm_token_param = var.ssm_token_param
    agent_image     = var.agent_image
    enable_agent    = var.enable_agent
  })

  user_data_replace_on_change = true
  disable_api_termination     = var.retain

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name_prefix}-${var.run_id}-instance"
      ClusterId = var.cluster_id
    }
  )
}
