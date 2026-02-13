terraform {
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

data "aws_caller_identity" "me" {}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runner" {
  name               = "radix-actions-runner"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "runner" {
  statement {
    actions   = ["ssm:GetParameter", "ssm:GetParameters", "kms:Decrypt"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "runner" {
  name   = "radix-actions-runner-policy"
  policy = data.aws_iam_policy_document.runner.json
}

resource "aws_iam_role_policy_attachment" "attach" {
  role       = aws_iam_role.runner.name
  policy_arn = aws_iam_policy.runner.arn
}

resource "aws_iam_instance_profile" "runner" {
  name = "radix-actions-runner-profile"
  role = aws_iam_role.runner.name
}

resource "aws_security_group" "runner" {
  name        = "radix-runner-sg"
  vpc_id      = var.vpc_id
  description = "Runner SG"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

locals {
  labels = join(",", var.runner_labels)
  user_data_b64 = base64encode(templatefile("${path.module}/user_data.sh.tmpl", {
    region        = var.region
    github_owner  = var.repo_owner
    github_repo   = var.repo_name
    ssm_pat_param = var.ssm_pat_param
    runner_labels = local.labels
  }))
}

resource "aws_instance" "runner" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.runner.name
  vpc_security_group_ids = [aws_security_group.runner.id]
  user_data_base64       = local.user_data_b64
  key_name               = var.key_name
  monitoring             = false

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
      }
    }
  }

  tags = {
    Name = "radix-actions-runner"
  }

  root_block_device {
    volume_size           = 8
    volume_type           = "gp3"
    encrypted             = false
    delete_on_termination = true
  }
}

data "aws_ami" "al2023" {
  owners      = ["137112412989"]
  most_recent = true

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

output "runner_public_ip" {
  value = aws_instance.runner.public_ip
}

output "runner_instance_id" {
  value = aws_instance.runner.id
}
