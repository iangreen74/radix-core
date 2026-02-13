# Radix Infrastructure - AWS EKS Cluster for GPU Orchestration
# This is a FRAMEWORK ONLY - not for actual deployment
# All resources are designed for research and development purposes

terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

# Provider configuration
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "radix"
      Environment = var.environment
      ManagedBy   = "terraform"
      Purpose     = "research-framework"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# AWS caller identity removed - not used in this configuration

# Local values
locals {
  cluster_name = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
