# Variables for Radix Infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "radix"
}

variable "cluster_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.28"
}

variable "node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    min_size       = number
    max_size       = number
    desired_size   = number
    capacity_type  = string
    labels         = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))

  default = {
    # CPU nodes for general workloads
    cpu_nodes = {
      instance_types = ["m5.large", "m5.xlarge"]
      min_size       = 1
      max_size       = 10
      desired_size   = 2
      capacity_type  = "ON_DEMAND"
      labels = {
        "node-type" = "cpu"
        "workload"  = "general"
      }
      taints = []
    }

    # GPU nodes for ML workloads (research framework)
    gpu_nodes = {
      instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]
      min_size       = 0
      max_size       = 5
      desired_size   = 0      # Start with 0 for cost control
      capacity_type  = "SPOT" # Use spot for cost efficiency
      labels = {
        "node-type"     = "gpu"
        "workload"      = "ml"
        "gpu-type"      = "nvidia-t4"
        "radix-capable" = "true"
      }
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

variable "enable_gpu_operator" {
  description = "Enable NVIDIA GPU Operator"
  type        = bool
  default     = true
}

variable "enable_karpenter" {
  description = "Enable Karpenter for node autoscaling"
  type        = bool
  default     = true
}

variable "enable_kueue" {
  description = "Enable Kueue for job queueing"
  type        = bool
  default     = true
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}
