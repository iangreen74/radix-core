variable "vpc_id" {
  description = "VPC ID where benchmark harness will run"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for benchmark harness instances (public or private)"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "radix-bench"
}

variable "run_id" {
  description = "Unique run identifier (e.g., GitHub Actions run ID)"
  type        = string
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

variable "ami_id" {
  description = "AMI ID for benchmark instance"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g5.2xlarge"
}

variable "radix_api_base" {
  description = "Radix API base URL"
  type        = string
}

variable "tenant_id" {
  description = "Radix tenant ID"
  type        = string
}

variable "cluster_id" {
  description = "Radix cluster ID"
  type        = string
}

variable "ssm_token_param" {
  description = "SSM parameter name containing onboarding token"
  type        = string
}

variable "agent_image" {
  description = "Radix agent Docker image"
  type        = string
  default     = "ghcr.io/iangreen74/radix-cluster-agent:latest"
}

variable "retain" {
  description = "Enable API termination protection"
  type        = bool
  default     = false
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 150
}

variable "enable_agent" {
  description = "Whether to bootstrap/run the Radix cluster agent"
  type        = bool
  default     = true
}
