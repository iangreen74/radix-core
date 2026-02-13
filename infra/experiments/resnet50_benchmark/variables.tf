# Variables for ResNet-50 benchmark GPU instance

variable "aws_region" {
  description = "AWS region to deploy the GPU instance"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type for GPU benchmarking.
    Recommended options:
    - g5.xlarge: 1 GPU (A10G), 4 vCPUs, 16 GB RAM - good for smoke tests
    - g5.2xlarge: 1 GPU (A10G), 8 vCPUs, 32 GB RAM
    - g5.12xlarge: 4 GPUs (A10G), 48 vCPUs, 192 GB RAM - for full benchmark
    - p3.2xlarge: 1 GPU (V100), 8 vCPUs, 61 GB RAM
    - p3.8xlarge: 4 GPUs (V100), 32 vCPUs, 244 GB RAM
  EOT
  type        = string
  default     = "g5.xlarge"
}

variable "ssh_key_name" {
  description = <<-EOT
    (Optional) Name of an existing EC2 key pair for SSH access.
    If not provided, you can still access the instance via AWS Systems Manager Session Manager.
    Example: "my-ec2-keypair"
  EOT
  type        = string
  default     = ""
}

variable "ssh_allowed_cidr" {
  description = <<-EOT
    CIDR block allowed to SSH into the instance.
    WARNING: Default is 0.0.0.0/0 (open to the world) for convenience.
    CHANGE THIS for production use to your specific IP or VPN CIDR.
    Example: "203.0.113.0/24" or use your IP with /32
  EOT
  type        = string
  default     = "0.0.0.0/0"
}

variable "ami_id" {
  description = <<-EOT
    REQUIRED: The GPU EC2 AMI ID to use for the ResNet-50 benchmark instance.
    Recommended: a Deep Learning GPU AMI with PyTorch on Ubuntu in the target region.
    Example (us-west-2): "ami-0123456789abcdef0"
    
    To find a suitable AMI:
    - Go to EC2 Console → AMIs
    - Set filters: Public images, Owner = Amazon
    - Search for "Deep Learning AMI GPU PyTorch Ubuntu"
    - Copy the AMI ID from your target region
  EOT
  type        = string
}
