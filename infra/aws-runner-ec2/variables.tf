variable "region" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "t3.small"
}

variable "use_spot" {
  type    = bool
  default = true
}

variable "runner_labels" {
  type    = list(string)
  default = ["self-hosted", "linux", "x64", "radix", "aws-ec2"]
}

variable "repo_owner" {
  type = string
}

variable "repo_name" {
  type = string
}

variable "ssm_pat_param" {
  type = string
}

variable "key_name" {
  type    = string
  default = null
}
