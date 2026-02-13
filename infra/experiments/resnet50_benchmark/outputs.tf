# Outputs for ResNet-50 benchmark GPU instance

output "instance_id" {
  description = "EC2 instance ID of the GPU benchmark instance"
  value       = aws_instance.gpu_benchmark.id
}

output "public_ip" {
  description = "Public IP address for SSH access to the GPU instance"
  value       = aws_instance.gpu_benchmark.public_ip
}

output "availability_zone" {
  description = "Availability zone where the instance is running"
  value       = aws_instance.gpu_benchmark.availability_zone
}

output "security_group_id" {
  description = "Security group ID attached to the instance"
  value       = aws_security_group.benchmark_sg.id
}

output "instance_type" {
  description = "Instance type (GPU configuration)"
  value       = aws_instance.gpu_benchmark.instance_type
}

output "ami_id" {
  description = "AMI ID used for the instance"
  value       = aws_instance.gpu_benchmark.ami
}

output "ssh_command" {
  description = "SSH command to connect to the instance (if key pair was provided)"
  value       = var.ssh_key_name != "" ? "ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${aws_instance.gpu_benchmark.public_ip}" : "No SSH key configured. Use AWS Systems Manager Session Manager instead."
}

output "ssm_command" {
  description = "AWS CLI command to connect via Systems Manager Session Manager"
  value       = "aws ssm start-session --target ${aws_instance.gpu_benchmark.id} --region ${var.aws_region}"
}
