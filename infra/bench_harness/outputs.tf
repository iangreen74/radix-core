output "bench_security_group_id" {
  description = "Security group ID for benchmark harness instances (maps to BENCH_SECURITY_GROUP_ID GitHub secret)"
  value       = aws_security_group.bench_sg.id
}

output "bench_instance_profile_name" {
  description = "Instance profile name for benchmark harness instances (maps to BENCH_INSTANCE_PROFILE_NAME GitHub secret)"
  value       = aws_iam_instance_profile.bench_instance_profile.name
}

output "bench_instance_profile_arn" {
  description = "Instance profile ARN for benchmark harness instances"
  value       = aws_iam_instance_profile.bench_instance_profile.arn
}

output "bench_iam_role_name" {
  description = "IAM role name for benchmark harness instances"
  value       = aws_iam_role.bench_instance_role.name
}

output "bench_iam_role_arn" {
  description = "IAM role ARN for benchmark harness instances"
  value       = aws_iam_role.bench_instance_role.arn
}

output "is_public_subnet" {
  description = "Whether the subnet is public (has IGW route) or private"
  value       = local.is_public_subnet
}

output "vpc_endpoints_created" {
  description = "Whether VPC endpoints were created (true for private subnets)"
  value       = !local.is_public_subnet
}

output "bench_instance_id" {
  description = "EC2 instance ID for benchmark harness"
  value       = aws_instance.bench_instance.id
}

output "bench_public_ip" {
  description = "Public IP address of benchmark instance (if in public subnet)"
  value       = aws_instance.bench_instance.public_ip
}

output "bench_private_ip" {
  description = "Private IP address of benchmark instance"
  value       = aws_instance.bench_instance.private_ip
}
