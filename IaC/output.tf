output "head_instance_public_ip" {
  description = "Public IP of the head instance"
  value       = module.ec2-cluster.head_instance_public_ip
}

output "head_instance_private_ip" {
  description = "Private IP of the head instance"
  value       = module.ec2-cluster.head_instance_private_ip
}

output "instance_ids" {
  description = "Map of instance IDs by instance type and index"
  value       = module.ec2-cluster.instance_ids
}

output "instance_public_ips" {
  description = "Map of public IPs by instance type and index"
  value       = module.ec2-cluster.instance_public_ips
}

output "instance_private_ips" {
  description = "Map of private IPs by instance type and index"
  value       = module.ec2-cluster.instance_private_ips
}
