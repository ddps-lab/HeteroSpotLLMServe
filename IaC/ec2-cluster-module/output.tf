output "instance_ids" {
  description = "Map of instance IDs by instance type and index"
  value = {
    for key, instance in module.shunt-worker-instance :
    key => instance.id
  }
}

output "instance_public_ips" {
  description = "Map of public IPs by instance type and index"
  value = {
    for key, instance in module.shunt-worker-instance :
    key => instance.public_ip
  }
}

output "instance_private_ips" {
  description = "Map of private IPs by instance type and index"
  value = {
    for key, instance in module.shunt-worker-instance :
    key => instance.private_ip
  }
}

output "head_instance_public_ip" {
  description = "Public IP of the head instance"
  value = module.shunt-head-instance.public_ip
}

output "head_instance_private_ip" {
  description = "Private IP of the head instance"
  value = module.shunt-head-instance.private_ip
}