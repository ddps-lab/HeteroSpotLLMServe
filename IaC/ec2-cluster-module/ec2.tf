locals {
  hf_user_data = var.hf_token != "" ? "#!/bin/bash\necho 'HF_TOKEN=${var.hf_token}' >> /etc/environment\necho 'export HF_TOKEN=${var.hf_token}' >> /home/ubuntu/.bashrc" : null

  instances = merge([
    for instance_type, count in var.instance_type_count : {
      for i in range(count) :
      "${instance_type}-${i}" => instance_type
    }
  ]...)
}

module "shunt-head-instance" {
  source = "terraform-aws-modules/ec2-instance/aws"

  name = "${var.prefix}-shunt-head-instance"

  ami                    = var.ami_id
  instance_type          = var.head_instance_type
  key_name               = var.key_name
  monitoring             = true
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group_id]

  user_data            = local.hf_user_data
  iam_instance_profile = var.s3_instance_profile_name

  tags = {
    Name = "${var.prefix}-shunt-head-instance"
  }
}

module "shunt-worker-instance" {
  source = "terraform-aws-modules/ec2-instance/aws"

  for_each = local.instances

  name                   = "${var.prefix}-shunt-worker-${each.key}"
  ami                    = var.ami_id
  instance_type          = each.value
  key_name               = var.key_name
  monitoring             = true
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group_id]

  user_data            = local.hf_user_data
  iam_instance_profile = var.s3_instance_profile_name

  tags = {
    Name         = "${var.prefix}-shunt-worker-${each.key}"
    InstanceType = each.value
    Index        = split("-", each.key)[length(split("-", each.key)) - 1]
  }

}
