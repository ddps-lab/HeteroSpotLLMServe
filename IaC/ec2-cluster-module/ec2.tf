module "shunt-head-instance" {
  source = "terraform-aws-modules/ec2-instance/aws"

  name = "${var.prefix}-shunt-head-instance"

  ami                    = var.ami_id
  instance_type          = "m5.large"
  key_name               = var.key_name
  monitoring             = true
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group_id]

  user_data = <<EOF
#!/bin/bash
sudo su
su ubuntu
cd /home/ubuntu/HeteroSpotLLMServe
git pull
EOF

  iam_instance_profile = var.admin_instance_profile_name

  tags = {
    Name = "${var.prefix}-shunt-head-instance"
  }
}

locals {
  instances = merge([
    for instance_type, count in var.instance_type_count : {
      for i in range(count) :
      "${instance_type}-${i}" => instance_type
    }
  ]...)
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

  iam_instance_profile = var.admin_instance_profile_name
  user_data = <<EOF
#!/bin/bash
sudo su
su ubuntu
cd /home/ubuntu/HeteroSpotLLMServe
git pull
EOF

  tags = {
    Name         = "${var.prefix}-shunt-worker-${each.key}"
    InstanceType = each.value
    Index        = split("-", each.key)[length(split("-", each.key)) - 1]
  }

}
