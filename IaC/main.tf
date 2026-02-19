module "ec2-cluster" {
  source = "./ec2-cluster-module"

  ami_id                      = var.ami_id
  key_name                    = var.key_name
  prefix                      = var.prefix
  head_instance_type          = var.head_instance_type
  security_group_id           = module.security-group.security_group_id
  subnet_id                   = module.vpc.public_subnets[0]
  s3_instance_profile_name    = aws_iam_instance_profile.s3-instance-profile.name
  hf_token                    = var.hf_token

  instance_type_count = {
    "g6.xlarge" = 5   # 3 initial + 2 replacement
  }
}
