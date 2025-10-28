module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "${var.prefix}-shunt-vpc"
  cidr = "192.168.0.0/16"

  azs             = ["${var.region}a", "${var.region}b"]
  public_subnets  = ["192.168.0.0/24", "192.168.1.0/24"]
  map_public_ip_on_launch = true
}

module "security-group" {
  source = "terraform-aws-modules/security-group/aws"

  name = "${var.prefix}-shunt-security-group"
  vpc_id = module.vpc.vpc_id
  ingress_with_cidr_blocks = [
    {
      from_port = 22
      to_port = 22
      protocol = "tcp"
      cidr_blocks = "0.0.0.0/0"
    }
  ]
  ingress_with_self = [
    {
      from_port = 0
      to_port = 0
      protocol = "-1"
    }
  ]
  egress_with_cidr_blocks = [
    {
      from_port = 0
      to_port = 0
      protocol = "-1"
      cidr_blocks = "0.0.0.0/0"
    }
  ]
}

