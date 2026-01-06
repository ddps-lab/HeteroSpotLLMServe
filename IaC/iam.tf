resource "aws_iam_role" "admin-role" {
    name = "${var.prefix}-shunt-admin-role"
    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Action = "sts:AssumeRole"
                Effect = "Allow"
                Principal = {
                    Service = "ec2.amazonaws.com"
                }
            }
        ]
    })
}

resource "aws_iam_role_policy_attachment" "admin-role-policy-attachment" {
    role = aws_iam_role.admin-role.name
    policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

resource "aws_iam_instance_profile" "admin-instance-profile" {
    name = "${var.prefix}-shunt-admin-instance-profile"
    role = aws_iam_role.admin-role.name
}
