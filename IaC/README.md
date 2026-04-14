# Infrastructure as Code

Terraform configuration for provisioning the ShuntServe evaluation cluster on AWS. Creates a VPC, security group, IAM role, one head node, and a configurable set of GPU worker instances.

All instances are on-demand. Spot interruptions are simulated at the application level -- no actual spot instances are used.

## Provisioned Resources

| Resource | Details |
|---|---|
| VPC | `192.168.0.0/16`, 2 AZs, 2 public subnets |
| Security Group | SSH (port 22) from anywhere + all intra-cluster traffic |
| IAM Instance Profile | `AmazonS3FullAccess` (for S3 model weight access) |
| Head instance | 1x `m5.large` (default, configurable via `head_instance_type`) |
| Worker instances | Configurable via `instance_type_count` map |

Example worker configuration:

| Instance Type | GPU | Count |
|---|---|---|
| g5.12xlarge | 4x NVIDIA A10G | 2 |
| g6.12xlarge | 4x NVIDIA L4 | 3 |
| g6e.xlarge | 1x NVIDIA L40S | 4 |

Worker instances are not fixed -- you can provision any combination of instance types and counts by editing the `instance_type_count` map in `main.tf`.

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) installed
- AWS CLI configured with a named profile (`aws configure --profile <name>`)
- An EC2 Key Pair created in your target region
- A pre-built AMI with CUDA, NCCL, Python, and vLLM installed. See the project root [README.md](../README.md) for environment setup instructions (to be added).

## Configuration

Copy the sample variable file and fill in your values:

```bash
cp var.tf.sample var.tf
```

Edit `var.tf`:

| Variable | Description | Example |
|---|---|---|
| `prefix` | Naming prefix for all resources | `"my-shuntserve"` |
| `region` | AWS region | `"us-west-2"` |
| `awscli_profile` | AWS CLI profile name | `"my-profile"` |
| `ami_id` | AMI ID with pre-installed environment | `"ami-0abc..."` |
| `key_name` | EC2 Key Pair name for SSH access | `"my-keypair"` |
| `hf_token` | HuggingFace token for gated models (Llama) | `"hf_abc..."` |

`var.tf` is gitignored so your credentials and configuration are never committed.

## Customizing the Cluster

### Worker instances

Edit `instance_type_count` in `main.tf` to provision any combination of instance types:

```hcl
instance_type_count = {
  "g5.12xlarge" = 2
  "g6.12xlarge" = 3
  "g6e.xlarge"  = 4
}
```

For SpotInterruption experiments, you need additional replacement instances. Increase the counts accordingly:

```hcl
instance_type_count = {
  "g5.12xlarge" = 4   # 2 initial + 2 replacement
  "g6.12xlarge" = 5   # 3 initial + 2 replacement
  "g6e.xlarge"  = 8   # 4 initial + 4 replacement
}
```

For the UnitTest8B minimum functional test (see [`ArtifactEvaluation/SpotTolerance/UnitTest8B`](../ArtifactEvaluation/SpotTolerance/UnitTest8B)):

```hcl
instance_type_count = {
  "g6.xlarge" = 5   # 3 initial + 2 replacement
}
```

### Head instance

The head instance type defaults to `m5.large`. To change it, pass `head_instance_type` to the module in `main.tf`:

```hcl
module "ec2-cluster" {
  source = "./ec2-cluster-module"
  # ...
  head_instance_type = "m5.xlarge"
}
```

## Usage

```bash
cd IaC

# Initialize Terraform and download providers
terraform init

# Preview the infrastructure changes
terraform plan

# Create the cluster
terraform apply

# View instance IPs (use these in ArtifactEvaluation/*/nodes.py)
terraform output

# Tear down the cluster
terraform destroy
```

## Outputs

After `terraform apply`, the following outputs are available:

| Output | Description |
|---|---|
| `head_instance_public_ip` | Public IP of the head node |
| `head_instance_private_ip` | Private IP of the head node |
| `instance_ids` | Map of `{instance_type-index => instance_id}` for all workers |
| `instance_public_ips` | Map of `{instance_type-index => public_ip}` for all workers |
| `instance_private_ips` | Map of `{instance_type-index => private_ip}` for all workers |

Use the private IPs from `instance_private_ips` to fill in the `nodes.py` files in `ArtifactEvaluation/`. See [ArtifactEvaluation/README.md](../ArtifactEvaluation/README.md) Step 5 for details.

## Directory Structure

```
IaC/
  main.tf                        # Root module: wires up ec2-cluster-module
  provider.tf                    # AWS provider configuration
  vpc.tf                         # VPC + Security Group
  iam.tf                         # IAM Role + Instance Profile
  var.tf.sample                  # Variable template (copy to var.tf and fill in)
  ec2-cluster-module/            # Reusable EC2 cluster module
    ec2.tf                       #   Head + worker EC2 instance definitions
    variable.tf                  #   Module input variables
    output.tf                    #   Module outputs (IPs, IDs)
  measure_spot_boot_time.py      # Spot boot time measurement tool (research utility)
  run_boot_time_measurement.sh   # Shell wrapper for the measurement tool
```

## Appendix: Spot Boot Time Measurement

`measure_spot_boot_time.py` is a standalone research utility (not part of the cluster provisioning). It measures how long spot instances of each GPU type take to become SSH-ready, from request to fulfillment to running to SSH connection.

Edit `run_boot_time_measurement.sh` and set `AMI_ID` to your AMI ID, then run:

```bash
bash run_boot_time_measurement.sh
```

Results are saved to `results_YYYYMMDD_HHMMSS.json`.
