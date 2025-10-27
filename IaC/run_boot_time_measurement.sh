#!/bin/bash

# Example script to measure spot instance boot times in us-west-2
# Security group will be automatically created and deleted

# Required: Set your AMI ID
AMI_ID="ami-xxxxxxxxxxxxxxxxx"  # Replace with your AMI ID

# Optional: Specify subnet (if not specified, uses default VPC)
# SUBNET_ID="subnet-xxxxxxxxxxxxxxxxx"

# Run measurement
echo "Starting spot instance boot time measurement..."
echo "AMI: $AMI_ID"
echo "Region: us-west-2"
echo "Instance types: g6.12xlarge, g5.12xlarge, g6e.xlarge"
echo ""

if [ -z "$SUBNET_ID" ]; then
    python3 measure_spot_boot_time.py \
        --ami-id "$AMI_ID" \
        --region us-west-2 \
        --instance-types g6.12xlarge g5.12xlarge g6e.xlarge \
        --iterations 3 \
        --output results_$(date +%Y%m%d_%H%M%S).json
else
    python3 measure_spot_boot_time.py \
        --ami-id "$AMI_ID" \
        --region us-west-2 \
        --subnet-id "$SUBNET_ID" \
        --instance-types g6.12xlarge g5.12xlarge g6e.xlarge \
        --iterations 3 \
        --output results_$(date +%Y%m%d_%H%M%S).json
fi

echo ""
echo "Measurement completed!"
