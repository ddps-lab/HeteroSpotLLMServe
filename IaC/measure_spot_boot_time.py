#!/usr/bin/env python3
"""
Measure time from spot instance request to SSH-ready state.
Supports multiple instance types and configurable AMI.
"""

import boto3
import time
import argparse
import socket
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import random
import string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpotInstanceBootTimer:
    def __init__(
        self,
        region: str = 'us-west-2',
        ami_id: str = None,
        security_group_id: str = None,
        subnet_id: str = None
    ):
        """
        Initialize the boot time measurement tool.

        Args:
            region: AWS region (default: us-west-2)
            ami_id: AMI ID to use for instances
            security_group_id: Security group ID (optional, will be auto-created if not specified)
            subnet_id: Subnet ID (optional, uses default if not specified)
        """
        self.region = region
        self.ami_id = ami_id
        self.security_group_id = security_group_id
        self.subnet_id = subnet_id
        self.created_security_group_id = None  # Track if we created a security group

        self.ec2_client = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)

    def _generate_random_name(self, prefix: str = 'spot-boot-timer') -> str:
        """Generate a random name with prefix."""
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{prefix}-{random_suffix}"

    def _get_vpc_id(self) -> str:
        """Get VPC ID from subnet or default VPC."""
        if self.subnet_id:
            # Get VPC ID from subnet
            response = self.ec2_client.describe_subnets(SubnetIds=[self.subnet_id])
            return response['Subnets'][0]['VpcId']
        else:
            # Get default VPC
            response = self.ec2_client.describe_vpcs(
                Filters=[{'Name': 'is-default', 'Values': ['true']}]
            )
            if response['Vpcs']:
                return response['Vpcs'][0]['VpcId']
            else:
                raise Exception("No default VPC found. Please specify --subnet-id")

    def create_security_group(self) -> str:
        """
        Create a temporary security group that allows SSH access.

        Returns:
            Security group ID
        """
        vpc_id = self._get_vpc_id()
        sg_name = self._generate_random_name()

        logger.info(f"Creating temporary security group: {sg_name}")

        try:
            response = self.ec2_client.create_security_group(
                GroupName=sg_name,
                Description='Temporary security group for spot instance boot time measurement',
                VpcId=vpc_id
            )
            sg_id = response['GroupId']

            # Add SSH ingress rule
            self.ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access for boot time measurement'}]
                    }
                ]
            )

            logger.info(f"Created security group: {sg_id}")
            self.created_security_group_id = sg_id
            return sg_id

        except Exception as e:
            logger.error(f"Failed to create security group: {e}")
            raise

    def delete_security_group(self, sg_id: str):
        """Delete a security group."""
        try:
            logger.info(f"Deleting security group: {sg_id}")
            self.ec2_client.delete_security_group(GroupId=sg_id)
            logger.info(f"Deleted security group: {sg_id}")
        except Exception as e:
            logger.error(f"Failed to delete security group {sg_id}: {e}")

    def cleanup_resources(self):
        """Clean up any resources created during measurement."""
        if self.created_security_group_id:
            self.delete_security_group(self.created_security_group_id)
            self.created_security_group_id = None

    def check_port_open(self, host: str, port: int = 22, timeout: int = 0.1) -> bool:
        """Check if a port is open on the given host."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Port check failed: {e}")
            return False

    def test_ssh_ready(self, host: str) -> bool:
        """Test if SSH port is open and ready."""
        return self.check_port_open(host, port=22, timeout=0.1)

    def request_spot_instance(
        self,
        instance_type: str,
        max_price: str = None
    ) -> Tuple[str, datetime]:
        """
        Request a spot instance.

        Returns:
            Tuple of (spot_request_id, request_time)
        """
        request_time = datetime.now()

        # Prepare launch specification
        launch_spec = {
            'ImageId': self.ami_id,
            'InstanceType': instance_type,
        }

        if self.security_group_id:
            launch_spec['SecurityGroupIds'] = [self.security_group_id]

        if self.subnet_id:
            launch_spec['SubnetId'] = self.subnet_id

        # Prepare spot request parameters
        spot_params = {
            'InstanceCount': 1,
            'Type': 'one-time',
            'LaunchSpecification': launch_spec,
        }

        if max_price:
            spot_params['SpotPrice'] = max_price

        logger.info(f"Requesting spot instance: {instance_type}")
        response = self.ec2_client.request_spot_instances(**spot_params)

        spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        logger.info(f"Spot request created: {spot_request_id}")

        return spot_request_id, request_time

    def wait_for_instance_fulfillment(
        self,
        spot_request_id: str,
        timeout: int = 600
    ) -> Tuple[str, datetime]:
        """
        Wait for spot request to be fulfilled.

        Returns:
            Tuple of (instance_id, fulfillment_time)
        """
        logger.info("Waiting for spot request fulfillment...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.ec2_client.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )

            request = response['SpotInstanceRequests'][0]
            status = request['Status']['Code']

            if status == 'fulfilled':
                fulfillment_time = datetime.now()
                instance_id = request['InstanceId']
                logger.info(f"Spot request fulfilled: {instance_id}")
                return instance_id, fulfillment_time
            elif status in ['cancelled', 'closed', 'failed']:
                raise Exception(f"Spot request failed with status: {status}")

            time.sleep(2)

        raise TimeoutError("Timeout waiting for spot request fulfillment")

    def wait_for_instance_running(
        self,
        instance_id: str,
        timeout: int = 600
    ) -> Tuple[str, datetime]:
        """
        Wait for instance to reach running state.

        Returns:
            Tuple of (public_ip, running_time)
        """
        logger.info("Waiting for instance to reach running state...")
        instance = self.ec2_resource.Instance(instance_id)

        start_time = time.time()
        while time.time() - start_time < timeout:
            instance.reload()
            if instance.state['Name'] == 'running':
                running_time = datetime.now()
                public_ip = instance.public_ip_address
                logger.info(f"Instance running with IP: {public_ip}")
                return public_ip, running_time
            time.sleep(2)

        raise TimeoutError("Timeout waiting for instance to run")

    def wait_for_ssh_ready(
        self,
        public_ip: str,
        timeout: int = 600
    ) -> datetime:
        """
        Wait for SSH port to be open and ready.

        Returns:
            ssh_ready_time
        """
        logger.info("Waiting for SSH port to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.test_ssh_ready(public_ip):
                ssh_ready_time = datetime.now()
                logger.info("SSH port is ready!")
                return ssh_ready_time
            time.sleep(0.1)

        raise TimeoutError("Timeout waiting for SSH port to be ready")

    def terminate_instance(self, instance_id: str):
        """Terminate the instance."""
        logger.info(f"Terminating instance: {instance_id}")
        self.ec2_client.terminate_instances(InstanceIds=[instance_id])

    def measure_boot_time(
        self,
        instance_type: str,
        max_price: str = None,
        cleanup: bool = True
    ) -> Dict:
        """
        Measure complete boot time for a spot instance.

        Returns:
            Dictionary with timing information
        """
        instance_id = None

        try:
            # 1. Request spot instance
            spot_request_id, request_time = self.request_spot_instance(
                instance_type, max_price
            )

            # 2. Wait for fulfillment
            instance_id, fulfillment_time = self.wait_for_instance_fulfillment(
                spot_request_id
            )

            # 3. Wait for running state
            public_ip, running_time = self.wait_for_instance_running(instance_id)

            # 4. Wait for SSH ready
            ssh_ready_time = self.wait_for_ssh_ready(public_ip)

            # Calculate durations
            fulfillment_duration = (fulfillment_time - request_time).total_seconds()
            running_duration = (running_time - request_time).total_seconds()
            ssh_ready_duration = (ssh_ready_time - request_time).total_seconds()

            results = {
                'instance_type': instance_type,
                'instance_id': instance_id,
                'public_ip': public_ip,
                'request_time': request_time.isoformat(),
                'fulfillment_time': fulfillment_time.isoformat(),
                'running_time': running_time.isoformat(),
                'ssh_ready_time': ssh_ready_time.isoformat(),
                'time_to_fulfillment': fulfillment_duration,
                'time_to_running': running_duration,
                'time_to_ssh_ready': ssh_ready_duration,
                'ami_id': self.ami_id,
            }

            logger.info("\n" + "="*60)
            logger.info(f"Boot Time Measurement Results for {instance_type}")
            logger.info("="*60)
            logger.info(f"Time to fulfillment: {fulfillment_duration:.2f} seconds")
            logger.info(f"Time to running state: {running_duration:.2f} seconds")
            logger.info(f"Time to SSH ready: {ssh_ready_duration:.2f} seconds")
            logger.info("="*60 + "\n")

            return results

        except Exception as e:
            logger.error(f"Error measuring boot time: {e}")
            raise

        finally:
            if cleanup and instance_id:
                try:
                    self.terminate_instance(instance_id)
                except Exception as e:
                    logger.error(f"Error terminating instance: {e}")

    def measure_multiple_instance_types(
        self,
        instance_types: List[str],
        max_price: str = None,
        cleanup: bool = True,
        iterations: int = 1
    ) -> List[Dict]:
        """
        Measure boot times for multiple instance types.

        Args:
            instance_types: List of instance types to test
            max_price: Max spot price
            cleanup: Whether to terminate instances after measurement
            iterations: Number of iterations per instance type

        Returns:
            List of result dictionaries
        """
        all_results = []

        # Create security group if not provided
        if not self.security_group_id:
            self.security_group_id = self.create_security_group()

        try:
            for instance_type in instance_types:
                for iteration in range(iterations):
                    logger.info(f"\nMeasuring {instance_type} (iteration {iteration + 1}/{iterations})")
                    try:
                        result = self.measure_boot_time(
                            instance_type=instance_type,
                            max_price=max_price,
                            cleanup=cleanup
                        )
                        result['iteration'] = iteration + 1
                        all_results.append(result)

                        # Wait a bit between iterations to avoid throttling
                        if iteration < iterations - 1:
                            time.sleep(10)

                    except Exception as e:
                        logger.error(f"Failed to measure {instance_type}: {e}")
                        continue

        finally:
            # Clean up security group if we created it
            self.cleanup_resources()

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Measure spot instance boot time to SSH-ready state'
    )
    parser.add_argument(
        '--region',
        default='us-west-2',
        help='AWS region (default: us-west-2)'
    )
    parser.add_argument(
        '--ami-id',
        required=True,
        help='AMI ID to use for instances'
    )
    parser.add_argument(
        '--security-group-id',
        help='Security group ID (optional, uses default VPC security group if not specified)'
    )
    parser.add_argument(
        '--subnet-id',
        help='Subnet ID (optional, uses default subnet if not specified)'
    )
    parser.add_argument(
        '--instance-types',
        nargs='+',
        default=['g6.12xlarge', 'g5.12xlarge', 'g6e.xlarge'],
        help='Instance types to test (default: g6.12xlarge g5.12xlarge g6e.xlarge)'
    )
    parser.add_argument(
        '--max-price',
        help='Maximum spot price (optional)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations per instance type (default: 1)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not terminate instances after measurement'
    )
    parser.add_argument(
        '--output',
        help='Output file for results (JSON format)'
    )

    args = parser.parse_args()

    # Initialize timer
    timer = SpotInstanceBootTimer(
        region=args.region,
        ami_id=args.ami_id,
        security_group_id=args.security_group_id,
        subnet_id=args.subnet_id
    )

    # Measure boot times
    results = timer.measure_multiple_instance_types(
        instance_types=args.instance_types,
        max_price=args.max_price,
        cleanup=not args.no_cleanup,
        iterations=args.iterations
    )

    # Save results if output file specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in results:
        print(f"\n{result['instance_type']} (iteration {result['iteration']}):")
        print(f"  Time to SSH ready: {result['time_to_ssh_ready']:.2f} seconds")
        print(f"  - Fulfillment: {result['time_to_fulfillment']:.2f}s")
        print(f"  - Running: {result['time_to_running']:.2f}s")
        print(f"  - SSH ready: {result['time_to_ssh_ready']:.2f}s")
    print("="*80)


if __name__ == '__main__':
    main()
