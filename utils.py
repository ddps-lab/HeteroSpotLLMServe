import ray
from typing import Dict, List, Optional
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def run_server_remote(hostname: str, username: str, command_to_execute: str, log_file: str = "tensor_server.log", 
                      ssh_key_path: Optional[str] = None, ssh_port: int = 22) -> tuple[bool, Optional[subprocess.Popen]]:
    """
    Disables fingerprint verification on initial connection.

    :param hostname: Remote server address (IP or domain name)
    :param username: SSH username
    :param command_to_execute: Full command string to execute on the remote server (including cd, source, python, etc.)
    :param log_file: Log file path on the remote server to store output
    :param ssh_key_path: (Optional) Path to a specific SSH private key file to use
    :param ssh_port: (Optional) SSH port
    :return: Success status (True/False), message or error
    """
    if not command_to_execute:
        logger.error(f"[{hostname}] No command to execute was provided.")
        return False, "No command to execute was provided."

    # Basic SSH command configuration
    ssh_command_parts = [
        "ssh",
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",  # Disable fingerprint verification (or accept-new)
        # "-o", "UserKnownHostsFile=/dev/null", # Do not use known_hosts file (stronger disabling)
    ]

    # If using a specific SSH key
    if ssh_key_path:
        ssh_command_parts.extend(["-i", ssh_key_path])

    # Append user@host
    ssh_command_parts.append(f"{username}@{hostname}")

    log_file_dir = os.path.dirname(log_file)
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)

    # Construct foreground command to execute remotely (including log redirection)
    remote_shell_command = f"{command_to_execute} > {log_file} 2>&1"
    
    # Final SSH command (pass the remote shell command as an argument)
    ssh_command_parts.append(remote_shell_command)

    try:
        # Popen starts the process asynchronously.
        process = subprocess.Popen(ssh_command_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Since & is not used, the SSH session is maintained until the remote command terminates.
        # Because mt_tensor_store_server.py is a server that does not terminate on its own, this SSH session will remain open.
        logger.info(f"[{hostname}] Submitted remote command in foreground. Check the log file '{log_file}' on the remote server.")
        logger.info(f"[{hostname}] SSH process PID: {process.pid}. This session will be maintained while the remote server is running.")
        return True, process

    except FileNotFoundError:
        # When the ssh command cannot be found (SSH client is not installed on the local system)
        logger.error(f"[{hostname}] Error: 'ssh' command not found. Please verify that an SSH client is installed.")
        return False, None
    except subprocess.CalledProcessError as e:
        # Raised when check=True is used (not directly raised here since Popen is used; determined by returncode after communicate)
        logger.error(f"[{hostname}] Error occurred during command execution: {e}")
        logger.error(f"STDOUT: {e.stdout.decode().strip() if e.stdout else ''}")
        logger.error(f"STDERR: {e.stderr.decode().strip() if e.stderr else ''}")
        return False, f"Command execution error: {e.stderr.decode().strip() if e.stderr else str(e)}"
    except Exception as e:
        logger.error(f"[{hostname}] Unexpected error occurred: {e}")
        return False, None

def create_placement_group_and_bundle_indices(node_rank_mapping: Dict[str, List[int]], ray_address: str):
    logger.info(f"Creating placement group and bundle indices for node rank mapping: {node_rank_mapping}")

    if not ray.is_initialized():
        logger.info(f"Initialize Ray with address: {ray_address} at create_placement_group_and_bundle_indices")
        ray.init(address=ray_address)

    # Calculate total number of ranks and create rank-to-IP mapping
    total_ranks = 0
    rank_to_ip = {}
    for ip, ranks in node_rank_mapping.items():
        num_ranks_on_ip = len(ranks)
        total_ranks += num_ranks_on_ip
        for rank in ranks:
            if rank in rank_to_ip:
                 raise ValueError(f"Rank {rank} is assigned to multiple IPs. Ranks must be unique.")
            rank_to_ip[rank] = ip

    # Verify that ranks are contiguous from 0 to total_ranks-1
    if set(rank_to_ip.keys()) != set(range(total_ranks)):
        raise ValueError(f"Ranks must be contiguous from 0 to {total_ranks - 1}. Found ranks: {sorted(rank_to_ip.keys())}")

    # Create placement group spec for each rank
    placement_group_specs: List[Dict[str, float]] = []
    for rank in range(total_ranks):
        ip = rank_to_ip[rank]
        placement_group_specs.append({
            'GPU': 1.0,
            f"node:{ip}": 0.001 # Indicates that a specific node should be used
        })
    
    # If strategy is set to STRICT_SPREAD, all ranks must be distributed to different nodes.
    # This causes an infinite wait in ray.get. Changed to PACK instead.
    placement_group = ray.util.placement_group(placement_group_specs, strategy="PACK")
    ray.get(placement_group.ready())

    # Map created bundles to assigned node IPs
    bundle_to_node = {} # { bundle_idx: node_ip }
    for bundle_id, bundle in enumerate(placement_group.bundle_specs):
        for resource_key in bundle:
            if resource_key.startswith("node:"):
                node_ip = resource_key[5:] # 'node:172.31.16.230' -> '172.31.16.230'
                bundle_to_node[bundle_id] = node_ip
                break # Move to the next bundle once the node IP is found

    # Assign bundle indices according to rank order
    bundle_indices = [None] * total_ranks
    # List of unassigned ranks per IP (sorted)
    remaining_ranks_for_ip = {ip: sorted(ranks) for ip, ranks in node_rank_mapping.items()}
    assigned_bundles = set() # Track already assigned bundles

    for bundle_idx in range(len(placement_group.bundle_specs)):
        if bundle_idx not in bundle_to_node:
            raise RuntimeError(f"Bundle {bundle_idx} is not assigned to any node.")

        # Extract the bundle index and its corresponding node IP.
        node_ip = bundle_to_node[bundle_idx]

        if node_ip in remaining_ranks_for_ip and remaining_ranks_for_ip[node_ip]:
            # Get the lowest rank that needs to be assigned to this IP
            assigned_rank = remaining_ranks_for_ip[node_ip].pop(0)
            if bundle_indices[assigned_rank] is not None:
                 raise RuntimeError(f"Rank {assigned_rank} is already assigned to bundle {bundle_indices[assigned_rank]}. Trying to assign bundle {bundle_idx}.")
            bundle_indices[assigned_rank] = str(bundle_idx)
            assigned_bundles.add(bundle_idx) # Record as assigned bundle

            # Remove from the dictionary if all ranks for this IP have been assigned
            if not remaining_ranks_for_ip[node_ip]:
                del remaining_ranks_for_ip[node_ip]
        else:
            # No matching rank found for this bundle (logic error or placement group creation issue)
             raise RuntimeError(f"Could not find a rank assignment for bundle {bundle_idx} on node {node_ip}. Remaining ranks: {remaining_ranks_for_ip}")


    # Verify that all ranks have been assigned to bundles
    if None in bundle_indices:
        unassigned_ranks = [i for i, b in enumerate(bundle_indices) if b is None]
        raise RuntimeError(f"Could not assign bundles to all ranks. Unassigned ranks: {unassigned_ranks}. Assignments: {bundle_indices}")
    if remaining_ranks_for_ip:
         raise RuntimeError(f"Some ranks were not assigned bundles: {remaining_ranks_for_ip}")
    if len(assigned_bundles) != len(placement_group.bundle_specs):
        raise RuntimeError(f"Number of assigned bundles ({len(assigned_bundles)}) does not match total bundles ({len(placement_group.bundle_specs)}).")


    # Set and print environment variables
    os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(bundle_indices)
    print(f"VLLM_RAY_BUNDLE_INDICES is setted to {os.environ['VLLM_RAY_BUNDLE_INDICES']}")
    print(f"bundle specs : {placement_group.bundle_specs}")

    return placement_group
