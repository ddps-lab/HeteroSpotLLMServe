import os
from typing import Optional, List

# Global configuration
DEFAULT_TENSOR_STORE_BASE_PORT = 10001
DEFAULT_API_SERVER_BASE_PORT = 8001

PYTHON = "/usr/bin/python"
PROJECT_PATH = "/home/ubuntu/HeteroSpotLLMServe"

def get_tensor_store_command(model_name: str, 
                            tensor_parallel_size: int,
                            local_rank: int, 
                            start_layer_id: int,
                            end_layer_id: int,
                            status_port: Optional[int] = None,
                            dtype: Optional[str] = None,
                            s3_path: Optional[str] = None,
                            aws_profile: Optional[str] = None) -> str:
    """
    Generate command to start tensor store server.
    
    Args:
        model_name: Name of the model (e.g., 'meta-llama/Llama-2-7b-hf')
        tensor_parallel_size: Number of GPUs for tensor parallelism
        local_rank: Local GPU rank (0 to tensor_parallel_size-1)
        start_layer_id: Starting layer index for this node
        end_layer_id: Ending layer index for this node (exclusive)
        status_port: Port for status/readiness check
        dtype: Data type for model weights
        s3_path: S3 path where model tensors are stored (e.g., 's3://bucket/path/to/models')
        aws_profile: AWS profile to use for S3 access
    
    Returns:
        Command string to execute
    """
    
    # Use provided port + local_rank or calculate from global base port + local_rank
    if status_port is None:
        final_port = DEFAULT_TENSOR_STORE_BASE_PORT + local_rank
    else:
        final_port = status_port + local_rank
    
    # Use S3 tensor store server if s3_path is provided
    if s3_path:
        cmd = (
            f"{PYTHON} {PROJECT_PATH}/TensorStore/s3_tensor_store_server.py "
            f"--model-name {model_name} "
            f"--s3-path {s3_path} "
            f"--tensor-parallel-size {tensor_parallel_size} "
            f"--local-rank {local_rank} "
            f"--start-layer-id {start_layer_id} "
            f"--end-layer-id {end_layer_id} "
            f"--status-port {final_port}"
        )
        if aws_profile:
            cmd += f" --aws-profile {aws_profile}"
        if dtype is not None:
            cmd += f" --dtype {dtype}"
    else:
        # Use original tensor store server for HuggingFace
        cmd = (
            f"{PYTHON} {PROJECT_PATH}/TensorStore/mt_tensor_store_server.py "
            f"--model-name {model_name} "
            f"--tensor-parallel-size {tensor_parallel_size} "
            f"--local-rank {local_rank} "
            f"--start-layer-id {start_layer_id} "
            f"--end-layer-id {end_layer_id} "
            f"--status-port {final_port} "
        )
        if dtype is not None:
            cmd += f" --dtype {dtype}"
    
    return cmd


def get_api_server_command(model_name: str,
                          pp_layer_partition: str,
                          parallel_strategy: List[int],
                          host: str,
                          node_rank_mapping: str,
                          port: Optional[int] = None,
                          dtype: Optional[str] = None,
                          max_model_len: Optional[int] = None,
                          gpu_memory_utilization: Optional[float] = None,
                          max_num_batched_tokens: Optional[int] = None,
                          max_num_seqs: Optional[int] = None) -> str:
    """
    Generate command to start API server based on launch_server_example.sh.
    
    Note: TensorStore servers must be running on the same device as the API server
    to share memory.
    
    Args:
        model_name: Name of the model (required)
        pp_layer_partition: Pipeline layer partition (required, comma-separated)
        parallel_strategy: Parallel strategy (required, list of integers)
        host: Host address for the API server
        port: API server port (optional, uses default if None)
        dtype: Data type for model weights
        max_model_len: Maximum model length
        node_rank_mapping: JSON string for node rank mapping (required)
        gpu_memory_utilization: GPU memory utilization ratio (optional)
        max_num_batched_tokens: Maximum number of batched tokens
        max_num_seqs: Maximum number of sequences
    
    Returns:
        Command string to execute
    """
    
    
    # Use provided port or default base port
    if port is None:
        port = DEFAULT_API_SERVER_BASE_PORT
    
    # Convert parallel_strategy list to space-separated string
    parallel_strategy_str = " ".join(map(str, parallel_strategy))
    
    cmd_parts = [
        f"{PYTHON} {PROJECT_PATH}/InferenceServer/api_server.py",
        f"--model={model_name}",
        f"--host={host}",
        f"--port={port}",
        f'--pp-layer-partition="{pp_layer_partition}"',
        f"--parallel-strategy {parallel_strategy_str}"
    ]
    
    if dtype is not None:
        cmd_parts.append(f"--dtype={dtype}")
    
    # Add optional arguments
    if max_model_len is not None:
        cmd_parts.append(f"--max_model_len={max_model_len}")
    
    # Escape quotes for SSH command execution
    escaped_mapping = node_rank_mapping.replace('"', '\\"')
    cmd_parts.append(f'--node-rank-mapping="{escaped_mapping}"')
    
    if gpu_memory_utilization is not None:
        cmd_parts.append(f"--gpu-memory-utilization={gpu_memory_utilization}")
    
    if max_num_batched_tokens is not None:
        cmd_parts.append(f"--max-num-batched-tokens={max_num_batched_tokens}")
    
    if max_num_seqs is not None:
        cmd_parts.append(f"--max-num-seqs={max_num_seqs}")
    
    return " ".join(cmd_parts)