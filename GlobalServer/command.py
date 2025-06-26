import os
from typing import List


def get_tensor_store_command(model_name: str, 
                            tensor_parallel_size: int,
                            local_rank: int, 
                            start_layer_id: int,
                            end_layer_id: int,
                            status_port: int,
                            dtype: str = "float16") -> str:
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
    
    Returns:
        Command string to execute
    """
    python_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cmd = (
        f"cd {python_path} && "
        f"python TensorStore/mt_tensor_store_server.py "
        f"--model-name {model_name} "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--local-rank {local_rank} "
        f"--start-layer-id {start_layer_id} "
        f"--end-layer-id {end_layer_id} "
        f"--status-port {status_port} "
        f"--dtype {dtype}"
    )
    
    return cmd


def get_api_server_command(model_name: str,
                          tensor_parallel_size: int,
                          pipeline_rank: int,
                          port: int,
                          tensor_store_addrs: List[str]) -> str:
    """
    Generate command to start API server.
    
    Args:
        model_name: Name of the model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_rank: Pipeline parallel rank
        port: API server port
        tensor_store_addrs: List of tensor store addresses (host:port)
    
    Returns:
        Command string to execute
    """
    python_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cmd = (
        f"cd {python_path} && "
        f"python InferenceServer/api_server.py "
        f"--model {model_name} "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--pipeline-rank {pipeline_rank} "
        f"--port {port} "
        f"--tensor-store-addrs {','.join(tensor_store_addrs)}"
    )
    
    return cmd