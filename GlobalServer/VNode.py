from typing import List, Dict, Tuple
import socket
import ray
import subprocess
import logging
import time
import os
from command import get_tensor_store_command, get_api_server_command

class Cluster:
    def __init__(self):
        if not ray.is_initialized():
            ray.init(address="auto")
        self.pipelines: List[Pipeline] = []
        
    def create_pipeline(self, 
                       node_layer_mapping: List[Tuple[str, int]], 
                       config: Dict):
        """
        Create a pipeline for distributed LLM inference.
        
        Args:
            node_layer_mapping: List of (node_ip, num_layers) tuples
            config: Configuration dictionary containing:
                - model_name (required): Name of the model
                - total_num_layers (required): Total number of layers in the model
                - pp_layer_partition (required): Pipeline layer partition string
                - parallel_strategy (required): List of parallel strategy integers
                - tensor_store_base_port (optional): Base port for tensor store servers
                - api_server_base_port (optional): Base port for API servers
                - dtype (optional): Data type for model weights
                - max_model_len (optional): Maximum model length
                - node_rank_mapping (optional): JSON string for node rank mapping
                - node_rank_mapping_path (optional): Path to node rank mapping JSON file
                - gpu_memory_utilization (optional): GPU memory utilization ratio
                - max_num_batched_tokens (optional): Maximum number of batched tokens
                - max_num_seqs (optional): Maximum number of sequences
        """
        # Validate required config parameters
        required_keys = ["model_name", "total_num_layers", "pp_layer_partition", "parallel_strategy"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"config must contain '{key}'")
            
        pipeline = Pipeline()
        pipeline.initialize_pipeline(node_layer_mapping, config)
        self.pipelines.append(pipeline)

class Pipeline:
    def __init__(self):
        self.vnodes: List[VNode] = []
        self.model_name: str = ""
        self.total_layers: int = 0

    def initialize_pipeline(self, 
                            node_layer_mapping: List[Tuple[str, int]], 
                            config: Dict):
        assert len(node_layer_mapping) > 0, "node_layer_mapping is empty"
        
        # Extract required config parameters
        model_name = config["model_name"]
        total_num_layers = config["total_num_layers"]
        
        # Check layer validity
        total_assigned_layers = sum(layers for _, layers in node_layer_mapping)
        assert total_assigned_layers == total_num_layers, (
            f"Total assigned layers ({total_assigned_layers}) does not match "
            f"model total layers ({total_num_layers})"
        )

        if not ray.is_initialized():
            ray.init(address="auto")
        
        self.model_name = model_name
        self.total_layers = total_num_layers
        self.config = config
        
        start_layer_idx = 0
        for pipeline_rank, (node_ip, layer_partition) in enumerate(node_layer_mapping):
            num_gpu = None
            while True:
                for node_info in ray.nodes():
                    ray_node_ip = node_info.get("NodeManagerAddress")
                    if ray_node_ip == node_ip:
                        num_gpu = int(node_info.get("Resources").get("GPU", 0))
                        break
                if num_gpu is not None and num_gpu > 0:
                    break
                logging.info(f"Waiting for node {node_ip} to be entered into Ray cluster...")
                time.sleep(1)

            vnode = VNode(
                node_ip=node_ip, 
                num_gpu=num_gpu, 
                pipeline_rank=pipeline_rank,
                layer_start_id=start_layer_idx, 
                layer_end_id=start_layer_idx + layer_partition,
                total_layers=self.total_layers
            )
            start_layer_idx += layer_partition
            self.vnodes.append(vnode)
        
        # Start tensor stores on all VNodes
        tensor_store_base_port = config.get("tensor_store_base_port")
        for i, vnode in enumerate(self.vnodes):
            # Each VNode gets unique ports to avoid conflicts
            if tensor_store_base_port is not None:
                tensor_store_port = tensor_store_base_port + i * 10  # Leave room for multiple GPUs
            else:
                tensor_store_port = None  # Will use global default in command.py
            vnode.start_tensor_store(tensor_store_port, config)
        
        # Start API server only on the first node (pipeline rank 0)
        if len(self.vnodes) > 0:
            first_vnode = self.vnodes[0]
            api_server_base_port = config.get("api_server_base_port")
            first_vnode.start_api_server(api_server_base_port, config)
        
        # Wait for all services to be ready
        self._wait_for_services()
    
    def _wait_for_services(self):
        """Wait for all tensor stores and API server to be ready."""
        tensor_store_statuses = [False] * len(self.vnodes)
        api_server_ready = False
        
        while not (all(tensor_store_statuses) and api_server_ready):
            # Check tensor store status for all nodes
            for i, vnode in enumerate(self.vnodes):
                if not tensor_store_statuses[i]:
                    tensor_store_statuses[i] = vnode.check_tensor_store_status()
            
            # Check API server status only for the first node
            if not api_server_ready and len(self.vnodes) > 0:
                api_server_ready = self.vnodes[0].check_api_server_status()
            
            ready_ts = sum(tensor_store_statuses)
            api_status = "Ready" if api_server_ready else "Not ready"
            logging.info(f"Waiting for services... Tensor stores: {ready_ts}/{len(self.vnodes)}, "
                        f"API server: {api_status}")
            
            if not (all(tensor_store_statuses) and api_server_ready):
                time.sleep(3)


class VNode:
    """
    Virtual Node abstraction for distributed LLM inference.
    
    Each VNode represents a physical node with GPUs. Within a VNode:
    - Multiple GPUs use Tensor Parallelism (TP)
    - Different VNodes use Pipeline Parallelism (PP)
    """
    def __init__(self, 
                 node_ip: str, 
                 num_gpu: int,
                 pipeline_rank: int,
                 layer_start_id: int,
                 layer_end_id: int,
                 total_layers: int):
        # Node configuration
        self.node_ip = node_ip
        self.num_gpu = num_gpu
        self.pipeline_rank = pipeline_rank
        
        # Layer assignment for Pipeline Parallelism
        self.layer_start_id = layer_start_id
        self.layer_end_id = layer_end_id
        self.total_layers = total_layers
        
        # Tensor store configuration
        self.tensor_store_port = None
        self.is_tensor_store_ready = False
        self.tensor_store_processes = []  # List of subprocess objects
        
        # API server configuration
        self.api_server_port = None
        self.is_api_server_ready = False
        self.api_server_process = None
        
        # Create log directory if it doesn't exist
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Parallelism settings
        self.tensor_parallel_size = num_gpu  # TP size = number of GPUs on this node
        self.is_first_stage = (layer_start_id == 0)
        self.is_last_stage = (layer_end_id >= total_layers)
        
        logging.info(f"VNode created: {self}")
    
    def __repr__(self):
        return (f"VNode(ip={self.node_ip}, gpus={self.num_gpu}, "
                f"rank={self.pipeline_rank}, layers=[{self.layer_start_id}, {self.layer_end_id}), "
                f"TP={self.tensor_parallel_size})")

    def start_tensor_store(self, tensor_store_port: int, config: Dict):
        """Start tensor store server on this VNode."""
        self.tensor_store_port = tensor_store_port
        self.is_tensor_store_ready = False
        
        model_name = config["model_name"]
        dtype = config.get("dtype", "float16")
        
        # Start tensor store processes for each GPU (for TP)
        for local_rank in range(self.num_gpu):
            command = get_tensor_store_command(
                model_name=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                local_rank=local_rank,
                start_layer_id=self.layer_start_id,
                end_layer_id=self.layer_end_id,
                status_port=tensor_store_port,  # Base port, local_rank will be added in command
                dtype=dtype
            )
            
            # Prepare log files
            log_filename = f"tensorstore_{self.node_ip}_{local_rank}.log"
            log_path = os.path.join(self.log_dir, log_filename)
            
            # Use SSH to start the process on the remote node with log redirection
            ssh_command = f"ssh {self.node_ip} '{command} > {log_path} 2>&1 &'"
            
            try:
                process = subprocess.Popen(
                    ssh_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.tensor_store_processes.append(process)
                logging.info(f"Started tensor store process (rank {local_rank}) on {self.node_ip}, log: {log_path}")
            except Exception as e:
                logging.error(f"Failed to start tensor store on {self.node_ip}: {e}")
                
        logging.info(f"Started {self.num_gpu} tensor store processes on {self.node_ip}")
    
    def start_api_server(self, api_server_port: int, config: Dict):
        """Start API server on this VNode (should only be called for pipeline rank 0)."""
        if self.pipeline_rank != 0:
            logging.warning(f"start_api_server called on non-first node (rank {self.pipeline_rank}). Skipping.")
            return
            
        self.api_server_port = api_server_port
        self.is_api_server_ready = False
        
        command = get_api_server_command(
            model_name=config["model_name"],
            pp_layer_partition=config["pp_layer_partition"],
            parallel_strategy=config["parallel_strategy"],
            host=config.get("host", "127.0.0.1"),
            port=api_server_port,
            dtype=config.get("dtype", "float16"),
            max_model_len=config.get("max_model_len"),
            node_rank_mapping=config.get("node_rank_mapping"),
            node_rank_mapping_path=config.get("node_rank_mapping_path"),
            gpu_memory_utilization=config.get("gpu_memory_utilization"),
            max_num_batched_tokens=config.get("max_num_batched_tokens"),
            max_num_seqs=config.get("max_num_seqs")
        )
        
        # Prepare log file
        log_filename = f"apiserver_{self.node_ip}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        # Use SSH to start the API server on the remote node with log redirection
        ssh_command = f"ssh {self.node_ip} '{command} > {log_path} 2>&1 &'"
        
        try:
            self.api_server_process = subprocess.Popen(
                ssh_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logging.info(f"Started API server on {self.node_ip}:{api_server_port} (pipeline rank 0), log: {log_path}")
        except Exception as e:
            logging.error(f"Failed to start API server on {self.node_ip}: {e}")

    def check_tensor_store_status(self, timeout: float = 2.0) -> bool:
        """Check if all tensor store servers on this node are ready."""
        all_ready = True
        
        for i in range(self.num_gpu):
            port = self.tensor_store_port + i
            try:
                with socket.create_connection((self.node_ip, port), timeout=timeout) as sock:
                    sock.settimeout(timeout)
                    resp = sock.recv(4)
                    if resp.strip() != b"1":
                        all_ready = False
                        break
            except (socket.timeout, ConnectionRefusedError, OSError):
                all_ready = False
                break
        
        self.is_tensor_store_ready = all_ready
        return all_ready

    def check_api_server_status(self, timeout: float = 2.0) -> bool:
        """Check if API server is ready."""
        # Only check if this is the first node and API server was started
        if self.pipeline_rank != 0 or self.api_server_port is None:
            return True  # Not applicable for non-first nodes
            
        try:
            # Simple HTTP health check
            import requests
            response = requests.get(
                f"http://{self.node_ip}:{self.api_server_port}/health",
                timeout=timeout
            )
            self.is_api_server_ready = (response.status_code == 200)
            return self.is_api_server_ready
        except Exception:
            self.is_api_server_ready = False
            return False
    
    def get_node_resources(self) -> Dict:
        """Get resource information for this node."""
        return {
            "node_ip": self.node_ip,
            "num_gpu": self.num_gpu,
            "pipeline_rank": self.pipeline_rank,
            "layers": f"[{self.layer_start_id}, {self.layer_end_id})",
            "tensor_parallel_size": self.tensor_parallel_size,
            "is_first_stage": self.is_first_stage,
            "is_last_stage": self.is_last_stage,
            "tensor_store_ready": self.is_tensor_store_ready,
            "api_server_ready": self.is_api_server_ready if self.pipeline_rank == 0 else "N/A (not first node)"
        }



def example_usage():
    """Example of how to use the VNode and Pipeline classes."""
    logging.basicConfig(level=logging.INFO)
    
    # Create a cluster
    cluster = Cluster()
    
    # Define node-layer mapping for pipeline parallelism
    # Each tuple is (node_ip, number_of_layers)
    node_layer_mapping = [
        ("", 32)
    ]
    
    # Create pipeline for Llama-3.1-8B (32 layers total)
    config = {
        "model_name": "meta-llama/Llama-3.1-8B",
        "total_num_layers": 32,
        "pp_layer_partition": "32",
        "parallel_strategy": [1],
        "node_rank_mapping_path": "../node_rank_mapping.json",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.25,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 16
        # tensor_store_base_port and api_server_base_port not specified - will use global defaults
    }
    
    cluster.create_pipeline(
        node_layer_mapping=node_layer_mapping,
        config=config
    )
    
    # The pipeline will:
    # 1. Detect GPU count on each node automatically
    # 2. Apply Tensor Parallelism within each node (across GPUs)
    # 3. Apply Pipeline Parallelism across nodes
    # 4. Start tensor store servers for model weight management on all nodes
    # 5. Start API server for inference on the first node only (pipeline rank 0)
    
    logging.info("Pipeline created successfully!")
    
    # Print VNode information
    for vnode in cluster.pipelines[0].vnodes:
        logging.info(f"VNode resources: {vnode.get_node_resources()}")
    
    # Note: API server runs only on the first node
    logging.info(f"\nAPI server is running on node {cluster.pipelines[0].vnodes[0].node_ip} (pipeline rank 0)")


if __name__ == "__main__":
    example_usage()