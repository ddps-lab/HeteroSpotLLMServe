from typing import List, Dict, Tuple
import socket
import ray
import subprocess
import logging
import time
from .command import get_tensor_store_command, get_api_server_command

class Cluster:
    def __init__(self):
        if not ray.is_initialized():
            ray.init(address="auto")
        self.pipelines: List[Pipeline] = []
        
    def create_pipeline(self, 
                       node_layer_mapping: List[Tuple[str, int]], 
                       model_name: str,
                       total_layers: int,
                       tensor_store_base_port: int = 10001,
                       api_server_base_port: int = 8001):
        """
        Create a pipeline for distributed LLM inference.
        
        Args:
            node_layer_mapping: List of (node_ip, num_layers) tuples
            model_name: Name of the model (e.g., 'meta-llama/Llama-2-7b-hf')
            total_layers: Total number of layers in the model (required)
            tensor_store_base_port: Base port for tensor store servers
            api_server_base_port: Base port for API servers
        """
        pipeline = Pipeline()
        pipeline.initialize_pipeline(
            node_layer_mapping, 
            model_name,
            total_layers,
            tensor_store_base_port,
            api_server_base_port
        )
        self.pipelines.append(pipeline)

class Pipeline:
    def __init__(self):
        self.vnodes: List[VNode] = []
        self.model_name: str = ""
        self.total_layers: int = 0

    def initialize_pipeline(self, 
                            node_layer_mapping: List[Tuple[str, int]], 
                            model_name: str,
                            total_layers: int,
                            tensor_store_base_port: int,
                            api_server_base_port: int):
        assert len(node_layer_mapping) > 0, "node_layer_mapping is empty"
        
        # Check layer validity
        total_assigned_layers = sum(layers for _, layers in node_layer_mapping)
        assert total_assigned_layers == total_layers, (
            f"Total assigned layers ({total_assigned_layers}) does not match "
            f"model total layers ({total_layers})"
        )

        if not ray.is_initialized():
            ray.init(address="auto")
        
        self.model_name = model_name
        self.total_layers = total_layers
        
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
        
        # Start tensor stores and API servers on each VNode
        for i, vnode in enumerate(self.vnodes):
            # Each VNode gets unique ports to avoid conflicts
            tensor_store_port = tensor_store_base_port + i * 10  # Leave room for multiple GPUs
            api_server_port = api_server_base_port + i
            
            vnode.start_tensor_store(tensor_store_port, model_name)
            vnode.start_api_server(api_server_port, model_name)
        
        # Wait for all services to be ready
        self._wait_for_services()
    
    def _wait_for_services(self):
        """Wait for all tensor stores and API servers to be ready."""
        tensor_store_statuses = [False] * len(self.vnodes)
        api_server_statuses = [False] * len(self.vnodes)
        
        while not (all(tensor_store_statuses) and all(api_server_statuses)):
            for i, vnode in enumerate(self.vnodes):
                if not tensor_store_statuses[i]:
                    tensor_store_statuses[i] = vnode.check_tensor_store_status()
                if not api_server_statuses[i]:
                    api_server_statuses[i] = vnode.check_api_server_status()
            
            ready_ts = sum(tensor_store_statuses)
            ready_api = sum(api_server_statuses)
            logging.info(f"Waiting for services... Tensor stores: {ready_ts}/{len(self.vnodes)}, "
                        f"API servers: {ready_api}/{len(self.vnodes)}")
            
            if not (all(tensor_store_statuses) and all(api_server_statuses)):
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
        self.tensor_store_process = None
        self.tensor_store_actors = []  # Ray actors for tensor store
        
        # API server configuration
        self.api_server_port = None
        self.is_api_server_ready = False
        self.api_server_process = None
        
        # Parallelism settings
        self.tensor_parallel_size = num_gpu  # TP size = number of GPUs on this node
        self.is_first_stage = (layer_start_id == 0)
        self.is_last_stage = (layer_end_id >= total_layers)
        
        logging.info(f"VNode created: {self}")
    
    def __repr__(self):
        return (f"VNode(ip={self.node_ip}, gpus={self.num_gpu}, "
                f"rank={self.pipeline_rank}, layers=[{self.layer_start_id}, {self.layer_end_id}), "
                f"TP={self.tensor_parallel_size})")

    def start_tensor_store(self, tensor_store_port: int, model_name: str):
        """Start tensor store server on this VNode."""
        self.tensor_store_port = tensor_store_port
        self.is_tensor_store_ready = False
        
        # Start tensor store processes for each GPU (for TP)
        for local_rank in range(self.num_gpu):
            command = get_tensor_store_command(
                model_name=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                local_rank=local_rank,
                start_layer_id=self.layer_start_id,
                end_layer_id=self.layer_end_id,
                status_port=tensor_store_port + local_rank  # Each GPU gets its own port
            )
            
            # Use Ray to start the process on the specific node
            @ray.remote(num_gpus=1, resources={f"node:{self.node_ip}": 1})
            def start_tensor_store_remote(cmd):
                import subprocess
                process = subprocess.Popen(cmd, shell=True)
                return process.pid
            
            actor = start_tensor_store_remote.remote(command)
            self.tensor_store_actors.append(actor)
            
        logging.info(f"Started {self.num_gpu} tensor store processes on {self.node_ip}")
    
    def start_api_server(self, api_server_port: int, model_name: str):
        """Start API server on this VNode."""
        self.api_server_port = api_server_port
        self.is_api_server_ready = False
        
        # Build tensor store addresses for this node
        tensor_store_addrs = []
        for i in range(self.num_gpu):
            tensor_store_addrs.append(f"{self.node_ip}:{self.tensor_store_port + i}")
        
        command = get_api_server_command(
            model_name=model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_rank=self.pipeline_rank,
            port=api_server_port,
            tensor_store_addrs=tensor_store_addrs
        )
        
        # Use Ray to start the API server on the specific node
        @ray.remote(num_gpus=self.num_gpu, resources={f"node:{self.node_ip}": 1})
        def start_api_server_remote(cmd):
            import subprocess
            process = subprocess.Popen(cmd, shell=True)
            return process.pid
        
        self.api_server_process = start_api_server_remote.remote(command)
        logging.info(f"Started API server on {self.node_ip}:{api_server_port}")

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
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                all_ready = False
                break
        
        self.is_tensor_store_ready = all_ready
        return all_ready

    def check_api_server_status(self, timeout: float = 2.0) -> bool:
        """Check if API server is ready."""
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
            "api_server_ready": self.is_api_server_ready
        }



def example_usage():
    """Example of how to use the VNode and Pipeline classes."""
    logging.basicConfig(level=logging.INFO)
    
    # Create a cluster
    cluster = Cluster()
    
    # Define node-layer mapping for pipeline parallelism
    # Each tuple is (node_ip, number_of_layers)
    node_layer_mapping = [
        ("192.168.1.10", 8),   # First node handles layers 0-7
        ("192.168.1.11", 8),   # Second node handles layers 8-15
        ("192.168.1.12", 8),   # Third node handles layers 16-23
        ("192.168.1.13", 8),   # Fourth node handles layers 24-31
    ]
    
    # Create pipeline for Llama-2-7b (32 layers total)
    cluster.create_pipeline(
        node_layer_mapping=node_layer_mapping,
        model_name="meta-llama/Llama-2-7b-hf",
        total_layers=32,  # Required parameter
        tensor_store_base_port=10001,
        api_server_base_port=8001
    )
    
    # The pipeline will:
    # 1. Detect GPU count on each node automatically
    # 2. Apply Tensor Parallelism within each node (across GPUs)
    # 3. Apply Pipeline Parallelism across nodes
    # 4. Start tensor store servers for model weight management
    # 5. Start API servers for inference
    
    logging.info("Pipeline created successfully!")
    
    # Print VNode information
    for vnode in cluster.pipelines[0].vnodes:
        logging.info(f"VNode resources: {vnode.get_node_resources()}")


if __name__ == "__main__":
    example_usage()