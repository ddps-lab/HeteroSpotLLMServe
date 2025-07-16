from typing import List, Dict, Tuple
import socket
import ray
import subprocess
import logging
import time
import os
from command import get_tensor_store_command, get_api_server_command, get_ray_start_worker_command, get_ray_stop_command, DEFAULT_TENSOR_STORE_BASE_PORT, DEFAULT_API_SERVER_BASE_PORT
import json
import sys
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for protocols import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocols import TensorStoreRequest, TensorStoreResponse

# Configure logging for Cluster and VNode
cluster_logger = logging.getLogger(__name__)
cluster_logger.setLevel(logging.INFO)

# Get the GlobalServer directory path (where this file is located)
GLOBAL_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

# Create logs directory if it doesn't exist
LOG_BASE_DIR = os.path.join(GLOBAL_SERVER_DIR, "logs")
os.makedirs(LOG_BASE_DIR, exist_ok=True)

# Create file handler for cluster logs
cluster_log_file = os.path.join(LOG_BASE_DIR, "Cluster.log")
file_handler = logging.FileHandler(cluster_log_file)
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
cluster_logger.addHandler(file_handler)

# Prevent propagation to root logger (to avoid console output)
cluster_logger.propagate = False

class Cluster:
    def __init__(self):
        self.pipelines: List[Pipeline] = []
        self.ideal_throughput: float = 0.0
        
    def create_pipeline(self, 
                       node_layer_mapping: List[Tuple[str, int]], 
                       config: Dict,
                       ideal_throughput: float):
        """
        Create a pipeline for distributed LLM inference.
        
        Args:
            node_layer_mapping: List of (node_ip, num_layers) tuples
            config: Configuration dictionary containing:
                - model_name (required): Name of the model
                - total_num_layers (required): Total number of layers in the model
                - pp_layer_partition (required): Pipeline layer partition string
                - parallel_strategy (required): List of parallel strategy integers
                - model_source (optional): Source of model weights - 'huggingface' or 's3' (default: 'huggingface')
                - s3_path (optional): S3 path where model tensors are stored (required if model_source='s3')
                - aws_profile (optional): AWS profile to use for S3 access
                - tensor_store_base_port (optional): Base port for tensor store servers
                - api_server_base_port (optional): Base port for API servers
                - dtype (optional): Data type for model weights
                - max_model_len (optional): Maximum model length
                - node_rank_mapping (optional): JSON string for node rank mapping
                - node_rank_mapping_path (optional): Path to node rank mapping JSON file
                - gpu_memory_utilization (optional): GPU memory utilization ratio
                - max_num_batched_tokens (optional): Maximum number of batched tokens
                - max_num_seqs (optional): Maximum number of sequences
            ideal_throughput: Expected throughput for this pipeline (requests/sec)
        """
        # Validate required config parameters
        required_keys = ["model_name", "total_num_layers", "pp_layer_partition", "parallel_strategy"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"config must contain '{key}'")
            
        pipeline = Pipeline()
        pipeline.initialize_pipeline(node_layer_mapping, config, ideal_throughput)
        self.pipelines.append(pipeline)
        self.ideal_throughput += ideal_throughput

    def stop_all_pipelines(self):
        """Stop all pipelines in the cluster."""
        cluster_logger.info(f"Stopping {len(self.pipelines)} pipelines...")
        
        # Handle case when no pipelines exist
        if not self.pipelines:
            cluster_logger.info("No pipelines to stop.")
            return
        
        # Stop all pipelines in parallel
        with ThreadPoolExecutor(max_workers=len(self.pipelines)) as executor:
            futures = []
            
            for i, pipeline in enumerate(self.pipelines):
                future = executor.submit(pipeline.stop_pipeline)
                futures.append((i, future))
            
            # Wait for all to complete
            for pipeline_idx, future in futures:
                try:
                    future.result(timeout=300)  # 5 minutes timeout per pipeline
                    cluster_logger.info(f"Pipeline {pipeline_idx} stopped successfully")
                except Exception as e:
                    cluster_logger.error(f"Failed to stop pipeline {pipeline_idx}: {e}")
        
        cluster_logger.info("All pipelines stopped")

    def switch_node(self, old_node_ip: str, new_node_ip: str):
        """
        Switch a node in the cluster by replacing old_node_ip with new_node_ip.
        
        Args:
            old_node_ip: IP address of the node to be replaced
            new_node_ip: IP address of the new node
        """
        cluster_logger.info(f"Starting node switch: {old_node_ip} -> {new_node_ip}")
        
        # Find the pipeline containing the target VNode
        target_pipeline = None
        target_vnode = None
        
        for pipeline in self.pipelines:
            for vnode in pipeline.vnodes:
                if vnode.node_ip == old_node_ip:
                    target_vnode = vnode
                    target_pipeline = pipeline
                    break
            if target_vnode:
                break
        
        if not target_pipeline or not target_vnode:
            raise ValueError(f"No VNode found with IP {old_node_ip}")
        
        cluster_logger.info(f"Found VNode in pipeline: {target_vnode}")
        
        # Delegate to the pipeline's switch_node method
        target_pipeline.switch_node(old_node_ip, new_node_ip)

        cluster_logger.info(f"Node switch completed successfully: {old_node_ip} -> {new_node_ip}")

class Pipeline:
    def __init__(self):
        self.vnodes: List[VNode] = []
        self.model_name: str = ""
        self.total_layers: int = 0
        self.node_rank_mapping: Dict[str, List[int]] = {}
        self.ideal_throughput: float = 0.0
        self.is_ready: bool = False  # Pipeline readiness status
        
        # Ray cluster management for this pipeline
        self.ray_port = None  # Ray port for this pipeline's cluster
        self.ray_head_ip = None  # IP of the Ray head node (global server node)

        self.api_server_host: str = None
        self.api_server_port: int = None

    def initialize_pipeline(self, 
                            node_layer_mapping: List[Tuple[str, int]], 
                            config: Dict,
                            ideal_throughput: float):
        assert len(node_layer_mapping) > 0, "node_layer_mapping is empty"

        cluster_logger.info(f"Initializing pipeline with {node_layer_mapping}")
        
        # Extract required config parameters
        model_name = config["model_name"]
        total_num_layers = config["total_num_layers"]
        
        # Check layer validity
        total_assigned_layers = sum(layers for _, layers in node_layer_mapping)
        assert total_assigned_layers == total_num_layers, (
            f"Total assigned layers ({total_assigned_layers}) does not match "
            f"model total layers ({total_num_layers})"
        )
        
        self.model_name = model_name
        self.total_layers = total_num_layers
        self.config = config
        self.ideal_throughput = ideal_throughput
        
        # First, create VNode objects with placeholder GPU count
        start_layer_idx = 0
        for pipeline_rank, (node_ip, layer_partition) in enumerate(node_layer_mapping):
            vnode = VNode(
                node_ip=node_ip, 
                num_gpu=None,  # Placeholder, will be updated after Ray cluster is ready
                pipeline_rank=pipeline_rank,
                layer_start_id=start_layer_idx, 
                layer_end_id=start_layer_idx + layer_partition,
                total_layers=self.total_layers
            )
            start_layer_idx += layer_partition
            self.vnodes.append(vnode)
        
        # Now start Ray cluster with all vnodes
        ray_port = self.get_alternate_ray_port()
        self.start_ray_cluster(ray_port)
        
        # Update VNode GPU counts from Ray cluster information
        for vnode in self.vnodes:
            num_gpu = None
            while True:
                for node_info in ray.nodes():
                    ray_node_ip = node_info.get("NodeManagerAddress")
                    if ray_node_ip == vnode.node_ip:
                        num_gpu = int(node_info.get("Resources").get("GPU", 0))
                        break
                if num_gpu is not None and num_gpu > 0:
                    break
                cluster_logger.info(f"Waiting for node {vnode.node_ip} to be entered into Ray cluster...")
                time.sleep(1)
            
            # Update the vnode's GPU count
            vnode.num_gpu = num_gpu
            cluster_logger.info(f"Updated VNode {vnode.node_ip} with {num_gpu} GPUs")
        
        # Start tensor stores on all VNodes
        tensor_store_base_port = config.get("tensor_store_base_port")
        parallel_strategy = config["parallel_strategy"]
        for i, vnode in enumerate(self.vnodes):
            # Each VNode gets unique ports to avoid conflicts
            if tensor_store_base_port is not None:
                tensor_store_port = tensor_store_base_port
            else:
                tensor_store_port = DEFAULT_TENSOR_STORE_BASE_PORT # use global default in command.py
            cluster_logger.info(f"Starting tensor store on {vnode.node_ip} at port {tensor_store_port + i}")
            vnode.start_tensor_store(tensor_store_port, config, len(parallel_strategy))
        
        # Generate node_rank_mapping based on vnodes
        self._generate_node_rank_mapping()
        
        # Start API server only on the first node (pipeline rank 0)
        first_vnode = self.vnodes[0]
        api_server_base_port = config.get("api_server_base_port", DEFAULT_API_SERVER_BASE_PORT)
        ray_address = f"{self.ray_head_ip}:{self.ray_port}"
        first_vnode.start_api_server(api_server_base_port, config, self.node_rank_mapping, ray_address)
        self.api_server_host = first_vnode.node_ip
        self.api_server_port = first_vnode.api_server_port
        
        # Wait for all services to be ready
        self._wait_for_services()

        self.api_server_host = first_vnode.node_ip
        self.api_server_port = first_vnode.api_server_port
        
        # Mark pipeline as ready
        self.is_ready = True
        cluster_logger.info(f"Pipeline initialized and ready")
    
    def _generate_node_rank_mapping(self):
        """Generate node_rank_mapping based on current vnodes."""
        self.node_rank_mapping = {}
        current_rank = 0
        
        for vnode in self.vnodes:
            ranks = list(range(current_rank, current_rank + vnode.num_gpu))
            self.node_rank_mapping[vnode.node_ip] = ranks
            current_rank += vnode.num_gpu
        
        cluster_logger.info(f"Generated node_rank_mapping: {self.node_rank_mapping}")
    
    def get_alternate_ray_port(self):
        """Get the alternate Ray port (6379 or 6380) for this pipeline."""
        # If current port is 6379, return 6380, and vice versa
        if self.ray_port == 6379:
            return 6380
        else:
            return 6379
    
    def start_ray_cluster(self, ray_port: int):
        """Start Ray cluster and ensure all vnodes are connected.
        
        Args:
            ray_port: Port for the Ray cluster (should be 6379 or 6380)
        """
        if ray_port not in [6379, 6380]:
            raise ValueError(f"Ray port must be 6379 or 6380, got {ray_port}")
            
        self.ray_port = ray_port
        self.ray_head_ip = socket.gethostbyname(socket.gethostname())
        
        cluster_logger.info(f"Starting Ray cluster on port {ray_port}")
        
        # Initialize or connect to Ray cluster
        ray_address = f"{self.ray_head_ip}:{ray_port}"
        try:
            if ray.is_initialized():
                ray.shutdown()#_exiting_interpreter=True)
            ray.init(address=ray_address, ignore_reinit_error=True)
            cluster_logger.info(f"Connected to Ray cluster on {ray_address}")
        except Exception as e:
            cluster_logger.error(f"Failed to connect to Ray cluster: {e}")
            raise
        
        # Get currently connected nodes
        ray_nodes = ray.nodes()
        connected_ips = {node.get("NodeManagerAddress") for node in ray_nodes if node.get("Alive")}
        cluster_logger.info(f"Currently connected nodes: {connected_ips}")
        
        # Find vnodes that need to be connected
        vnode_ips = {vnode.node_ip for vnode in self.vnodes}
        nodes_to_connect = vnode_ips - connected_ips
        
        if nodes_to_connect:
            cluster_logger.info(f"Need to connect nodes: {nodes_to_connect}")
            ssh_options = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
            
            for node_ip in nodes_to_connect:
                # Join the Ray cluster
                ray_command = get_ray_start_worker_command(f"{self.ray_head_ip}:{self.ray_port}")
                join_cmd = f"ssh {ssh_options} {node_ip} '{ray_command}'"
                cluster_logger.info(f"Connecting {node_ip} to Ray cluster with command: {join_cmd}")
                
                result = subprocess.run(join_cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    cluster_logger.error(f"Failed to connect {node_ip}: {result.stderr}")
                    raise RuntimeError(f"Failed to connect {node_ip} to Ray cluster")
            
            # Wait for all nodes to be connected with retry logic
            max_attempts = 30  # 30 attempts with 1 second interval = 30 seconds timeout
            attempt = 0
            
            while attempt < max_attempts:
                ray_nodes = ray.nodes()
                connected_ips = {node.get("NodeManagerAddress") for node in ray_nodes if node.get("Alive")}
                
                if vnode_ips.issubset(connected_ips):
                    cluster_logger.info(f"All vnodes successfully connected to Ray cluster")
                    break
                
                missing = vnode_ips - connected_ips
                attempt += 1
                cluster_logger.info(f"Waiting for nodes to connect (attempt {attempt}/{max_attempts}). Missing: {missing}")
                time.sleep(1)
            
            # Final check
            if not vnode_ips.issubset(connected_ips):
                missing = vnode_ips - connected_ips
                cluster_logger.error(f"Failed to connect all nodes after {max_attempts} attempts. Missing: {missing}")
                raise RuntimeError(f"Failed to connect nodes: {missing}")
        
        cluster_logger.info(f"All vnodes connected to Ray cluster on port {ray_port}")
    
    def _wait_for_services(self):
        """Wait for all tensor stores and API server to be ready."""
        tensor_store_statuses = [False] * len(self.vnodes)
        api_server_ready = False
        dots = 0
        
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
            
            # Create dynamic dots animation
            dots = (dots + 1) % 4
            dots_str = "." * dots + " " * (3 - dots)
            
            # Log status to file only
            status_msg = f"Waiting for services{dots_str} Tensor stores - {ready_ts}/{len(self.vnodes)}, API server {(self.api_server_host)}:{self.api_server_port} - {api_status}"
            cluster_logger.info(status_msg)
            
            if not (all(tensor_store_statuses) and api_server_ready):
                time.sleep(1)
        
        # Print final status only once to console
        cluster_logger.info(f"✓ All services ready! Tensor stores: {len(self.vnodes)}/{len(self.vnodes)}, API server: Ready")
        print(f"✓ All services ready! Tensor stores: {len(self.vnodes)}/{len(self.vnodes)}, API server: Ready")
    

    def stop_pipeline(self):
        """Stop all services in the pipeline."""
        cluster_logger.info(f"Stopping pipeline with {len(self.vnodes)} nodes...")
        
        # Stop all VNodes in parallel
        with ThreadPoolExecutor(max_workers=len(self.vnodes)) as executor:
            futures = []
            
            # Stop tensor stores on all nodes
            for vnode in self.vnodes:
                future = executor.submit(vnode.stop_tensor_store)
                futures.append(("tensor_store", vnode.node_ip, future))
            
            # Stop API server on first node only
            if self.vnodes:
                future = executor.submit(self.vnodes[0].stop_api_server)
                futures.append(("api_server", self.vnodes[0].node_ip, future))
            
            # Wait for all to complete
            for service_type, node_ip, future in futures:
                try:
                    future.result(timeout=120)  # 2 minutes timeout per service
                    cluster_logger.info(f"{service_type} stopped successfully on {node_ip}")
                except Exception as e:
                    cluster_logger.error(f"Failed to stop {service_type} on {node_ip}: {e}")
        
        cluster_logger.info("Pipeline shutdown completed")

    def switch_node(self, old_node_ip: str, new_node_ip: str):
        """
        Switch a node in this pipeline by replacing old_node_ip with new_node_ip.
        
        Args:
            old_node_ip: IP address of the node to be replaced
            new_node_ip: IP address of the new node
        """
        cluster_logger.info(f"Switching node in pipeline: {old_node_ip} -> {new_node_ip}")
        switch_e2e_start = time.time()
        
        # Find the target VNode
        target_vnode = None
        target_index = None
        
        for i, vnode in enumerate(self.vnodes):
            if vnode.node_ip == old_node_ip:
                target_vnode = vnode
                target_index = i
                break
        
        if not target_vnode:
            raise ValueError(f"No VNode found with IP {old_node_ip} in this pipeline")
        
        # 1. 기존 Ray port 저장
        old_ray_port = self.ray_port
        new_ray_port = self.get_alternate_ray_port()
        cluster_logger.info(f"Switching from Ray port {old_ray_port} to {new_ray_port}")
        
        # 2. 새로운 노드를 관리할 VNode 객체를 생성한다 (placeholder GPU count)
        new_vnode = VNode(
            node_ip=new_node_ip,
            num_gpu=None,  # Placeholder, will be updated after Ray cluster is ready
            pipeline_rank=target_vnode.pipeline_rank,
            layer_start_id=target_vnode.layer_start_id,
            layer_end_id=target_vnode.layer_end_id,
            total_layers=target_vnode.total_layers
        )
        
        # 기존의 api server 정보를 저장한다.
        old_api_server_host = self.api_server_host
        old_api_server_port = self.api_server_port
        old_first_vnode = self.vnodes[0]
        
        # 3. Pipeline 객체에 새로운 노드를 참여시킨다.
        self.vnodes[target_index] = new_vnode
        
        # 4. 새로운 Ray cluster를 시작한다.
        cluster_logger.info(f"Starting new Ray cluster on port {new_ray_port}")
        self.start_ray_cluster(new_ray_port)
        
        # 5. Get GPU count for new node from Ray cluster
        new_num_gpu = None
        while True:
            for node_info in ray.nodes():
                ray_node_ip = node_info.get("NodeManagerAddress")
                if ray_node_ip == new_node_ip:
                    new_num_gpu = int(node_info.get("Resources").get("GPU", 0))
                    break
            if new_num_gpu is not None and new_num_gpu > 0:
                break
            cluster_logger.info(f"Waiting for new node {new_node_ip} to be entered into Ray cluster...")
            time.sleep(1)
        
        # Update the new vnode's GPU count
        new_vnode.num_gpu = new_num_gpu
        cluster_logger.info(f"Updated new VNode {new_node_ip} with {new_num_gpu} GPUs")
        
        # 6. 새로운 노드에서 Tensor store 를 시작한다.
        tensor_store_port = target_vnode.tensor_store_port
        parallel_strategy = self.config["parallel_strategy"]
        new_vnode.start_tensor_store(tensor_store_port, self.config, len(parallel_strategy))
        
        # Tensor Store 가 준비될 때 까지 기다린다.
        cluster_logger.info(f"Checking tensor store status on new node {new_node_ip}")
        status_check_time = 0
        while not new_vnode.check_tensor_store_status():
            status_check_time += 1
            cluster_logger.info(f"Waiting for tensor store to be ready on new node {new_node_ip} ({status_check_time})...")
            time.sleep(2)
        
        cluster_logger.info(f"Tensor store ready on new node {new_node_ip}")
        
        # Node Rank Mapping Dictionary 를 업데이트 한다.
        self._generate_node_rank_mapping()

        # 4. 새로운 API server 동작
        new_first_vnode = self.vnodes[0]
        new_api_server_host = new_first_vnode.node_ip
        new_api_server_port = old_api_server_port

        # 만약 old_api_server_host 와 new_api_server_host 가 동일하다면
        # head 노드가 동일하다는 의미이다. 이 경우 새로운 API server 포트를 사용해야 한다.
        if old_api_server_host == new_api_server_host:
            # Increment port to avoid conflict
            new_api_server_port += 1

        # 새로운 api server 시작
        new_ray_address = f"{self.ray_head_ip}:{new_ray_port}"
        new_first_vnode.start_api_server(new_api_server_port, self.config, self.node_rank_mapping, new_ray_address)
        
        # 새로운 api server 가 준비될 때 까지 기다린다.
        cluster_logger.info(f"Checking API server status on node {new_first_vnode.node_ip}:{new_api_server_port}")
        status_check_time = 0
        while not new_first_vnode.check_api_server_status():
            status_check_time += 1
            cluster_logger.info(f"Waiting for API server {new_api_server_host}:{new_api_server_port} to be ready ({status_check_time})...")
            time.sleep(2)

        cluster_logger.info(f"API server ready on node {new_first_vnode.node_ip}")

        # 5. Pipeline 전환
        start_downtime = time.time()
        self.is_ready = False
        # 이제 기존의 api server 를 종료해야 한다.
        cluster_logger.info(f"Stopping old API server on {old_first_vnode.node_ip}:{old_api_server_port}")
        old_first_vnode.stop_api_server(old_api_server_port)
        self.api_server_host = new_api_server_host
        self.api_server_port = new_api_server_port
        self.is_ready = True
        end_downtime = time.time()
        switch_e2e_end = time.time()
        switch_e2e_latency = switch_e2e_end - switch_e2e_start
        downtime_duration = end_downtime - start_downtime
        cluster_logger.info(f"Node Switch Completed. End-to-End Latency: {switch_e2e_latency:.2f} / Downtime: {downtime_duration:.2f} seconds")

        # Clean up Ray workers on the target node being switched
        cluster_logger.info(f"Cleaning up Ray workers on target node {target_vnode.node_ip}")
        ssh_options = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=3 -o LogLevel=ERROR"
        cleanup_cmd = get_ray_stop_command()
        ssh_cmd = f"ssh {ssh_options} {target_vnode.node_ip} '{cleanup_cmd}'"
        
        try:
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                cluster_logger.info(f"Successfully cleaned up Ray workers on {target_vnode.node_ip}")
            else:
                # Filter out harmless SSH messages
                stderr_filtered = result.stderr.strip()
                if stderr_filtered and not stderr_filtered.startswith("Warning: Permanently added"):
                    cluster_logger.warning(f"Ray worker cleanup command failed on {target_vnode.node_ip}: {stderr_filtered}")
                else:
                    cluster_logger.info(f"Ray worker cleanup completed on {target_vnode.node_ip} (with SSH host key warning)")
        except subprocess.TimeoutExpired:
            cluster_logger.warning(f"Ray worker cleanup timed out on {target_vnode.node_ip} (node may be unreachable)")
        except Exception as e:
            cluster_logger.warning(f"Error cleaning up Ray workers on {target_vnode.node_ip}: {e}")
        
        # Stop tensor store on old node (target_vnode == old_vnode)
        cluster_logger.info(f"Stopping tensor store on old node {old_node_ip}")
        target_vnode.stop_tensor_store()
        
        cluster_logger.info(f"Node switch completed in pipeline: {old_node_ip} -> {new_node_ip}")


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
        # Use absolute path to GlobalServer/logs directory
        self.log_dir = LOG_BASE_DIR
        self.remote_log_dir = os.path.join(LOG_BASE_DIR, "remote")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.remote_log_dir, exist_ok=True)
        
        # Parallelism settings
        self.is_first_stage = (layer_start_id == 0)
        self.is_last_stage = (layer_end_id >= total_layers)
        
        cluster_logger.info(f"VNode created: {self}")
    
    def __repr__(self):
        return (f"VNode(ip={self.node_ip}, gpus={self.num_gpu}, "
                f"rank={self.pipeline_rank}, layers=[{self.layer_start_id}, {self.layer_end_id}), "
                f"TP={self.num_gpu})")

    def start_tensor_store(self, tensor_store_port: int, config: Dict, pipeline_parallel_size: int):
        """Start tensor store server on this VNode."""
        self.tensor_store_port = tensor_store_port
        self.is_tensor_store_ready = False
        
        model_name = config["model_name"]
        dtype = config.get("dtype")
        
        # Check if using S3 or HuggingFace
        model_source = config.get("model_source", "huggingface")  # Default to HuggingFace
        s3_path = config.get("s3_path") if model_source == "s3" else None
        aws_profile = config.get("aws_profile") if model_source == "s3" else None
        
        if model_source == "s3" and not s3_path:
            raise ValueError("s3_path must be provided when model_source is 's3'")

        block_size = config.get("block_size", 16)
        gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        swap_space = config.get("swap_space", 4.0)  # Default to 4GB swap space
        cache_dtype = config.get("cache_dtype", "auto")  # Default to auto-detect dtype
        max_model_len = config.get("max_model_len", 4096)
        
        # Start tensor store processes for each GPU (for TP)
        for local_rank in range(self.num_gpu):
            try:
                command = get_tensor_store_command(
                    model_name=model_name,
                    tensor_parallel_size=self.num_gpu,
                    local_rank=local_rank,
                    pipeline_parallel_size=pipeline_parallel_size,
                    pipeline_parallel_rank=self.pipeline_rank,
                    start_layer_id=self.layer_start_id,
                    end_layer_id=self.layer_end_id,
                    status_port=tensor_store_port,  # Base port, local_rank will be added in command
                    dtype=dtype,
                    s3_path=s3_path,
                    aws_profile=aws_profile,
                    block_size=block_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                    cache_dtype=cache_dtype,
                    max_model_len=max_model_len
                )
                
                # Prepare log files - save directly to local remote logs directory
                log_file_index = 0
                log_filename = f"tensorstore_{self.node_ip}_{local_rank}_{log_file_index}.log"
                while os.path.exists(os.path.join(self.remote_log_dir, log_filename)):
                    log_file_index += 1
                    log_filename = f"tensorstore_{self.node_ip}_{local_rank}_{log_file_index}.log"
                local_log_path = os.path.join(self.remote_log_dir, log_filename)
                
                # Use SSH to start the process and stream logs directly to local file
                # Add SSH options to avoid host key verification prompts
                ssh_options = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
                # Run command on remote and redirect output to local file
                ssh_command = f"ssh {ssh_options} {self.node_ip} '{command}' > {local_log_path} 2>&1 &"
                
                # Debug: print the command
                cluster_logger.info(f"Executing SSH command: {ssh_command}")
            except Exception as e:
                raise ValueError(f"Failed to prepare tensor store command for {self.node_ip}: {e}")
            
            
            try:
                process = subprocess.Popen(
                    ssh_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.tensor_store_processes.append(process)
                cluster_logger.info(f"Started tensor store process (rank {local_rank}) on {self.node_ip}, log: {local_log_path}")
                
                # Wait a bit and check if process started successfully
                time.sleep(0.5)
                _, stderr = process.communicate(timeout=0.1)
                if stderr:
                    cluster_logger.error(f"SSH command stderr: {stderr.decode()}")
            except subprocess.TimeoutExpired:
                # This is expected - process is still running
                pass
            except Exception as e:
                cluster_logger.error(f"Failed to start tensor store on {self.node_ip}: {e}")
                
        cluster_logger.info(f"Started {self.num_gpu} tensor store processes on {self.node_ip}")
        
        # Log streaming is no longer needed since logs are directly saved locally
    
    def start_api_server(self, api_server_port: int, config: Dict, node_rank_mapping: Dict[str, List[int]], ray_address: str):
        """Start API server on this VNode (should only be called for pipeline rank 0)."""
        if self.pipeline_rank != 0:
            cluster_logger.warning(f"start_api_server called on non-first node (rank {self.pipeline_rank}). Skipping.")
            return
            
        self.api_server_port = api_server_port
        self.is_api_server_ready = False
        
        command = get_api_server_command(
            model_name=config["model_name"],
            pp_layer_partition=config["pp_layer_partition"],
            parallel_strategy=config["parallel_strategy"],
            host=config.get("host", "0.0.0.0"),
            port=api_server_port,
            dtype=config.get("dtype"),
            max_model_len=config.get("max_model_len"),
            node_rank_mapping=json.dumps(node_rank_mapping),
            ray_address=ray_address,
            gpu_memory_utilization=config.get("gpu_memory_utilization"),
            max_num_batched_tokens=config.get("max_num_batched_tokens"),
            max_num_seqs=config.get("max_num_seqs")
        )
        
        # Prepare log file - save directly to local remote logs directory
        log_file_index = 0
        log_filename = f"apiserver_{self.node_ip}_{log_file_index}.log"
        while os.path.exists(os.path.join(self.remote_log_dir, log_filename)):
            log_file_index += 1
            log_filename = f"apiserver_{self.node_ip}_{log_file_index}.log"
        local_log_path = os.path.join(self.remote_log_dir, log_filename)
        
        # Use SSH to start the API server and stream logs directly to local file
        # Add SSH options to avoid host key verification prompts
        ssh_options = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        # Run command on remote and redirect output to local file
        ssh_command = f"ssh {ssh_options} {self.node_ip} '{command}' > {local_log_path} 2>&1 &"
        
        # Debug: print the command
        cluster_logger.info(f"Executing SSH command for API server: {ssh_command}")
        
        try:
            self.api_server_process = subprocess.Popen(
                ssh_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            cluster_logger.info(f"Started API server on {self.node_ip}:{api_server_port} (pipeline rank 0), log: {local_log_path}")
            
            # Log streaming is no longer needed since logs are directly saved locally
        except Exception as e:
            cluster_logger.error(f"Failed to start API server on {self.node_ip}: {e}")

    def check_tensor_store_status(self, timeout: float = 2.0) -> bool:
        """Check if all tensor store servers on this node are ready."""
        # If tensor_store_port was not set, use the default base port
        if self.tensor_store_port is None:
            raise ValueError("tensor_store_port is not set")

        base_port = self.tensor_store_port
            
        all_ready = True
        
        for i in range(self.num_gpu):
            port = base_port + i
            try:
                with socket.create_connection((self.node_ip, port), timeout=timeout) as sock:
                    sock.settimeout(timeout)
                    # Send status check command using new protocol
                    sock.send(TensorStoreRequest.STATUS_CHECK.value)
                    resp = sock.recv(1)
                    if resp != TensorStoreResponse.READY.value:
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
        if self.pipeline_rank != 0:
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
    
    def get_remote_logs(self):
        """Get all content from remote log files."""
        logs = {}
        
        # Get tensor store logs
        for i in range(self.num_gpu):
            log_filename = f"tensorstore_{self.node_ip}_{i}.log"
            remote_log_path = f"~/logs/{log_filename}"
            
            ssh_command = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {self.node_ip} 'cat {remote_log_path} 2>/dev/null || echo \"\"'"
            
            try:
                result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
                logs[log_filename] = result.stdout
            except Exception as e:
                logs[log_filename] = f"Failed to get log: {e}"
        
        # Get API server log if this is the first node
        if self.pipeline_rank == 0:
            log_filename = f"apiserver_{self.node_ip}.log"
            remote_log_path = f"~/logs/{log_filename}"
            
            ssh_command = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {self.node_ip} 'cat {remote_log_path} 2>/dev/null || echo \"\"'"
            
            try:
                result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
                logs[log_filename] = result.stdout
            except Exception as e:
                logs[log_filename] = f"Failed to get log: {e}"
        
        return logs
    
    def _get_indexed_log_path(self, base_filename):
        """Get a log file path with an index that doesn't exist yet."""
        name, ext = os.path.splitext(base_filename)
        index = 0
        while True:
            indexed_filename = f"{name}_{index}{ext}"
            log_path = os.path.join(self.remote_log_dir, indexed_filename)
            if not os.path.exists(log_path):
                return log_path, indexed_filename
            index += 1

    def stream_tensor_store_logs_continuously(self, callback=None):
        """Continuously stream tensor store logs from remote server in a separate thread."""
        def _stream_worker():
            last_logs = {}
            file_found = {}
            local_log_files = {}
            
            while True:
                try:
                    tensor_store_logs = {}
                    for i in range(self.num_gpu):
                        log_filename = f"tensorstore_{self.node_ip}_{i}.log"
                        remote_log_path = f"~/logs/{log_filename}"
                        ssh_command = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {self.node_ip} 'cat {remote_log_path} 2>/dev/null || echo \"\"'"
                        
                        try:
                            result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
                            tensor_store_logs[log_filename] = result.stdout
                        except Exception as e:
                            tensor_store_logs[log_filename] = f"Failed to get log: {e}"
                    
                    for filename, content in tensor_store_logs.items():
                        if filename not in last_logs:
                            last_logs[filename] = ""
                            file_found[filename] = False
                            local_filename = f"local_{self.node_ip}_{filename}"
                            local_log_path, indexed_filename = self._get_indexed_log_path(local_filename)
                            local_log_files[filename] = open(local_log_path, 'w')
                        
                        if not file_found[filename] and content:
                            file_found[filename] = True
                            cluster_logger.info(f"Remote log file created: {filename} on {self.node_ip}")
                        
                        if len(content) > len(last_logs[filename]):
                            new_content = content[len(last_logs[filename]):]
                            if callback:
                                callback(filename, new_content)
                            else:
                                if new_content.strip():
                                    local_log_files[filename].write(new_content)
                                    local_log_files[filename].flush()
                            last_logs[filename] = content
                    
                    time.sleep(3)
                except Exception as e:
                    if not hasattr(_stream_worker, 'error_count'):
                        _stream_worker.error_count = 0
                    _stream_worker.error_count += 1
                    if _stream_worker.error_count % 10 == 1:
                        cluster_logger.error(f"Error in tensor store log streaming: {e}")
                    time.sleep(2)
        
        stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        stream_thread.start()
        return stream_thread

    def stream_api_server_logs_continuously(self, callback=None):
        """Continuously stream API server logs from remote server in a separate thread."""
        def _stream_worker():
            last_log = ""
            file_found = False
            local_log_file = None
            
            while True:
                try:
                    log_filename = f"apiserver_{self.node_ip}.log"
                    remote_log_path = f"~/logs/{log_filename}"
                    ssh_command = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {self.node_ip} 'cat {remote_log_path} 2>/dev/null || echo \"\"'"
                    
                    try:
                        result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
                        content = result.stdout
                    except Exception as e:
                        content = f"Failed to get log: {e}"
                    
                    if local_log_file is None:
                        local_filename = f"local_{self.node_ip}_{log_filename}"
                        local_log_path, indexed_filename = self._get_indexed_log_path(local_filename)
                        local_log_file = open(local_log_path, 'w')
                    
                    if not file_found and content:
                        file_found = True
                        cluster_logger.info(f"Remote log file created: {log_filename} on {self.node_ip}")
                    
                    if len(content) > len(last_log):
                        new_content = content[len(last_log):]
                        if callback:
                            callback(log_filename, new_content)
                        else:
                            if new_content.strip():
                                local_log_file.write(new_content)
                                local_log_file.flush()
                        last_log = content
                    
                    time.sleep(3)
                except Exception as e:
                    if not hasattr(_stream_worker, 'error_count'):
                        _stream_worker.error_count = 0
                    _stream_worker.error_count += 1
                    if _stream_worker.error_count % 10 == 1:
                        cluster_logger.error(f"Error in API server log streaming: {e}")
                    time.sleep(2)
        
        stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        stream_thread.start()
        return stream_thread

    def get_node_resources(self) -> Dict:
        """Get resource information for this node."""
        return {
            "node_ip": self.node_ip,
            "num_gpu": self.num_gpu,
            "pipeline_rank": self.pipeline_rank,
            "layers": f"[{self.layer_start_id}, {self.layer_end_id})",
            "tensor_parallel_size": self.num_gpu,
            "is_first_stage": self.is_first_stage,
            "is_last_stage": self.is_last_stage,
            "tensor_store_ready": self.is_tensor_store_ready,
            "api_server_ready": self.is_api_server_ready if self.pipeline_rank == 0 else "N/A (not first node)"
        }

    def stop_tensor_store(self):
        """Stop all tensor store servers on this VNode."""
        if not self.is_tensor_store_ready or self.tensor_store_port is None:
            cluster_logger.info(f"TensorStore not running on {self.node_ip}")
            return
        
        cluster_logger.info(f"Stopping TensorStore servers on {self.node_ip}...")
        
        def stop_single_tensor_store(local_rank):
            """Stop a single tensor store process."""
            status_port = self.tensor_store_port + local_rank
            try:
                # Send shutdown command via TCP
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5.0)
                    sock.connect((self.node_ip, status_port))
                    sock.send(TensorStoreRequest.SHUTDOWN.value)
                    response = sock.recv(1)
                    
                    if response == TensorStoreResponse.OK.value:
                        cluster_logger.info(f"TensorStore GPU {local_rank} shutdown accepted on {self.node_ip}")
                        return True
                    else:
                        cluster_logger.warning(f"Unexpected response from TensorStore GPU {local_rank}: {response}")
                        return False
                        
            except Exception as e:
                cluster_logger.error(f"Failed to stop TensorStore GPU {local_rank} on {self.node_ip}: {e}")
                return False
        
        # Stop all tensor store processes in parallel
        with ThreadPoolExecutor(max_workers=self.num_gpu) as executor:
            futures = []
            for local_rank in range(self.num_gpu):
                future = executor.submit(stop_single_tensor_store, local_rank)
                futures.append(future)
            
            # Wait for all to complete
            success_count = 0
            for future in futures:
                if future.result():
                    success_count += 1
            
            cluster_logger.info(f"TensorStore shutdown: {success_count}/{self.num_gpu} processes stopped successfully")
        
        self.is_tensor_store_ready = False
        cluster_logger.info(f"TensorStore shutdown completed on {self.node_ip}")

    def stop_api_server(self, api_server_port: int = None):
        """Stop the API server on this VNode (only applicable for first node)."""
        if self.pipeline_rank != 0:
            cluster_logger.info(f"API server not running on {self.node_ip} (not first node)")
            return
        
        cluster_logger.info(f"Stopping API server on {self.node_ip}...")

        if api_server_port is None:
            api_server_port = self.api_server_port
        
        try:
            # Send shutdown request via HTTP
            url = f"http://{self.node_ip}:{api_server_port}/shutdown"
            response = requests.post(url, timeout=3)
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") == "shutdown_accepted":
                    cluster_logger.info(f"API server shutdown accepted on {self.node_ip}")
                else:
                    cluster_logger.warning(f"Unexpected API server response: {response_data}")
            else:
                cluster_logger.error(f"Failed to stop API server: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError as e:
            # Connection refused is expected if the server is already stopped
            cluster_logger.info(f"API server on {self.node_ip}:{api_server_port} appears to be already stopped (connection refused)")
        except Exception as e:
            cluster_logger.error(f"Failed to stop API server on {self.node_ip}: {e}")
        
        self.is_api_server_ready = False
        cluster_logger.info(f"API server shutdown completed on {self.node_ip}")



def example_usage():
    """Example of how to use the VNode and Pipeline classes."""
    logging.basicConfig(level=logging.INFO)
    
    # Create a cluster
    cluster = Cluster()
    
    # Define node-layer mapping for pipeline parallelism
    # Each tuple is (node_ip, number_of_layers)
    node_layer_mapping = [
        ("172.31.6.247", 32)
    ]

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bucket_name = "hetero-spot-llm-serve-models"
    
    # Create pipeline for Llama-3.1-8B (32 layers total)
    # Option 1: Using HuggingFace (default)
    # config = {
    #     "model_name": model_name,
    #     "total_num_layers": 32,
    #     "pp_layer_partition": "32",
    #     "parallel_strategy": [1],
    #     "max_model_len": 4096,
    #     "gpu_memory_utilization": 0.25,
    #     "max_num_batched_tokens": 4096,
    #     "max_num_seqs": 16,
    #     # tensor_store_base_port and api_server_base_port not specified - will use global defaults
    # }
    
    # Option 2: Using S3 (uncomment to use)
    config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "pp_layer_partition": "32",
        "parallel_strategy": [1],
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.25,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 16,
        "model_source": "s3",
        "s3_path": f"s3://{bucket_name}/{model_name}",
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
    
    cluster_logger.info("Pipeline created successfully!")
    
    # Print VNode information
    for vnode in cluster.pipelines[0].vnodes:
        cluster_logger.info(f"VNode resources: {vnode.get_node_resources()}")
    
    # Get API server details
    api_node = cluster.pipelines[0].vnodes[0]
    api_url = f"http://{api_node.node_ip}:{api_node.api_server_port}"
    cluster_logger.info(f"\nAPI server is running on {api_url}")
    
    try:
        # Interactive inference loop
        print("\n" + "="*50)
        print("🚀 HeteroSpotLLMServe Interactive Mode")
        print("="*50)
        print("Enter your prompts below. Type 'exit' to quit.")
        print(f"API Server: {api_url}")
        print("-"*50)
        
        while True:
            try:
                # Get user input
                prompt = input("\n>>> ")
                
                # Check for exit command
                if prompt.lower().strip() in ['exit', 'quit', 'q']:
                    print("Exiting...")
                    break
                
                if not prompt.strip():
                    continue
                
                # Prepare request payload
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "stream": False
                }
                
                # Send request to API server
                print("🤖 Generating response...")
                response = requests.post(
                    f"{api_url}/v1/completions",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_message = result['choices'][0]['text']
                    print(f"\n💬 Assistant: {assistant_message}")
                else:
                    print(f"❌ Error: HTTP {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except KeyboardInterrupt:
                print("\n\nReceived Ctrl+C. Shutting down...")
                break
            except Exception as e:
                print(f"❌ Error during inference: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n\nReceived Ctrl+C. Shutting down...")
    except Exception as e:
        cluster_logger.error(f"Error in interactive mode: {e}")

    try:
        # Switching Node Test
        print("Switching Node Test")
        old_node_ip = node_layer_mapping[0][0]
        new_node_ip = "172.31.14.46"

        start_switch_time = time.perf_counter()
        cluster.switch_node(old_node_ip, new_node_ip)
        end_switch_time = time.perf_counter()
        print(f"Switching Node Time: {end_switch_time - start_switch_time} seconds")

        input("Press Enter to exit...")
    finally:
        # Graceful shutdown
        print("\n🛑 Initiating graceful shutdown...")
        try:
            cluster.stop_all_pipelines()
            print("✅ Shutdown completed successfully!")
        except Exception as e:
            cluster_logger.error(f"Error during shutdown: {e}")
            print("❌ Error during shutdown. Some processes may still be running.")


if __name__ == "__main__":
    example_usage()