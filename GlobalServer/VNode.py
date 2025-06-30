from typing import List, Dict, Tuple
import socket
import ray
import subprocess
import logging
import time
import os
from command import get_tensor_store_command, get_api_server_command, DEFAULT_TENSOR_STORE_BASE_PORT, DEFAULT_API_SERVER_BASE_PORT
import json
import sys
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for protocols import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocols import TensorStoreRequest, TensorStoreResponse

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
        """
        # Validate required config parameters
        required_keys = ["model_name", "total_num_layers", "pp_layer_partition", "parallel_strategy"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"config must contain '{key}'")
            
        pipeline = Pipeline()
        pipeline.initialize_pipeline(node_layer_mapping, config)
        self.pipelines.append(pipeline)

    def stop_all_pipelines(self):
        """Stop all pipelines in the cluster."""
        logging.info(f"Stopping {len(self.pipelines)} pipelines...")
        
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
                    logging.info(f"Pipeline {pipeline_idx} stopped successfully")
                except Exception as e:
                    logging.error(f"Failed to stop pipeline {pipeline_idx}: {e}")
        
        logging.info("All pipelines stopped")

    def switch_node(self, old_node_ip: str, new_node_ip: str):
        """
        Switch a node in the cluster by replacing old_node_ip with new_node_ip.
        
        Args:
            old_node_ip: IP address of the node to be replaced
            new_node_ip: IP address of the new node
        """
        logging.info(f"Starting node switch: {old_node_ip} -> {new_node_ip}")
        
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
        
        logging.info(f"Found VNode in pipeline: {target_vnode}")
        
        # Delegate to the pipeline's switch_node method
        target_pipeline.switch_node(old_node_ip, new_node_ip)
        
        logging.info(f"Node switch completed successfully: {old_node_ip} -> {new_node_ip}")

class Pipeline:
    def __init__(self):
        self.vnodes: List[VNode] = []
        self.model_name: str = ""
        self.total_layers: int = 0
        self.node_rank_mapping: Dict[str, List[int]] = {}

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
                tensor_store_port = tensor_store_base_port
            else:
                tensor_store_port = DEFAULT_TENSOR_STORE_BASE_PORT # use global default in command.py
            vnode.start_tensor_store(tensor_store_port, config)
        
        # Generate node_rank_mapping based on vnodes
        self._generate_node_rank_mapping()
        
        # Start API server only on the first node (pipeline rank 0)
        first_vnode = self.vnodes[0]
        api_server_base_port = config.get("api_server_base_port", DEFAULT_API_SERVER_BASE_PORT)
        first_vnode.start_api_server(api_server_base_port, config, self.node_rank_mapping)
        
        # Wait for all services to be ready
        self._wait_for_services()
    
    def _generate_node_rank_mapping(self):
        """Generate node_rank_mapping based on current vnodes."""
        self.node_rank_mapping = {}
        current_rank = 0
        
        for vnode in self.vnodes:
            ranks = list(range(current_rank, current_rank + vnode.num_gpu))
            self.node_rank_mapping[vnode.node_ip] = ranks
            current_rank += vnode.num_gpu
        
        logging.info(f"Generated node_rank_mapping: {self.node_rank_mapping}")
    
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
            
            # Use carriage return to overwrite the same line
            status_msg = f"\rWaiting for services{dots_str} Tensor stores: {ready_ts}/{len(self.vnodes)}, API server: {api_status}"
            sys.stdout.write(status_msg)
            sys.stdout.flush()
            
            if not (all(tensor_store_statuses) and api_server_ready):
                time.sleep(1)
        
        # Print final status on a new line
        print(f"\n✓ All services ready! Tensor stores: {len(self.vnodes)}/{len(self.vnodes)}, API server: Ready")
    
    def start_log_streaming(self):
        """Start streaming logs from all VNodes."""
        logging.info("Starting log streaming from all nodes...")
        threads = []
        for vnode in self.vnodes:
            thread = vnode.stream_logs_continuously()
            threads.append(thread)
        return threads

    def stop_pipeline(self):
        """Stop all services in the pipeline."""
        logging.info(f"Stopping pipeline with {len(self.vnodes)} nodes...")
        
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
                    logging.info(f"{service_type} stopped successfully on {node_ip}")
                except Exception as e:
                    logging.error(f"Failed to stop {service_type} on {node_ip}: {e}")
        
        logging.info("Pipeline shutdown completed")

    def switch_node(self, old_node_ip: str, new_node_ip: str):
        """
        Switch a node in this pipeline by replacing old_node_ip with new_node_ip.
        
        Args:
            old_node_ip: IP address of the node to be replaced
            new_node_ip: IP address of the new node
        """
        logging.info(f"Switching node in pipeline: {old_node_ip} -> {new_node_ip}")
        
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
        
        # Get GPU count for the new node
        new_num_gpu = None
        while True:
            for node_info in ray.nodes():
                ray_node_ip = node_info.get("NodeManagerAddress")
                if ray_node_ip == new_node_ip:
                    new_num_gpu = int(node_info.get("Resources").get("GPU", 0))
                    break
            if new_num_gpu is not None and new_num_gpu > 0:
                break
            logging.info(f"Waiting for new node {new_node_ip} to be entered into Ray cluster...")
            time.sleep(1)
        
        # Create new VNode with same configuration but new IP and GPU count
        new_vnode = VNode(
            node_ip=new_node_ip,
            num_gpu=new_num_gpu,
            pipeline_rank=target_vnode.pipeline_rank,
            layer_start_id=target_vnode.layer_start_id,
            layer_end_id=target_vnode.layer_end_id,
            total_layers=target_vnode.total_layers
        )
        
        # Start tensor store on the new node
        tensor_store_port = target_vnode.tensor_store_port
        new_vnode.start_tensor_store(tensor_store_port, self.config)
        
        # Check tensor store status
        logging.info(f"Checking tensor store status on new node {new_node_ip}")
        status_check_time = 0
        while not new_vnode.check_tensor_store_status():
            status_check_time += 1
            logging.info(f"Waiting for tensor store to be ready on new node ({status_check_time})...")
            time.sleep(2)
        
        logging.info(f"Tensor store ready on new node {new_node_ip}")
        
        start_downtime = time.perf_counter()

        # Stop API server
        logging.info(f"Stopping API server which contains old node {old_node_ip}")
        self.vnodes[0].stop_api_server()
        
        # Replace the VNode immediately (don't wait for API server stop)
        self.vnodes[target_index] = new_vnode
        
        # Update node_rank_mapping
        self._generate_node_rank_mapping()
        
        # Start API server on the first node (pipeline rank 0)
        first_vnode = self.vnodes[0]
        api_server_port = target_vnode.api_server_port if target_vnode.api_server_port else DEFAULT_API_SERVER_BASE_PORT
        first_vnode.start_api_server(api_server_port, self.config, self.node_rank_mapping)
        
        # Check API server status
        logging.info(f"Checking API server status on node {first_vnode.node_ip}")
        status_check_time = 0
        while not first_vnode.check_api_server_status():
            status_check_time += 1
            logging.info(f"Waiting for API server to be ready ({status_check_time})...")
            time.sleep(2)
        
        logging.info(f"API server ready on node {first_vnode.node_ip}")

        end_downtime = time.perf_counter()
        downtime = end_downtime - start_downtime
        logging.info(f"Downtime: {downtime} seconds")
        
        # Stop tensor store on old node
        logging.info(f"Stopping tensor store on old node {old_node_ip}")
        target_vnode.stop_tensor_store()
        
        logging.info(f"Node switch completed in pipeline: {old_node_ip} -> {new_node_ip}")


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
        
        # Log streaming threads
        self.tensor_store_log_thread = None
        self.api_server_log_thread = None
        
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
        dtype = config.get("dtype")
        
        # Check if using S3 or HuggingFace
        model_source = config.get("model_source", "huggingface")  # Default to HuggingFace
        s3_path = config.get("s3_path") if model_source == "s3" else None
        aws_profile = config.get("aws_profile") if model_source == "s3" else None
        
        if model_source == "s3" and not s3_path:
            raise ValueError("s3_path must be provided when model_source is 's3'")
        
        # Start tensor store processes for each GPU (for TP)
        for local_rank in range(self.num_gpu):
            command = get_tensor_store_command(
                model_name=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                local_rank=local_rank,
                start_layer_id=self.layer_start_id,
                end_layer_id=self.layer_end_id,
                status_port=tensor_store_port,  # Base port, local_rank will be added in command
                dtype=dtype,
                s3_path=s3_path,
                aws_profile=aws_profile
            )
            
            # Prepare log files
            log_filename = f"tensorstore_{self.node_ip}_{local_rank}.log"
            log_path = os.path.join(self.log_dir, log_filename)
            
            # Use SSH to start the process on the remote node with log redirection
            # Add SSH options to avoid host key verification prompts
            ssh_options = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
            # Ensure log directory exists on remote node and run the command
            ssh_command = f"ssh {ssh_options} {self.node_ip} 'mkdir -p {self.log_dir} && {command} > {log_path} 2>&1 &'"
            
            # Debug: print the command
            logging.info(f"Executing SSH command: {ssh_command}")
            
            try:
                process = subprocess.Popen(
                    ssh_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.tensor_store_processes.append(process)
                logging.info(f"Started tensor store process (rank {local_rank}) on {self.node_ip}, log: {log_path}")
                
                # Wait a bit and check if process started successfully
                time.sleep(0.5)
                _, stderr = process.communicate(timeout=0.1)
                if stderr:
                    logging.error(f"SSH command stderr: {stderr.decode()}")
            except subprocess.TimeoutExpired:
                # This is expected - process is still running
                pass
            except Exception as e:
                logging.error(f"Failed to start tensor store on {self.node_ip}: {e}")
                
        logging.info(f"Started {self.num_gpu} tensor store processes on {self.node_ip}")
        
        # Start tensor store log streaming immediately after starting processes
        if self.tensor_store_log_thread is None:
            logging.info(f"Starting tensor store log streaming for {self.node_ip}")
            self.tensor_store_log_thread = self.stream_tensor_store_logs_continuously()
    
    def start_api_server(self, api_server_port: int, config: Dict, node_rank_mapping: Dict[str, List[int]]):
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
            host=config.get("host", "0.0.0.0"),
            port=api_server_port,
            dtype=config.get("dtype"),
            max_model_len=config.get("max_model_len"),
            node_rank_mapping=json.dumps(node_rank_mapping),
            gpu_memory_utilization=config.get("gpu_memory_utilization"),
            max_num_batched_tokens=config.get("max_num_batched_tokens"),
            max_num_seqs=config.get("max_num_seqs")
        )
        
        # Prepare log file
        log_filename = f"apiserver_{self.node_ip}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        # Use SSH to start the API server on the remote node with log redirection
        # Add SSH options to avoid host key verification prompts
        ssh_options = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        # Ensure log directory exists on remote node and run the command
        ssh_command = f"ssh {ssh_options} {self.node_ip} 'mkdir -p {self.log_dir} && {command} > {log_path} 2>&1 &'"
        
        # Debug: print the command
        logging.info(f"Executing SSH command for API server: {ssh_command}")
        
        try:
            self.api_server_process = subprocess.Popen(
                ssh_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logging.info(f"Started API server on {self.node_ip}:{api_server_port} (pipeline rank 0), log: {log_path}")
            
            # Start API server log streaming if not already started
            if self.api_server_log_thread is None:
                logging.info(f"Starting API server log streaming for {self.node_ip}")
                self.api_server_log_thread = self.stream_api_server_logs_continuously()
        except Exception as e:
            logging.error(f"Failed to start API server on {self.node_ip}: {e}")

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
            log_path = os.path.join(self.log_dir, indexed_filename)
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
                            logging.info(f"Remote log file created: {filename} on {self.node_ip}")
                        
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
                        logging.error(f"Error in tensor store log streaming: {e}")
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
                        logging.info(f"Remote log file created: {log_filename} on {self.node_ip}")
                    
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
                        logging.error(f"Error in API server log streaming: {e}")
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
            "tensor_parallel_size": self.tensor_parallel_size,
            "is_first_stage": self.is_first_stage,
            "is_last_stage": self.is_last_stage,
            "tensor_store_ready": self.is_tensor_store_ready,
            "api_server_ready": self.is_api_server_ready if self.pipeline_rank == 0 else "N/A (not first node)"
        }

    def stop_tensor_store(self):
        """Stop all tensor store servers on this VNode."""
        if not self.is_tensor_store_ready or self.tensor_store_port is None:
            logging.info(f"TensorStore not running on {self.node_ip}")
            return
        
        logging.info(f"Stopping TensorStore servers on {self.node_ip}...")
        
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
                        logging.info(f"TensorStore GPU {local_rank} shutdown accepted on {self.node_ip}")
                        return True
                    else:
                        logging.warning(f"Unexpected response from TensorStore GPU {local_rank}: {response}")
                        return False
                        
            except Exception as e:
                logging.error(f"Failed to stop TensorStore GPU {local_rank} on {self.node_ip}: {e}")
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
            
            logging.info(f"TensorStore shutdown: {success_count}/{self.num_gpu} processes stopped successfully")
        
        self.is_tensor_store_ready = False
        logging.info(f"TensorStore shutdown completed on {self.node_ip}")
        
        # Start cleanup thread for tensor store log streaming
        def cleanup_tensor_store_log():
            time.sleep(5)
            if self.tensor_store_log_thread and self.tensor_store_log_thread.is_alive():
                logging.info(f"Stopping tensor store log streaming thread on {self.node_ip}")
                self.tensor_store_log_thread = None
        
        cleanup_thread = threading.Thread(target=cleanup_tensor_store_log, daemon=True)
        cleanup_thread.start()

    def stop_api_server(self):
        """Stop the API server on this VNode (only applicable for first node)."""
        if self.pipeline_rank != 0:
            logging.info(f"API server not running on {self.node_ip} (not first node)")
            return
            
        if not self.is_api_server_ready or self.api_server_port is None:
            logging.info(f"API server not running on {self.node_ip}")
            return
        
        logging.info(f"Stopping API server on {self.node_ip}...")
        
        try:
            # Send shutdown request via HTTP
            url = f"http://{self.node_ip}:{self.api_server_port}/shutdown"
            response = requests.post(url, timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") == "shutdown_accepted":
                    logging.info(f"API server shutdown accepted on {self.node_ip}")
                else:
                    logging.warning(f"Unexpected API server response: {response_data}")
            else:
                logging.error(f"Failed to stop API server: HTTP {response.status_code}")
                
        except Exception as e:
            logging.error(f"Failed to stop API server on {self.node_ip}: {e}")
        
        self.is_api_server_ready = False
        logging.info(f"API server shutdown completed on {self.node_ip}")
        
        # Start cleanup thread for API server log streaming
        def cleanup_api_server_log():
            time.sleep(5)
            if self.api_server_log_thread and self.api_server_log_thread.is_alive():
                logging.info(f"Stopping API server log streaming thread on {self.node_ip}")
                self.api_server_log_thread = None
        
        cleanup_thread = threading.Thread(target=cleanup_api_server_log, daemon=True)
        cleanup_thread.start()



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
    
    logging.info("Pipeline created successfully!")
    
    # Print VNode information
    for vnode in cluster.pipelines[0].vnodes:
        logging.info(f"VNode resources: {vnode.get_node_resources()}")
    
    # Get API server details
    api_node = cluster.pipelines[0].vnodes[0]
    api_url = f"http://{api_node.node_ip}:{api_node.api_server_port}"
    logging.info(f"\nAPI server is running on {api_url}")
    
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
        logging.error(f"Error in interactive mode: {e}")

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
            logging.error(f"Error during shutdown: {e}")
            print("❌ Error during shutdown. Some processes may still be running.")


if __name__ == "__main__":
    example_usage()