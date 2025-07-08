"""
Global server for managing multiple pipelines with weighted round-robin scheduling.
"""
import random
import logging
import asyncio
import time
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from VNode import Cluster
from request_handler import Request, RequestInput, RequestOutput, async_request

# Configure logging for GlobalServer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get the GlobalServer directory path
GLOBAL_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_BASE_DIR = os.path.join(GLOBAL_SERVER_DIR, "logs")
os.makedirs(LOG_BASE_DIR, exist_ok=True)

# Create file handler for GlobalServer logs
global_server_log_file = os.path.join(LOG_BASE_DIR, "GlobalServer.log")
file_handler = logging.FileHandler(global_server_log_file)
file_handler.setLevel(logging.INFO)

# Create console handler for terminal output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to root logger (we handle logging ourselves)
logger.propagate = False

@dataclass
class InFlightRequest:
    """Represents a request currently being processed."""
    request: Request
    pipeline_index: int
    task: asyncio.Task
    
class GlobalServer:
    def __init__(self, cluster: Optional[Cluster] = None):
        self.cluster = cluster if cluster is not None else Cluster()
        
        # Request queues - urgent_queue for high priority requests (e.g., retries)
        self.urgent_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.waiting_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.inflight_requests: Dict[int, InFlightRequest] = {}  # request_id -> InFlightRequest
        
    
    def select_pipeline_index(self) -> int:
        """Select a pipeline index using weighted round-robin scheduling.
        Only considers pipelines that are ready.
        
        Returns:
            Pipeline index if a ready pipeline is available, -1 otherwise
        """
        # Get ready pipelines with their indices
        ready_pipelines = [(i, p) for i, p in enumerate(self.cluster.pipelines) if p.is_ready]
        
        if not ready_pipelines:
            return -1  # No ready pipelines available
        
        # Extract weights for ready pipelines
        weights = [p.ideal_throughput for _, p in ready_pipelines]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            # Equal probability if all weights are zero
            selected_idx = random.randint(0, len(ready_pipelines) - 1)
            return ready_pipelines[selected_idx][0]
        
        # Calculate cumulative weights dynamically
        cumulative_weights = []
        cumsum = 0
        for w in weights:
            cumsum += w / total_weight
            cumulative_weights.append(cumsum)
        
        # Use weighted selection
        rand_val = random.random()
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val < cum_weight:
                return ready_pipelines[i][0]
        
        # Fallback to last ready pipeline
        return ready_pipelines[-1][0]
    
    async def send_request(self, request: Request, pipeline_index: int) -> RequestOutput:
        """Send a request to a specific pipeline."""
        pipeline = self.cluster.pipelines[pipeline_index]
        first_vnode = pipeline.vnodes[0]
        
        # Set API URL for the request
        request.input.api_url = f"http://{first_vnode.node_ip}:{first_vnode.api_server_port}/v1/completions"
        
        # Mark when request was sent
        request.sended_at = time.time()
        
        return await async_request(request)
    
    async def run_global_server(self, check_interval: float = 0.1):
        """Main server loop that processes requests from the waiting queue.
        
        Args:
            check_interval: How often to check the waiting queue (in seconds)
        """
        logger.info("Starting global server...")
        
        while True:
            # First check if there are any ready pipelines
            pipeline_index = self.select_pipeline_index()
            
            # Only process requests if there are ready pipelines
            if pipeline_index != -1:
                request = None
                queue_type = None
                
                # Check urgent queue first, then waiting queue
                if not self.urgent_queue.empty():
                    try:
                        request = await self.urgent_queue.get()
                        queue_type = "urgent"
                    except Exception as e:
                        logger.error(f"Error getting request from urgent queue: {e}")
                        
                elif not self.waiting_queue.empty():
                    try:
                        request = await self.waiting_queue.get()
                        queue_type = "waiting"
                    except Exception as e:
                        logger.error(f"Error getting request from waiting queue: {e}")
                
                if request:
                    try:
                        # Create task to send request
                        task = asyncio.create_task(self._handle_request(request, pipeline_index))
                        
                        # Track in-flight request
                        self.inflight_requests[request.request_id] = InFlightRequest(
                            request=request,
                            pipeline_index=pipeline_index,
                            task=task
                        )
                        
                        logger.info(f"Dispatched request {request.request_id} from {queue_type} queue to pipeline {pipeline_index}")
                        
                    except Exception as e:
                        logger.error(f"Error dispatching request: {e}")
            
            # Clean up completed requests
            await self._cleanup_completed_requests()
            
            # Wait before next check
            await asyncio.sleep(check_interval)
    
    async def _handle_request(self, request: Request, pipeline_index: int):
        """Handle a single request and update its output.
        
        Args:
            request: The request to process
            pipeline_index: Which pipeline to send to
        """
        try:
            output = await self.send_request(request, pipeline_index)
            request.output = output
            logger.info(f"Request {request.request_id} completed successfully")
        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")
            # Mark when request was halted
            request.halted_at = time.time()
            request.retry_count += 1
            
            # Put back in urgent queue for retry (higher priority)
            await self.urgent_queue.put(request)
            logger.info(f"Request {request.request_id} added to urgent queue for retry (attempt #{request.retry_count})")
    
    async def _cleanup_completed_requests(self):
        """Remove completed requests from inflight tracking."""
        completed_ids = []
        
        for request_id, inflight in self.inflight_requests.items():
            if inflight.task.done():
                completed_ids.append(request_id)
        
        for request_id in completed_ids:
            del self.inflight_requests[request_id]
    
    async def add_request(self, request_input: RequestInput, urgent: bool = False) -> Request:
        """Add a new request to the appropriate queue.
        
        Args:
            request_input: The input for the request
            urgent: If True, add to urgent queue; otherwise to waiting queue
            
        Returns:
            The created Request object
        """
        request = Request.create(request_input)
        
        if urgent:
            await self.urgent_queue.put(request)
            logger.info(f"Added request {request.request_id} to urgent queue")
        else:
            await self.waiting_queue.put(request)
            logger.info(f"Added request {request.request_id} to waiting queue")
            
        return request
    
    def create_pipeline(self, node_layer_mapping: List[Tuple[str, int]], 
                       config: Dict, 
                       ideal_throughput: float):
        """Create a new pipeline in the cluster.
        
        Args:
            node_layer_mapping: List of (node_ip, num_layers) tuples
            config: Configuration dictionary for the pipeline
            ideal_throughput: Expected throughput for this pipeline (requests/sec)
        """
        # Create pipeline in the cluster
        self.cluster.create_pipeline(node_layer_mapping, config, ideal_throughput)
        
        logger.info(f"Created new pipeline with ideal throughput {ideal_throughput}. "
                   f"Total pipelines: {len(self.cluster.pipelines)}")
    
    
    def remove_pipeline(self, pipeline_index: int):
        """Remove a pipeline from the cluster.
        
        Args:
            pipeline_index: Index of the pipeline to remove
        """
        assert 0 <= pipeline_index < len(self.cluster.pipelines), f"Invalid pipeline index: {pipeline_index}"
        
        # Stop the pipeline
        self.cluster.pipelines[pipeline_index].stop_pipeline()
        
        # Remove from cluster
        removed_pipeline = self.cluster.pipelines.pop(pipeline_index)
        
        # Update cluster's total ideal throughput
        self.cluster.ideal_throughput -= removed_pipeline.ideal_throughput
        
        logger.info(f"Removed pipeline {pipeline_index} with weight {removed_pipeline.ideal_throughput}")
    
    def switch_node(self, old_node_ip: str, new_node_ip: str):
        """Switch a node in the cluster by replacing old_node_ip with new_node_ip.
        
        Args:
            old_node_ip: IP address of the node to be replaced
            new_node_ip: IP address of the new node
        """
        logger.info(f"Starting node switch through GlobalServer: {old_node_ip} -> {new_node_ip}")
        self.cluster.switch_node(old_node_ip, new_node_ip)
        logger.info(f"Node switch completed: {old_node_ip} -> {new_node_ip}")