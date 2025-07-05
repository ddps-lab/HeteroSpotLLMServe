"""
Global server for managing multiple pipelines with weighted round-robin scheduling.
"""
import random
import logging
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from VNode import Cluster
from request_handler import Request, RequestInput, RequestOutput, async_request

logger = logging.getLogger(__name__)

@dataclass
class InFlightRequest:
    """Represents a request currently being processed."""
    request: Request
    pipeline_index: int
    task: asyncio.Task
    
class GlobalServer:
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.weights: List[float] = []
        self.cumulative_weights: List[float] = []
        
        # Request queues
        self.waiting_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.inflight_requests: Dict[int, InFlightRequest] = {}  # request_id -> InFlightRequest
        
        # Initialize weights from pipeline ideal throughputs
        for pipeline in cluster.pipelines:
            self.weights.append(pipeline.ideal_throughput)
        
        # Calculate cumulative weights for weighted round-robin
        self._update_cumulative_weights()
        
    def _update_cumulative_weights(self):
        """Update cumulative weights for weighted selection."""
        total_weight = sum(self.weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in self.weights]
            self.cumulative_weights = []
            cumsum = 0
            for w in normalized_weights:
                cumsum += w
                self.cumulative_weights.append(cumsum)
        else:
            # Equal weights if all are zero
            n = len(self.weights)
            self.cumulative_weights = [(i + 1) / n for i in range(n)]
    
    def select_pipeline_index(self) -> int:
        """Select a pipeline index using weighted round-robin scheduling."""
        if not self.cluster.pipelines:
            raise ValueError("No pipelines available")
        
        # Use weighted selection
        rand_val = random.random()
        for i, cum_weight in enumerate(self.cumulative_weights):
            if rand_val < cum_weight:
                return i
        
        # Fallback to last pipeline
        return len(self.cluster.pipelines) - 1
    
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
            # Check waiting queue
            if not self.waiting_queue.empty():
                try:
                    # Get request from queue
                    request = await self.waiting_queue.get()
                    
                    # Select pipeline using weighted round-robin
                    pipeline_index = self.select_pipeline_index()
                    
                    # Create task to send request
                    task = asyncio.create_task(self._handle_request(request, pipeline_index))
                    
                    # Track in-flight request
                    self.inflight_requests[request.request_id] = InFlightRequest(
                        request=request,
                        pipeline_index=pipeline_index,
                        task=task
                    )
                    
                    logger.info(f"Dispatched request {request.request_id} to pipeline {pipeline_index}")
                    
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
            
            # Put back in waiting queue for retry
            await self.waiting_queue.put(request)
            logger.info(f"Request {request.request_id} added back to waiting queue for retry")
    
    async def _cleanup_completed_requests(self):
        """Remove completed requests from inflight tracking."""
        completed_ids = []
        
        for request_id, inflight in self.inflight_requests.items():
            if inflight.task.done():
                completed_ids.append(request_id)
        
        for request_id in completed_ids:
            del self.inflight_requests[request_id]
    
    async def add_request(self, request_input: RequestInput) -> Request:
        """Add a new request to the waiting queue.
        
        Args:
            request_input: The input for the request
            
        Returns:
            The created Request object
        """
        request = Request.create(request_input)
        await self.waiting_queue.put(request)
        logger.info(f"Added request {request.request_id} to waiting queue")
        return request