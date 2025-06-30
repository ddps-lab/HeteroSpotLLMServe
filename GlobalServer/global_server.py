"""
Global server for managing multiple pipelines with weighted round-robin scheduling.
"""
import random
import logging
from typing import List

from VNode import Cluster
from request_handler import RequestInput, RequestOutput, async_request

class GlobalServer:
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.weights: List[float] = []
        self.cumulative_weights: List[float] = []
        
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
    
    async def send_request(self, request: RequestInput, pipeline_index: int) -> RequestOutput:
        """Send a request to a specific pipeline."""
        pipeline = self.cluster.pipelines[pipeline_index]
        first_vnode = pipeline.vnodes[0]
        
        # Set API URL for the request
        request.api_url = f"http://{first_vnode.node_ip}:{first_vnode.api_server_port}/v1/completions"
        
        return await async_request(request)