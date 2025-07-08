"""
Unit test for single pipeline with multiple nodes (Pipeline Parallelism).
Tests creating one pipeline spanning across two nodes with layer partitioning.
"""
import asyncio
import logging
import time
import concurrent.futures
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_server import GlobalServer
from request_handler import generate_random_requests

logger = logging.getLogger(__name__)

async def test_single_pipeline_multi_node():
    """Test single pipeline with multiple nodes using pipeline parallelism."""
    # Configure logging for test script only
    test_logger = logging.getLogger(__name__)
    test_logger.setLevel(logging.INFO)
    
    # Create console handler for test logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    test_logger.addHandler(console_handler)
    test_logger.propagate = False
    
    global_server = GlobalServer()

    # Configuration for single pipeline across two nodes
    node_ip_1 = "172.31.18.31"
    node_ip_2 = "172.31.23.180"
    
    # Pipeline Parallelism: 16 layers on each node (total 32 layers)
    node_layer_mapping = [
        (node_ip_1, 16),  # Node 1 handles layers 0-15
        (node_ip_2, 16)   # Node 2 handles layers 16-31
    ]
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bucket_name = "hetero-spot-llm-serve-models"
    config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "pp_layer_partition": "16,16",  # Pipeline partition: 16 layers per node
        "parallel_strategy": [1, 1],     # 1 GPU per node (no tensor parallelism)
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.25,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 16,
        "model_source": "s3",
        "s3_path": f"s3://{bucket_name}/{model_name}",
    }
    dummy_throughput = 80  # Lower throughput due to pipeline overhead

    # Create pipeline helper function
    async def create_pipeline_async():
        """Create pipeline in a separate thread to avoid blocking"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, 
                global_server.create_pipeline,
                node_layer_mapping,
                config,
                dummy_throughput
            )
        test_logger.info("Multi-node pipeline creation completed")
    
    # Start pipeline creation task
    pipeline_task = asyncio.create_task(create_pipeline_async())
    
    # Start the global server in the background
    server_task = asyncio.create_task(global_server.run_global_server())

    # Generate and send requests
    num_requests = 100  # Fewer requests for multi-node testing
    request_inputs = generate_random_requests(
        num_prompts=num_requests,
        input_len=1024,   # Shorter input for faster processing
        output_len=128,   # Shorter output for faster processing
        model_name=model_name,
        ignore_eos=True  # Ignore EOS to ensure consistent output length
    )
    
    # Create tasks for all requests
    tasks = []
    requests = []
    
    async def send_request_with_delay(index, delay):
        """Send a request after a delay"""
        await asyncio.sleep(delay)
        request = await global_server.add_request(request_inputs[index])
        test_logger.info(f"[{index}] Added request {request.request_id} with prompt: {request.input.prompt[:50]}...")
        return request
    
    try:
        # Send requests with some delay between them
        for i in range(num_requests):
            delay = i * 3  # 3 second delay between requests
            task = asyncio.create_task(send_request_with_delay(i, delay))
            tasks.append(task)
        
        # Wait for all requests to be added
        requests = await asyncio.gather(*tasks)
        test_logger.info(f"All {num_requests} requests submitted to multi-node pipeline")
        
        # Monitor completion
        start_time = time.time()
        completed_count = 0
        last_status_time = 0
        
        while completed_count < num_requests:
            await asyncio.sleep(1)
            
            # Log pipeline status periodically
            elapsed = time.time() - start_time
            if elapsed - last_status_time >= 20:  # Every 20 seconds
                last_status_time = elapsed
                test_logger.info(f"Pipeline status at {elapsed:.0f}s:")
                for i, pipeline in enumerate(global_server.cluster.pipelines):
                    test_logger.info(f"  Pipeline {i}: ready={pipeline.is_ready}, throughput={pipeline.ideal_throughput}")
                    test_logger.info(f"    Nodes: {len(pipeline.vnodes)} nodes")
                    for j, vnode in enumerate(pipeline.vnodes):
                        test_logger.info(f"      Node {j} ({vnode.node_ip}): layers=[{vnode.layer_start_id}, {vnode.layer_end_id})")
            
            # Check which requests are completed
            for i, req in enumerate(requests):
                if req.output and req.output.success and not hasattr(req, '_logged'):
                    completed_count += 1
                    req._logged = True  # Mark as logged to avoid duplicate logs
                    
                    test_logger.info(f"[{i}] Request {req.request_id} completed!")
                    test_logger.info(f"[{i}] Response: {req.output.generated_text[:100]}...")
                    test_logger.info(f"[{i}] Metrics - Tokens: {req.output.output_tokens}, "
                                  f"Latency: {req.output.latency:.2f}s, TTFT: {req.output.ttft:.3f}s")
            
            # Timeout after 8 minutes
            if elapsed > 480:
                test_logger.warning("Timeout reached, stopping...")
                break
        
        # Final statistics
        successful = sum(1 for r in requests if r.output and r.output.success)
        test_logger.info(f"\nFinal Results: {successful}/{num_requests} requests successful")
        
        # Calculate average latency for successful requests
        successful_requests = [r for r in requests if r.output and r.output.success]
        if successful_requests:
            avg_latency = sum(r.output.latency for r in successful_requests) / len(successful_requests)
            avg_ttft = sum(r.output.ttft for r in successful_requests) / len(successful_requests)
            total_tokens = sum(r.output.output_tokens for r in successful_requests)
            test_logger.info(f"Performance metrics:")
            test_logger.info(f"  Average latency: {avg_latency:.2f}s")
            test_logger.info(f"  Average TTFT: {avg_ttft:.3f}s")
            test_logger.info(f"  Total tokens generated: {total_tokens}")
            test_logger.info(f"  Throughput: {total_tokens / (time.time() - start_time):.2f} tokens/s")
        
    except KeyboardInterrupt:
        test_logger.info("Shutting down...")
    finally:
        # Cancel all tasks
        server_task.cancel()
        
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        
        # Stop all pipelines
        global_server.cluster.stop_all_pipelines()

if __name__ == "__main__":
    asyncio.run(test_single_pipeline_multi_node())