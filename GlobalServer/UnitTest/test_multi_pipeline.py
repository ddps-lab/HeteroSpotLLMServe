"""
Unit test for multiple pipeline management in GlobalServer.
Tests creating two pipelines and removing one.
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

async def test_multi_pipeline():
    """Test multiple pipeline creation and removal."""
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

    # Configuration for two pipelines
    node_ip_1 = "172.31.18.31"
    node_ip_2 = "172.31.23.180"
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bucket_name = "hetero-spot-llm-serve-models"
    base_config = {
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

    # Create pipeline helper function
    async def create_pipeline_async(node_ip, throughput, pipeline_name):
        """Create pipeline in a separate thread to avoid blocking"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, 
                global_server.create_pipeline,
                [(node_ip, 32)],
                base_config,
                throughput
            )
        test_logger.info(f"{pipeline_name} creation completed with throughput {throughput}")
    
    # Start pipeline creation tasks
    pipeline_task_1 = asyncio.create_task(create_pipeline_async(node_ip_1, 100, "Pipeline 1"))
    pipeline_task_2 = asyncio.create_task(create_pipeline_async(node_ip_2, 150, "Pipeline 2"))
    
    # Start the global server in the background
    server_task = asyncio.create_task(global_server.run_global_server())

    # Schedule pipeline removal after 2 minutes
    async def remove_pipeline_after_delay():
        """Remove one pipeline after 2 minute delay"""
        await asyncio.sleep(120)  # Wait 2 minutes
        try:
            test_logger.info("Starting pipeline removal test")
            
            # Check which pipelines are available
            test_logger.info(f"Total pipelines before removal: {len(global_server.cluster.pipelines)}")
            for i, pipeline in enumerate(global_server.cluster.pipelines):
                test_logger.info(f"Pipeline {i}: throughput={pipeline.ideal_throughput}, ready={pipeline.is_ready}")
            
            # Remove pipeline 0 (first pipeline)
            if len(global_server.cluster.pipelines) > 1:
                pipeline_to_remove = 0
                test_logger.info(f"Removing pipeline {pipeline_to_remove}")
                
                # Execute removal in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    await loop.run_in_executor(
                        executor,
                        global_server.remove_pipeline,
                        pipeline_to_remove
                    )
                
                test_logger.info(f"Pipeline {pipeline_to_remove} removed successfully")
                test_logger.info(f"Remaining pipelines: {len(global_server.cluster.pipelines)}")
            else:
                test_logger.warning("Not enough pipelines to remove")
                
        except Exception as e:
            test_logger.error(f"Pipeline removal test failed: {e}")
    
    # Start pipeline removal task
    removal_task = asyncio.create_task(remove_pipeline_after_delay())

    # Generate and send requests
    num_requests = 100
    request_inputs = generate_random_requests(
        num_prompts=num_requests,
        input_len=1024,
        output_len=128,
        model_name=model_name,
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
        test_logger.info(f"All {num_requests} requests submitted")
        
        # Monitor completion and pipeline status
        start_time = time.time()
        completed_count = 0
        last_status_time = 0
        
        while completed_count < num_requests:
            await asyncio.sleep(1)
            
            # Log pipeline status periodically
            elapsed = time.time() - start_time
            if elapsed - last_status_time >= 30:  # Every 30 seconds
                last_status_time = elapsed
                test_logger.info(f"Pipeline status at {elapsed:.0f}s:")
                for i, pipeline in enumerate(global_server.cluster.pipelines):
                    test_logger.info(f"  Pipeline {i}: ready={pipeline.is_ready}, throughput={pipeline.ideal_throughput}")
            
            # Check which requests are completed
            for i, req in enumerate(requests):
                if req.output and req.output.success and not hasattr(req, '_logged'):
                    completed_count += 1
                    req._logged = True  # Mark as logged to avoid duplicate logs
                    
                    test_logger.info(f"[{i}] Request {req.request_id} completed!")
                    test_logger.info(f"[{i}] Response: {req.output.generated_text[:100]}...")
                    test_logger.info(f"[{i}] Metrics - Tokens: {req.output.output_tokens}, "
                              f"Latency: {req.output.latency:.2f}s")
            
            # Timeout after 8 minutes
            if elapsed > 480:
                test_logger.warning("Timeout reached, stopping...")
                break
        
        # Final statistics
        successful = sum(1 for r in requests if r.output and r.output.success)
        test_logger.info(f"\nFinal Results: {successful}/{num_requests} requests successful")
        test_logger.info(f"Final pipeline count: {len(global_server.cluster.pipelines)}")
        
    except KeyboardInterrupt:
        test_logger.info("Shutting down...")
    finally:
        # Cancel all tasks
        server_task.cancel()
        removal_task.cancel()
        
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        
        try:
            await removal_task
        except asyncio.CancelledError:
            pass
        
        # Stop all remaining pipelines
        global_server.cluster.stop_all_pipelines()

if __name__ == "__main__":
    asyncio.run(test_multi_pipeline())