"""
Unit test for node switching functionality in GlobalServer.
"""
import asyncio
import logging
import concurrent.futures
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_server import GlobalServer
from request_handler import generate_random_requests
from test_utils import (
    setup_test_logger,
    send_and_monitor_requests,
    log_pipeline_status
)

logger = logging.getLogger(__name__)

async def test_node_switch():
    """Test node switching functionality."""
    # Configure logging
    test_logger = setup_test_logger(__name__)
    
    global_server = GlobalServer()

    head_node_ip = "172.31.60.109"
    old_node_ip = "172.31.50.57"
    new_node_ip = "172.31.26.43"

    node_layer_mapping = [
        (head_node_ip, 16),
        (old_node_ip, 16),
    ]
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bucket_name = "hetero-spot-llm-serve-models"
    config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "pp_layer_partition": "16,16",  # Pipeline partition: 16 layers per node
        "parallel_strategy": [1,1],     # 1 GPU per node (no tensor parallelism)
        "max_model_len": 4096,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 16,
        "model_source": "s3",
        "s3_path": f"s3://{bucket_name}/{model_name}",
    }
    dummy_throughput = 100

    # Create pipeline asynchronously in a separate thread
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
        test_logger.info("Pipeline creation completed")
    
    # Start pipeline creation task
    pipeline_task = asyncio.create_task(create_pipeline_async())
    
    # Start the global server in the background
    server_task = asyncio.create_task(global_server.run_global_server())

    # Schedule node switch after 2 minutes
    async def switch_node_after_delay():
        """Switch node after 2 minute delay"""
        await asyncio.sleep(120)  # Wait 2 minutes
        try:
            test_logger.info(f"Starting node switch test: {old_node_ip} -> {new_node_ip}")
            
            # Execute switch in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    global_server.switch_node,
                    old_node_ip,
                    new_node_ip
                )
            test_logger.info("Node switch test completed")
        except Exception as e:
            test_logger.error(f"Node switch test failed: {e}")
    
    # Start node switch task
    switch_task = asyncio.create_task(switch_node_after_delay())

    # Generate and send requests
    # Generate test requests
    num_requests = 100
    request_inputs = generate_random_requests(
        num_prompts=num_requests,
        input_len=1024,
        output_len=1024,
        model_name=model_name,
        ignore_eos=True  # Ignore EOS to ensure consistent output length
    )
    
    try:
        # Send and monitor requests concurrently
        completed_count = await send_and_monitor_requests(
            global_server,
            request_inputs,
            delay_between_requests=10,
            logger=test_logger,
            timeout=600,
            status_interval=60,
            status_callback=lambda: log_pipeline_status(global_server, test_logger)
        )
        
        # Final statistics
        test_logger.info(f"\nFinal Results: {completed_count}/{num_requests} requests completed successfully")
        
    except KeyboardInterrupt:
        test_logger.info("Shutting down...")
    finally:
        # Cancel all tasks
        test_logger.info("Cancelling background tasks...")
        server_task.cancel()
        switch_task.cancel()
        pipeline_task.cancel()
        
        # Give tasks a chance to cancel gracefully
        await asyncio.sleep(1)
        
        # Force cleanup after short timeout
        try:
            done, pending = await asyncio.wait(
                [server_task, switch_task, pipeline_task], 
                timeout=2.0, 
                return_when=asyncio.ALL_COMPLETED
            )
            if pending:
                test_logger.info(f"Force cancelling {len(pending)} remaining tasks")
                for task in pending:
                    task.cancel()
        except Exception as e:
            test_logger.error(f"Error during task cleanup: {e}")
        
        # Stop all pipelines
        test_logger.info("Stopping pipelines...")
        try:
            global_server.cluster.stop_all_pipelines()
            test_logger.info("All pipelines stopped")
        except Exception as e:
            test_logger.error(f"Error stopping pipelines: {e}")
        
        test_logger.info("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(test_node_switch())