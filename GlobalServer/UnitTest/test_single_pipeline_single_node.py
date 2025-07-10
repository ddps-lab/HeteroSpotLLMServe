"""
Unit test for single pipeline with multiple nodes (Pipeline Parallelism).
Tests creating one pipeline spanning across two nodes with layer partitioning.
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

async def test_single_pipeline_single_node():
    """Test single pipeline with multiple nodes using pipeline parallelism."""
    # Configure logging
    test_logger = setup_test_logger(__name__)
    
    global_server = GlobalServer()

    # Configuration for single pipeline across two nodes
    node_ip_1 = "172.31.60.109"
    
    # Pipeline Parallelism: 16 layers on each node (total 32 layers)
    node_layer_mapping = [
        (node_ip_1, 32)
    ]
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bucket_name = "hetero-spot-llm-serve-models"
    config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "pp_layer_partition": "32",  # Pipeline partition: 16 layers per node
        "parallel_strategy": [1],     # 1 GPU per node (no tensor parallelism)
        "max_model_len": 4096,
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
        test_logger.info("Single-node pipeline creation completed")
    
    # Start pipeline creation task
    pipeline_task = asyncio.create_task(create_pipeline_async())
    
    # Start the global server in the background
    server_task = asyncio.create_task(global_server.run_global_server())

    # Generate and send requests
    num_requests = 100  # Fewer requests for multi-node testing
    request_inputs = generate_random_requests(
        num_prompts=num_requests,
        input_len=1024,   # Shorter input for faster processing
        output_len=1024,   # Shorter output for faster processing
        model_name=model_name,
        ignore_eos=True  # Ignore EOS to ensure consistent output length
    )
    
    try:
        # Send and monitor requests concurrently
        completed_count = await send_and_monitor_requests(
            global_server,
            request_inputs,
            delay_between_requests=5,
            logger=test_logger,
            timeout=480,
            status_interval=30,
            status_callback=lambda: log_pipeline_status(global_server, test_logger)
        )
        
        # Final statistics
        test_logger.info(f"\nFinal Results: {completed_count}/{num_requests} requests completed successfully")
        test_logger.info(f"Single-node pipeline test completed")
        
    except KeyboardInterrupt:
        test_logger.info("Shutting down...")
    finally:
        # Cancel all tasks
        test_logger.info("Cancelling background tasks...")
        server_task.cancel()
        pipeline_task.cancel()
        
        # Give tasks a chance to cancel gracefully
        await asyncio.sleep(1)
        
        # Force cleanup after short timeout
        try:
            done, pending = await asyncio.wait(
                [server_task, pipeline_task], 
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
    asyncio.run(test_single_pipeline_single_node())
