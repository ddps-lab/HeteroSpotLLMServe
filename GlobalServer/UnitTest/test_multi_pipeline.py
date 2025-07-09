"""
Unit test for multiple pipeline management in GlobalServer.
Tests creating two pipelines and removing one.
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

async def test_multi_pipeline():
    """Test multiple pipeline creation and removal."""
    # Configure logging
    test_logger = setup_test_logger(__name__)
    
    global_server = GlobalServer()

    # Configuration for two pipelines
    node_ip_1 = "172.31.53.208"
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
    pipeline_task_2 = asyncio.create_task(create_pipeline_async(node_ip_2, 900, "Pipeline 2"))
    
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
        ignore_eos=True  # Ignore EOS to ensure consistent output length
    )
    
    try:
        # Send and monitor requests concurrently
        completed_count = await send_and_monitor_requests(
            global_server,
            request_inputs,
            delay_between_requests=3,
            logger=test_logger,
            timeout=480,
            status_interval=30,
            status_callback=lambda: log_pipeline_status(global_server, test_logger)
        )
        
        # Final statistics
        test_logger.info(f"\nFinal Results: {completed_count}/{num_requests} requests completed successfully")
        test_logger.info(f"Final pipeline count: {len(global_server.cluster.pipelines)}")
        
    except KeyboardInterrupt:
        test_logger.info("Shutting down...")
    finally:
        # Cancel all tasks
        test_logger.info("Cancelling background tasks...")
        server_task.cancel()
        removal_task.cancel()
        pipeline_task_1.cancel()
        pipeline_task_2.cancel()
        
        # Give tasks a chance to cancel gracefully
        await asyncio.sleep(0.1)
        
        # Force cleanup after short timeout
        try:
            done, pending = await asyncio.wait(
                [server_task, removal_task, pipeline_task_1, pipeline_task_2], 
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
    asyncio.run(test_multi_pipeline())