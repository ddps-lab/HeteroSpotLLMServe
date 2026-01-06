"""
Unit test for node switching functionality in GlobalServer.
"""
import asyncio
import logging
import concurrent.futures
import sys
import os
import time
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_server import GlobalServer
from request_handler import generate_random_requests
from benchmark_utils import (
    calculate_benchmark_metrics,
    print_benchmark_results,
    run_benchmark_requests
)

logger = logging.getLogger(__name__)


async def run_benchmark(
    global_server,
    num_requests: int = 100,
    input_len: int = 1024,
    output_len: int = 128,
    request_rate: float = float('inf'),
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_concurrency: int = None,
    percentiles: List[float] = None,
    disable_tqdm: bool = False,
    run_initial_test: bool = True,
    test_requests_per_pipeline: int = 2
):
    """
    Run a benchmark test on the GlobalServer.

    Args:
        global_server: The GlobalServer instance
        num_requests: Number of requests to send
        input_len: Input token length
        output_len: Expected output token length
        request_rate: Requests per second (inf for no limit)
        model_name: Model name for generating requests
        max_concurrency: Maximum number of concurrent requests (None for no limit)
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 99])
        disable_tqdm: Whether to disable progress bar
        run_initial_test: Whether to run initial test requests
        test_requests_per_pipeline: Number of test requests per pipeline

    Returns:
        BenchmarkMetrics object with results
    """
    if not disable_tqdm:
        print("\n" + "=" * 50)
        print("Starting GlobalServer Benchmark")
        print(f"  Requests: {num_requests}")
        print(f"  Input length: {input_len} tokens")
        print(f"  Output length: {output_len} tokens")
        print(f"  Request rate: {request_rate if request_rate != float('inf') else 'unlimited'} req/s")
        print(f"  Model: {model_name}")
        print("=" * 50 + "\n")

    # Generate random requests
    if not disable_tqdm:
        print("Generating requests...")
    request_inputs = generate_random_requests(
        num_prompts=num_requests,
        input_len=input_len,
        output_len=output_len,
        model_name=model_name,
        ignore_eos=True  # Ensure consistent output length
    )

    # Run initial test if requested
    if run_initial_test:
        num_pipelines = len(global_server.cluster.pipelines)
        print(f"\nRunning initial test on {num_pipelines} pipeline(s)...")

        # Generate test requests (2 per pipeline)
        test_count = test_requests_per_pipeline * num_pipelines
        test_inputs = request_inputs[:test_count] if test_count <= len(request_inputs) else generate_random_requests(
            num_prompts=test_count,
            input_len=input_len,
            output_len=output_len,
            model_name=model_name,
            ignore_eos=True
        )

        # Send test requests
        print(f"Sending {test_count} test requests ({test_requests_per_pipeline} per pipeline)...")
        test_requests = []
        for test_input in test_inputs:
            request = await global_server.add_request_and_wait(test_input)
            test_requests.append(request)

        # Check results
        failed_count = 0
        for i, request in enumerate(test_requests):
            if not (request.output and request.output.success):
                failed_count += 1
                error_msg = request.output.error if request.output else "No output"
                print(f"  Test request {i+1} failed: {error_msg}")

        if failed_count > 0:
            raise ValueError(
                f"Initial test failed - {failed_count}/{test_count} requests failed. "
                "Please check pipeline configuration and server status."
            )
        else:
            print(f"Initial test completed successfully - all {test_count} requests succeeded!")
            print("Starting main benchmark run...\n")

    # Run benchmark (send requests and wait for completion)
    requests, actual_duration = await run_benchmark_requests(
        global_server, request_inputs, request_rate, max_concurrency, disable_tqdm
    )

    # Calculate metrics
    if not disable_tqdm:
        print("\nCalculating metrics...")
    metrics = calculate_benchmark_metrics(
        requests, request_inputs, actual_duration, percentiles
    )

    return metrics


async def test_node_switch():
    """Test node switching functionality."""
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    global_server = GlobalServer()

    # Node IPs
    old_node_ip_1    = "172.31.6.220"
    old_node_ip_2    = "172.31.8.77"
    new_node_ip_1    = "172.31.6.180"
    new_node_ip_2    = "172.31.15.39"

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bucket_name = "hetero-spot-llm-serve-models"
    pipeline_config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "16,16",
        "parallel_strategy": [1,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://{bucket_name}/{model_name}",
        "num_gpu_blocks": 10074,
        "max_batch_size": 162,
    }
    node_layer_mapping = [
        (old_node_ip_1, 16),
        (old_node_ip_2, 16),
    ]
    pipeline_throughput = 5.73

    # Create pipeline asynchronously in a separate thread
    async def create_pipeline_async(node_layer_mapping, config, throughput):
        """Create pipeline in a separate thread to avoid blocking"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                global_server.create_pipeline,
                node_layer_mapping,
                config,
                throughput
            )
        logger.info("Pipeline creation completed")

    # Start pipeline creation task
    pipeline_task = asyncio.create_task(create_pipeline_async(node_layer_mapping, pipeline_config, pipeline_throughput))
    
    # Start the global server in the background
    server_task = asyncio.create_task(global_server.run_global_server())

    # Schedule node switch (optional - comment out to disable)
    async def switch_node_after_delay():
        """Switch node after 120 seconds"""
        await asyncio.sleep(120)  # Wait 2 minutes
        try:
            logger.info(f"Starting node switch test: {old_node_ip_1}, {old_node_ip_2} -> {new_node_ip_1}, {new_node_ip_2}")

            # Time measurement start
            switch_start_time = time.time()

            # Execute switch in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    global_server.switch_nodes,
                    [old_node_ip_1, old_node_ip_2],
                    [new_node_ip_1, new_node_ip_2]
                )

            # Time measurement end and logging
            switch_end_time = time.time()
            switch_duration = switch_end_time - switch_start_time

            logger.info("Node switch test completed")
            logger.info(f"⏱️  Node Switch Duration: {switch_duration:.2f} seconds")

        except Exception as e:
            logger.error(f"Node switch test failed: {e}")

    switch_task = asyncio.create_task(switch_node_after_delay())
    logger.info("Node switch scheduled after 120 seconds")

    try:
        # Run benchmark using helper function
        metrics = await run_benchmark(
            global_server,
            num_requests=1000,
            input_len=763,
            output_len=232,
            request_rate=float('inf'),
            model_name=model_name,
            max_concurrency=None,  # No concurrency limit
            percentiles=[10, 25, 50, 75, 90, 95],
            disable_tqdm=False,  # Show progress bar
            run_initial_test=True,  # Skip initial test (pipeline already created)
            test_requests_per_pipeline=0
        )

        # Print results
        print_benchmark_results(metrics)
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")

        # Collect all background tasks
        tasks = []
        tasks.append(server_task)
        tasks.append(pipeline_task)
        tasks.append(switch_task)  # Comment out this line to disable node switch

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except:
            pass

        # Stop all pipelines
        logger.info("Stopping pipelines...")
        try:
            global_server.cluster.stop_all_pipelines()
            logger.info("All pipelines stopped")
        except Exception as e:
            logger.error(f"Error stopping pipelines: {e}")

if __name__ == "__main__":
    asyncio.run(test_node_switch())