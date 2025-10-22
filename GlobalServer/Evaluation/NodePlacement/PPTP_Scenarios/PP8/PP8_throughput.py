"""
Benchmark test for GlobalServer that measures single request latency.
Uses fixed-length synthetic requests instead of trace data.
"""
import asyncio
import concurrent.futures
import logging
import sys
import os
from typing import Dict, List, Tuple

from nodes import *

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(parent_dir)

from global_server import GlobalServer
from request_handler import generate_random_requests
from benchmark_utils import (
    calculate_benchmark_metrics,
    print_benchmark_results,
    run_benchmark_requests
)

from nodes import *


async def run_benchmark(
    global_server: GlobalServer,
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


async def test_benchmark():
    """Test benchmark with a single node configuration."""
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
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    # Create GlobalServer instance
    global_server = GlobalServer()

    # Create pipeline in background
    async def create_pipeline_async(config:Dict, node_layer_mapping:List[Tuple[str, int]], throughput:int):
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

    # Our Pipeline 1
    pipeline_1_config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "gpu_memory_utilization": 0.9,
        "pp_layer_partition": "4,4,4,4,4,4,4,4",
        "parallel_strategy": [1,1,1,1,1,1,1,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 2048,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 60480,
        "max_batch_size": 1512,
    }
    estimated_throughput_1 = 44.44
    node_layer_mapping_1 = [
        (node1_ip, 4),
        (node2_ip, 4),
        (node3_ip, 4),
        (node4_ip, 4),
        (node5_ip, 4),
        (node6_ip, 4),
        (node7_ip, 4),
        (node8_ip, 4),
    ]

    # Start pipeline creation
    pipeline_task_1 = asyncio.create_task(create_pipeline_async(pipeline_1_config, node_layer_mapping_1, estimated_throughput_1))

    # Start global server
    server_task = asyncio.create_task(global_server.run_global_server())

    try:
        # Wait for pipeline creation to complete
        logger.info("Waiting for pipeline creation to complete...")
        await pipeline_task_1
        logger.info("Pipelines are ready!")

        # Run benchmark - optimized for single request latency measurement
        metrics = await run_benchmark(
            global_server,
            num_requests=1512*3,  # Small number of requests for latency measurement
            input_len=763,
            output_len=232,
            request_rate=float('inf'),  # No rate limit
            model_name=model_name,
            max_concurrency=None, 
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,  # Show progress bars
            run_initial_test=False,  # Run test requests first
            test_requests_per_pipeline=2  # 2 test requests per pipeline
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
        server_task.cancel()
        pipeline_task_1.cancel()

        try:
            await asyncio.gather(server_task, return_exceptions=True)
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
    asyncio.run(test_benchmark())
