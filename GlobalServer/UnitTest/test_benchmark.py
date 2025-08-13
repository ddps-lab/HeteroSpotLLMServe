"""
Benchmark test for GlobalServer that measures throughput and latency metrics.
Similar to benchmark_serving.py but using GlobalServer's internal add_request.
"""
import asyncio
import concurrent.futures
import sys
import os
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_server import GlobalServer
from request_handler import generate_random_requests
from test_utils import setup_test_logger
from benchmark_utils import (
    calculate_benchmark_metrics, 
    print_benchmark_results,
    run_benchmark_requests
)


async def run_benchmark(
    global_server: GlobalServer,
    num_requests: int = 100,
    input_len: int = 1024,
    output_len: int = 128,
    request_rate: float = float('inf'),
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_concurrency: int = None,
    percentiles: List[float] = None,
    disable_tqdm: bool = False
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
    logger = setup_test_logger(__name__)
    
    # Create GlobalServer instance
    global_server = GlobalServer()
    
    # Configuration for single node
    node_ip = "172.31.62.65"  # Update with your actual node IP
    node_layer_mapping = [(node_ip, 32)]  # Single node with all 32 layers
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "pp_layer_partition": "32",
        "parallel_strategy": [1],  # Single GPU
        "max_model_len": 4096,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 256,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
    }
    dummy_throughput = 100
    
    # Create pipeline in background
    async def create_pipeline_async():
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                global_server.create_pipeline,
                node_layer_mapping,
                config,
                dummy_throughput
            )
        logger.info("Pipeline creation completed")
    
    # Start pipeline creation
    pipeline_task = asyncio.create_task(create_pipeline_async())
    
    # Start global server
    server_task = asyncio.create_task(global_server.run_global_server())
    
    try:
        # Wait for pipeline creation to complete
        logger.info("Waiting for pipeline creation to complete...")
        await pipeline_task
        logger.info("Pipeline is ready!")
        
        # Run benchmark
        metrics = await run_benchmark(
            global_server,
            num_requests=1024,  # Matching the example output
            input_len=1024,
            output_len=128,
            request_rate=float('inf'),  # No rate limit for max throughput test
            model_name=model_name,
            max_concurrency=None,  # Limit concurrent requests
            percentiles=[25, 50, 75, 99],  # Custom percentiles
            disable_tqdm=False  # Show progress bars
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
        pipeline_task.cancel()
        
        try:
            await asyncio.gather(server_task, pipeline_task, return_exceptions=True)
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