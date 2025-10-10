"""
Benchmark test for GlobalServer that measures throughput and latency metrics.
Similar to benchmark_serving.py but using GlobalServer's internal add_request.
"""
import asyncio
import concurrent.futures
import logging
import sys
import os
from typing import Dict, List, Tuple

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from global_server import GlobalServer
from request_handler import generate_random_requests
from benchmark_utils import print_benchmark_results
from evaluation_utils import (
    load_azure_trace,
    generate_requests_from_trace,
    run_trace_replay_benchmark
)

from nodes import *


async def run_benchmark(
    global_server: GlobalServer,
    dataset_path: str,
    num_requests: int = None,
    time_scale: float = 1.0,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    percentiles: List[float] = None,
    disable_tqdm: bool = False,
    run_initial_test: bool = True,
    test_requests_per_pipeline: int = 2
):
    """
    Run a trace-based benchmark test on the GlobalServer using Azure dataset.

    Args:
        global_server: The GlobalServer instance
        dataset_path: Path to Azure trace dataset CSV file
        num_requests: Maximum number of requests to load from dataset (None for all)
        time_scale: Time scale multiplier (1.0 = original, 0.5 = 2x faster, 2.0 = 2x slower)
        model_name: Model name for generating requests
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 99])
        disable_tqdm: Whether to disable progress bar
        run_initial_test: Whether to run initial test requests
        test_requests_per_pipeline: Number of test requests per pipeline

    Returns:
        BenchmarkMetrics object with results
    """
    # Run initial test if requested
    if run_initial_test:
        num_pipelines = len(global_server.cluster.pipelines)
        print(f"\nRunning initial test on {num_pipelines} pipeline(s)...")

        # Generate test requests with fixed length for stability
        test_count = test_requests_per_pipeline * num_pipelines
        test_inputs = generate_random_requests(
            num_prompts=test_count,
            input_len=512,
            output_len=128,
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
            print("Starting trace benchmark...\n")

    # Load Azure trace dataset
    print(f"Loading trace dataset from: {dataset_path}")
    trace_data = load_azure_trace(
        csv_path=dataset_path,
        max_requests=num_requests,
        max_context_tokens=2048,  # Filter out very long contexts
        max_generated_tokens=None
    )

    if not trace_data:
        raise ValueError("No trace data loaded. Check dataset path and filters.")

    # Generate requests from trace
    print("Generating requests from trace data...")
    trace_requests = generate_requests_from_trace(
        trace_data=trace_data,
        model_name=model_name,
        seed=0,
        ignore_eos=True
    )

    print(f"Generated {len(trace_requests)} requests from trace\n")

    # Run trace replay benchmark
    metrics = await run_trace_replay_benchmark(
        global_server=global_server,
        trace_requests=trace_requests,
        time_scale=time_scale,
        percentiles=percentiles,
        disable_tqdm=disable_tqdm
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
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    
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

    # Hexgen Pipeline 1
    pipeline_1_stage_0_node_ip = g6_12xlarge_node_ip_1
    pipeline_1_stage_1_node_ip = g6_12xlarge_node_ip_2
    pipeline_1_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.9,
        "pp_layer_partition": "41,39",
        "parallel_strategy": [4,4],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 2860,
        "max_batch_size": 46,
    }
    estimated_throughput_1 = 1.08
    node_layer_mapping_1 = [
        (pipeline_1_stage_0_node_ip, 41),
        (pipeline_1_stage_1_node_ip, 39),
    ]

    # Pipeline 2
    pipeline_2_stage_0_node_ip = g5_12xlarge_node_ip_1
    pipeline_2_stage_1_node_ip = g5_12xlarge_node_ip_2
    pipeline_2_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.9,
        "pp_layer_partition": "41,39",
        "parallel_strategy": [4,4],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 2860,
        "max_batch_size": 46,
    }
    estimated_throughput_2 = 1.62
    node_layer_mapping_2 = [
        (pipeline_2_stage_0_node_ip, 41),
        (pipeline_2_stage_1_node_ip, 39),
    ]

    # Pipeline 2
    pipeline_3_stage_0_node_ip = g6e_xlarge_node_ip_1
    pipeline_3_stage_1_node_ip = g6e_xlarge_node_ip_2
    pipeline_3_stage_2_node_ip = g6e_xlarge_node_ip_3
    pipeline_3_stage_3_node_ip = g6e_xlarge_node_ip_4
    pipeline_3_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.9,
        "pp_layer_partition": "20,20,21,19",
        "parallel_strategy": [1,1,1,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 2425,
        "max_batch_size": 39,
    }
    estimated_throughput_3 = 0.90
    node_layer_mapping_3 = [
        (pipeline_3_stage_0_node_ip, 20),
        (pipeline_3_stage_1_node_ip, 20),
        (pipeline_3_stage_2_node_ip, 21),
        (pipeline_3_stage_3_node_ip, 19),
    ]
    
    # Start pipeline creation
    pipeline_task_1 = asyncio.create_task(create_pipeline_async(pipeline_1_config, node_layer_mapping_1, estimated_throughput_1))
    pipeline_task_2 = asyncio.create_task(create_pipeline_async(pipeline_2_config, node_layer_mapping_2, estimated_throughput_2))
    pipeline_task_3 = asyncio.create_task(create_pipeline_async(pipeline_3_config, node_layer_mapping_3, estimated_throughput_3))

    # Start global server
    server_task = asyncio.create_task(global_server.run_global_server())
    
    try:
        # Wait for pipeline creation to complete
        logger.info("Waiting for pipeline creation to complete...")
        await pipeline_task_1
        await pipeline_task_2
        await pipeline_task_3
        logger.info("Pipelines are ready!")

        # Run benchmark
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "Datasets",
            "AzureLLMInferenceConvTrace_pruned_2048.csv"
        )
        metrics = await run_benchmark(
            global_server,
            dataset_path=dataset_path,
            num_requests=5000,  # Load first 5000 requests from dataset
            time_scale=1.0,  # Original trace speed
            model_name=model_name,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,  # Show progress bars
            run_initial_test=True,  # Run test requests first
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
        pipeline_task_2.cancel()
        pipeline_task_3.cancel()

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