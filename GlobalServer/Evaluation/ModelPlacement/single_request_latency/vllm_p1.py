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

# Add GlobalServer to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_latency_benchmark

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
    return await run_latency_benchmark(
        global_server=global_server,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        request_rate=request_rate,
        model_name=model_name,
        max_concurrency=max_concurrency,
        percentiles=percentiles,
        disable_tqdm=disable_tqdm,
        run_initial_test=run_initial_test,
        test_requests_per_pipeline=test_requests_per_pipeline,
    )


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

    # Homogeneous Pipeline 1
    pipeline_1_stage_0_node_ip = g6_12xlarge_node_ip_1
    pipeline_1_stage_1_node_ip = g6_12xlarge_node_ip_2
    pipeline_1_stage_2_node_ip = g6_12xlarge_node_ip_3
    pipeline_1_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "26,27,27",
        "parallel_strategy": [4,4,4],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 15360,
        "max_batch_size": 247,
    }
    estimated_throughput_1 = 2.80
    node_layer_mapping_1 = [
        (pipeline_1_stage_0_node_ip, 26),
        (pipeline_1_stage_1_node_ip, 27),
        (pipeline_1_stage_2_node_ip, 27),
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
            num_requests=10,  # Small number of requests for latency measurement
            input_len=763,
            output_len=232,
            request_rate=float('inf'),  # No rate limit
            model_name=model_name,
            max_concurrency=1,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,  # Show progress bars
            run_initial_test=True,  # Run test requests first
            test_requests_per_pipeline=0  # 0 test requests per pipeline
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
