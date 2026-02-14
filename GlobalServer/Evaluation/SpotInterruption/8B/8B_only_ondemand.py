"""
Spot interruption benchmark using on-demand instances with real Azure trace.
"""
import asyncio
import logging
import concurrent.futures
import sys
import os
import time
from typing import Dict, List, Tuple
from nodes_8B import *

# Add GlobalServer to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_trace_benchmark, DEFAULT_DATASET_PATH

logger = logging.getLogger(__name__)


async def run_benchmark(
    global_server: GlobalServer,
    dataset_path: str,
    num_requests: int = None,
    time_scale: float = 1.0,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    percentiles: List[float] = None,
    disable_tqdm: bool = False,
    run_initial_test: bool = True,
    test_requests_per_pipeline: int = 2,
    start_time: float = None,
    end_time: float = None
):
    return await run_trace_benchmark(
        global_server=global_server,
        dataset_path=dataset_path,
        trace_output_prefix="spotinterruption_8b_only_ondemand",
        num_requests=num_requests,
        time_scale=time_scale,
        model_name=model_name,
        percentiles=percentiles,
        disable_tqdm=disable_tqdm,
        run_initial_test=run_initial_test,
        test_requests_per_pipeline=test_requests_per_pipeline,
        start_time=start_time,
        end_time=end_time,
    )


async def main():
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
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tasks = []

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
    pipeline_1_stage_0_node_ip = spot_g6_xlarge_node_ip_1
    pipeline_1_stage_1_node_ip = spot_g6_xlarge_node_ip_2
    pipeline_1_config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "16,16",
        "parallel_strategy": [1,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 10074,
        "max_batch_size": 162,
    }
    estimated_throughput_1 = 5.73
    node_layer_mapping_1 = [
        (pipeline_1_stage_0_node_ip, 16),
        (pipeline_1_stage_1_node_ip, 16),
    ]

    # Pipeline 2
    pipeline_2_stage_0_node_ip = spot_g6_xlarge_node_ip_3
    pipeline_2_config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "32",
        "parallel_strategy": [1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 1181,
        "max_batch_size": 19,
    }
    estimated_throughput_2 = 1.25
    node_layer_mapping_2 = [
        (pipeline_2_stage_0_node_ip, 32),
    ]
    
    # Start pipeline creation
    pipeline_task_1 = asyncio.create_task(create_pipeline_async(pipeline_1_config, node_layer_mapping_1, estimated_throughput_1))
    pipeline_task_2 = asyncio.create_task(create_pipeline_async(pipeline_2_config, node_layer_mapping_2, estimated_throughput_2))
    await asyncio.gather(pipeline_task_1, pipeline_task_2)
    tasks.append(pipeline_task_1)
    tasks.append(pipeline_task_2)
    
    # Start the global server in the background
    server_task = asyncio.create_task(global_server.run_global_server())
    tasks.append(server_task)

    try:
        # Set dataset path
        dataset_path = DEFAULT_DATASET_PATH

        start_time = 0      # Start from beginning
        end_time = 15 * 60  # Run for 15 minutes

        # Run benchmark using helper function
        metrics = await run_benchmark(
            global_server,
            dataset_path=dataset_path,
            num_requests=None,  # Use all requests from trace
            time_scale=0,  # Original trace speed (0.0 = Offline, 1.0 = original speed)
            model_name=model_name,
            percentiles=[10, 25, 50, 75, 90, 95],
            disable_tqdm=False,  # Show progress bar
            run_initial_test=True,
            test_requests_per_pipeline=1,
            start_time=start_time,
            end_time=end_time
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
    asyncio.run(main())