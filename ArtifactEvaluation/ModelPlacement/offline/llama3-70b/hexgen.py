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
import time

# Add GlobalServer to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_trace_benchmark, DEFAULT_DATASET_PATH

from nodes import *

S3_BUCKET = "hetero-spot-llm-serve-models"


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
    return await run_trace_benchmark(
        global_server=global_server,
        dataset_path=dataset_path,
        trace_output_prefix="modelplacement_offline_hexgen",
        num_requests=num_requests,
        time_scale=time_scale,
        model_name=model_name,
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

    # Hexgen Pipeline 1
    pipeline_1_stage_0_node_ip = g6e_xlarge_node_ip_1
    pipeline_1_stage_1_node_ip = g6e_xlarge_node_ip_2
    pipeline_1_stage_2_node_ip = g6e_xlarge_node_ip_3
    pipeline_1_stage_3_node_ip = g5_12xlarge_node_ip_1
    pipeline_1_stage_4_node_ip = g5_12xlarge_node_ip_1
    pipeline_1_stage_5_node_ip = g5_12xlarge_node_ip_1
    pipeline_1_stage_6_node_ip = g5_12xlarge_node_ip_2
    pipeline_1_stage_7_node_ip = g5_12xlarge_node_ip_2
    pipeline_1_stage_8_node_ip = g5_12xlarge_node_ip_2
    pipeline_1_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "12,13,13,13,6,6,6,6,5",
        "parallel_strategy": [1,1,1,2,1,1,1,1,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://{S3_BUCKET}/{model_name}",
        "num_gpu_blocks": 17661,
        "max_batch_size": 283,
        "mode": "hexgen",
    }
    estimated_throughput_1 = 3.22
    # {(ip, layers, start_local_rank), ...} or {(ip, layers), ...}
    node_layer_mapping_1 = [
        (pipeline_1_stage_0_node_ip, 12),
        (pipeline_1_stage_1_node_ip, 13),
        (pipeline_1_stage_2_node_ip, 13),
        (pipeline_1_stage_3_node_ip, 13),
        (pipeline_1_stage_4_node_ip, 6),
        (pipeline_1_stage_5_node_ip, 6),
        (pipeline_1_stage_6_node_ip, 6),
        (pipeline_1_stage_7_node_ip, 6),
        (pipeline_1_stage_8_node_ip, 5),
    ]
        
    # Pipeline 2
    pipeline_2_stage_0_node_ip = g6e_xlarge_node_ip_4
    pipeline_2_stage_1_node_ip = g6_12xlarge_node_ip_1
    pipeline_2_stage_2_node_ip = g6_12xlarge_node_ip_1
    pipeline_2_stage_3_node_ip = g6_12xlarge_node_ip_2
    pipeline_2_stage_4_node_ip = g6_12xlarge_node_ip_2
    pipeline_2_stage_5_node_ip = g6_12xlarge_node_ip_3
    pipeline_2_stage_6_node_ip = g6_12xlarge_node_ip_3
    pipeline_2_stage_7_node_ip = "172.31.34.108"
    pipeline_2_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "13,9,11,11,11,11,11,3",
        "parallel_strategy": [1,2,2,2,2,2,2,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://{S3_BUCKET}/{model_name}",
        "num_gpu_blocks": 15173,
        "max_batch_size": 243,
        "mode": "hexgen",
    }
    estimated_throughput_2 = 2.797
    node_layer_mapping_2 = [
        (pipeline_2_stage_0_node_ip, 13),
        (pipeline_2_stage_1_node_ip, 9),
        (pipeline_2_stage_2_node_ip, 11),
        (pipeline_2_stage_3_node_ip, 11),
        (pipeline_2_stage_4_node_ip, 11),
        (pipeline_2_stage_5_node_ip, 11),
        (pipeline_2_stage_6_node_ip, 11),
        (pipeline_2_stage_7_node_ip, 3),
    ]
    
    # Start pipeline creation
    pipeline_task_1 = asyncio.create_task(create_pipeline_async(pipeline_1_config, node_layer_mapping_1, estimated_throughput_1))
    pipeline_task_2 = asyncio.create_task(create_pipeline_async(pipeline_2_config, node_layer_mapping_2, estimated_throughput_2))

    # Start global server
    server_task = asyncio.create_task(global_server.run_global_server())
    
    try:
        # Wait for pipeline creation to complete
        logger.info("Waiting for pipeline creation to complete...")
        await pipeline_task_1
        await pipeline_task_2
        logger.info("Pipelines are ready!")

        # Run benchmark
        dataset_path = DEFAULT_DATASET_PATH
        metrics = await run_benchmark(
            global_server,
            dataset_path=dataset_path,
            num_requests=None,
            time_scale=0.0,  # Original trace speed. If you set 0.0 then it will behave like offline setting
            model_name=model_name,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,  # Show progress bars
            run_initial_test=True,  # Run test requests first
            test_requests_per_pipeline=1  # 1 test requests per pipeline
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