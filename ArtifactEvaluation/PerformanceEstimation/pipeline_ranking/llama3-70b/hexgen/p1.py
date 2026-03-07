"""
HexGen Pipeline 1: g6e.xlarge×3 + g5.12xlarge×2
PP=9 (12,13,13,13,6,6,6,6,5), TP=[1,1,1,2,1,1,1,1,1]
mode=hexgen
Synthetic fixed-length requests (input=763, output=232)
"""
import asyncio
import concurrent.futures
import logging
import sys
import os
from typing import Dict, List, Tuple

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_latency_benchmark
from save_results import save_benchmark_results

from nodes import *

S3_BUCKET = "hetero-spot-llm-serve-models"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "hexgen_p1.json")


async def test_benchmark():
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

    global_server = GlobalServer()

    async def create_pipeline_async(config, node_layer_mapping, throughput):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, global_server.create_pipeline,
                node_layer_mapping, config, throughput
            )
        logger.info("Pipeline creation completed")

    config = {
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
    node_layer_mapping = [
        (g6e_xlarge_node_ip_1, 12),
        (g6e_xlarge_node_ip_2, 13),
        (g6e_xlarge_node_ip_3, 13),
        (g5_12xlarge_node_ip_1, 13),
        (g5_12xlarge_node_ip_1, 6),
        (g5_12xlarge_node_ip_1, 6),
        (g5_12xlarge_node_ip_2, 6),
        (g5_12xlarge_node_ip_2, 6),
        (g5_12xlarge_node_ip_2, 5),
    ]
    estimated_throughput = 3.22

    pipeline_task = asyncio.create_task(
        create_pipeline_async(config, node_layer_mapping, estimated_throughput)
    )
    server_task = asyncio.create_task(global_server.run_global_server())

    try:
        logger.info("Waiting for pipeline creation to complete...")
        await pipeline_task
        logger.info("Pipeline is ready!")

        metrics = await run_latency_benchmark(
            global_server=global_server,
            num_requests=2830,  # max_batch_size(283) × 10
            input_len=763,
            output_len=232,
            request_rate=float('inf'),
            model_name=model_name,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,
            run_initial_test=True,
            test_requests_per_pipeline=1,
        )

        print_benchmark_results(metrics)
        save_benchmark_results(metrics, OUTPUT_PATH, extra={
            "system": "HexGen",
            "pipeline": "P1",
            "pp_layer_partition": "12,13,13,13,6,6,6,6,5",
            "parallel_strategy": [1,1,1,2,1,1,1,1,1],
            "instances": ["g6e.xlarge×3", "g5.12xlarge×2"],
            "mode": "hexgen",
            "input_len": 763,
            "output_len": 232,
            "num_requests": 2830,
        })

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
    finally:
        logger.info("Cleaning up...")
        server_task.cancel()
        pipeline_task.cancel()
        try:
            await asyncio.gather(server_task, return_exceptions=True)
        except:
            pass
        logger.info("Stopping pipelines...")
        try:
            global_server.cluster.stop_all_pipelines()
            logger.info("All pipelines stopped")
        except Exception as e:
            logger.error(f"Error stopping pipelines: {e}")


if __name__ == "__main__":
    asyncio.run(test_benchmark())
