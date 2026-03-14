"""
Example Pipeline — Qwen3-4B on 1× g6.xlarge (single L4 GPU)

Unit test: verifies logging, benchmark metrics, and JSON result output
without running the optimizer. Small model fits entirely on one GPU.
"""
import asyncio
import concurrent.futures
import json
import logging
import sys
import os
import argparse

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_REPO_ROOT = _d
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_latency_benchmark
from save_results import save_benchmark_results

from nodes import *

# ─── Manual config (no optimizer) ────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B"
TOTAL_LAYERS = 36
INPUT_LEN = 512
OUTPUT_LEN = 64
S3_BUCKET = "hetero-spot-llm-serve-models"

# Estimator results for g6.xlarge (1× L4, 22494 MB):
#   max_batch_size = 117, num_blocks = 4212
#   throughput = 8.72 req/s, latency = 13421.60 ms
MAX_BATCH_SIZE = 117
NUM_GPU_BLOCKS = 4212
NUM_REQUESTS = MAX_BATCH_SIZE * 5

OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "ArtifactEvaluation", "ModelPlacement",
    "optimizer", "results", "qwen3-32b", "measured"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "example_Qwen3-4B.json")

# Single GPU — all layers on one node
NODE_LAYER_MAPPING = [
    (g6_xlarge_node_ip_2, TOTAL_LAYERS),  # g6.xlarge: 1× L4 GPU
]


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

    print("=" * 70)
    print(f"Example Pipeline — Unit Test")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layers: {TOTAL_LAYERS}")
    print(f"  Instance: g6.xlarge (1× L4 GPU)")
    print(f"  Input/Output: {INPUT_LEN}/{OUTPUT_LEN} tokens")
    print(f"  Requests: {NUM_REQUESTS}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 70)

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
        "model_name": MODEL_NAME,
        "total_num_layers": TOTAL_LAYERS,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": str(TOTAL_LAYERS),
        "parallel_strategy": [1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://{S3_BUCKET}/{MODEL_NAME}",
        "num_gpu_blocks": NUM_GPU_BLOCKS,
        "max_batch_size": MAX_BATCH_SIZE,
    }

    # Validate layer count
    _total = sum(layers for _, layers in NODE_LAYER_MAPPING)
    assert _total == config["total_num_layers"], (
        f"Layer mismatch: node mapping sum={_total} != config total={config['total_num_layers']}"
    )
    estimated_throughput = 14.20

    pipeline_task = asyncio.create_task(
        create_pipeline_async(config, NODE_LAYER_MAPPING, estimated_throughput)
    )
    server_task = asyncio.create_task(global_server.run_global_server())

    try:
        logger.info("Waiting for pipeline creation to complete...")
        await pipeline_task
        logger.info("Pipeline is ready!")

        parser = argparse.ArgumentParser()
        parser.add_argument("--single-request", action="store_true",
                            help="Run 5 requests sequentially (max_concurrency=1)")
        args = parser.parse_args()

        if args.single_request:
            num_requests = 5
            max_concurrency = 1
        else:
            num_requests = NUM_REQUESTS
            max_concurrency = None

        metrics = await run_latency_benchmark(
            global_server=global_server,
            num_requests=num_requests,
            input_len=INPUT_LEN,
            output_len=OUTPUT_LEN,
            request_rate=float('inf'),
            model_name=MODEL_NAME,
            max_concurrency=max_concurrency,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,
            run_initial_test=False,
            test_requests_per_pipeline=2,
        )

        print_benchmark_results(metrics)
        save_benchmark_results(metrics, OUTPUT_PATH, extra={
            "system": "example",
            "model": MODEL_NAME,
            "total_layers": TOTAL_LAYERS,
            "instance_type": "g6.xlarge",
            "num_gpus": 1,
            "pp_layer_partition": [TOTAL_LAYERS],
            "parallel_strategy": [1],
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "num_requests": num_requests,
            "max_batch_size": MAX_BATCH_SIZE,
            "num_gpu_blocks": NUM_GPU_BLOCKS,
            "estimated_throughput_rps": estimated_throughput,
            "percentiles": [10, 25, 50, 75, 90, 99],
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
