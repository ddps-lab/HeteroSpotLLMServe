"""
vLLM Pipeline 2 (vL-P2)
g5.12xlarge×2, TP=[4,4], Layers=[40,40]
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_latency_benchmark
from save_results import save_benchmark_results
from nodes import *

# ─── Load config from optimizer results ──────────────────────────────

PIPELINE_INDEX = 1  # vL-P2
STAGE_LAYER_COUNT_IDX = 1

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PREDICTED_FILE = [f for f in os.listdir(RESULTS_DIR) if f.startswith("predicted_vllm_")][0]

with open(os.path.join(RESULTS_DIR, PREDICTED_FILE)) as f:
    _data = json.load(f)
_pipeline = _data["pipelines"][PIPELINE_INDEX]

S3_BUCKET = "hetero-spot-llm-serve-models"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "vllm_p2.json")

# ─── Node assignment ─────────────────────────────────────────────────
# vL-P2: g5.12xlarge×2 (homogeneous, TP=4 each)

NODE_LAYER_MAPPING = [
    (g5_12xlarge_node_ip_1, int(_pipeline["stages"][0][STAGE_LAYER_COUNT_IDX])),  # g5.12xlarge TP=4
    (g5_12xlarge_node_ip_2, int(_pipeline["stages"][1][STAGE_LAYER_COUNT_IDX])),  # g5.12xlarge TP=4
]


# ─── Benchmark ───────────────────────────────────────────────────────

async def test_benchmark():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    model_name = _data["model"]

    print("=" * 70)
    print(f"vLLM Pipeline 2 — {PREDICTED_FILE}")
    print(f"  Stages: {_pipeline['stages']}")
    print(f"  PP: {_pipeline['pp_layer_partition']}  TP: {_pipeline['parallel_strategy']}")
    print(f"  Predicted: {_pipeline['predicted_throughput_rps']:.3f} req/s")
    print(f"  Max batch size: {_pipeline['max_batch_size']}")
    print(f"  Num GPU blocks: {_pipeline['num_blocks']}")
    print(f"  Node mapping:")
    for ip, layers in NODE_LAYER_MAPPING:
        print(f"    {ip or '(empty)'} → {layers} layers")
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
        "model_name": model_name,
        "total_num_layers": sum(int(s[STAGE_LAYER_COUNT_IDX]) for s in _pipeline["stages"]),
        "gpu_memory_utilization": _pipeline.get("gpu_memory_utilization", 0.85),
        "pp_layer_partition": _pipeline["pp_layer_partition"],
        "parallel_strategy": _pipeline["parallel_strategy"],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://{S3_BUCKET}/{model_name}",
        "num_gpu_blocks": _pipeline["num_blocks"],
        "max_batch_size": int(_pipeline["max_batch_size"]),
    }

    estimated_throughput = _pipeline["predicted_throughput_rps"]
    max_batch_size = int(_pipeline["max_batch_size"])

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
            num_requests = max_batch_size * 5
            max_concurrency = None

        metrics = await run_latency_benchmark(
            global_server=global_server,
            num_requests=num_requests,
            input_len=_data["workload"]["input_len"],
            output_len=_data["workload"]["output_len"],
            request_rate=float('inf'),
            model_name=model_name,
            max_concurrency=max_concurrency,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,
            run_initial_test=False,
            test_requests_per_pipeline=2,
        )

        print_benchmark_results(metrics)

        save_benchmark_results(metrics, OUTPUT_PATH, extra={
            "system": "vLLM",
            "pipeline": f"P{PIPELINE_INDEX + 1}",
            "pp_layer_partition": _pipeline["pp_layer_partition"],
            "parallel_strategy": _pipeline["parallel_strategy"],
            "stages": _pipeline["stages"],
            "input_len": _data["workload"]["input_len"],
            "output_len": _data["workload"]["output_len"],
            "num_requests": num_requests,
            "predicted_throughput_rps": estimated_throughput,
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
