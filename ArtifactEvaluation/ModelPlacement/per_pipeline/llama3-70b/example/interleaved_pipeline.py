"""
Interleaved Pipeline — Llama-3.1-8B on g6.12xlarge + g6.xlarge (5× L4 GPUs)

Tests interleaved GPU allocation across nodes:
  Stage 0: g6.12xlarge GPU 0-1 (TP=2, 13 layers)
  Stage 1: g6.xlarge   GPU 0   (TP=1,  6 layers)
  Stage 2: g6.12xlarge GPU 2-3 (TP=2, 13 layers)

The same g6.12xlarge node is used for stages 0 and 2,
verifying that non-contiguous GPU slices on a single node work correctly.
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
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TOTAL_LAYERS = 32
INPUT_LEN = 512
OUTPUT_LEN = 64
S3_BUCKET = "hetero-spot-llm-serve-models"

# Estimator results for interleaved pipeline (g6.12xlarge TP=2 + g6.xlarge TP=1 + g6.12xlarge TP=2):
#   max_batch_size = 1012, num_blocks = 36432
#   throughput = 8.70 req/s, latency = 116270.74 ms
MAX_BATCH_SIZE = 1012
NUM_GPU_BLOCKS = 36432
NUM_REQUESTS = MAX_BATCH_SIZE * 5

OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "ArtifactEvaluation", "ModelPlacement",
    "optimizer", "results", "llama3-70b", "measured"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "interleaved_Llama-3.1-8B.json")

# Interleaved pipeline — same g6.12xlarge used for stages 0 and 2
NODE_LAYER_MAPPING = [
    (g6_12xlarge_node_ip_1, 13),  # stage 0: g6.12xlarge GPU 0-1, TP=2
    (g6_xlarge_node_ip_1,    6),  # stage 1: g6.xlarge   GPU 0,   TP=1
    (g6_12xlarge_node_ip_1, 13),  # stage 2: g6.12xlarge GPU 2-3, TP=2
]

PP_LAYER_PARTITION = [13, 6, 13]
PARALLEL_STRATEGY = [2, 1, 2]


# ─── Benchmark ───────────────────────────────────────────────────────

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
    print(f"Interleaved Pipeline — Cross-Node GPU Test")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layers: {TOTAL_LAYERS} ({' + '.join(str(l) for l in PP_LAYER_PARTITION)})")
    print(f"  Strategy: TP={PARALLEL_STRATEGY}")
    print(f"  Nodes: g6.12xlarge (stages 0,2) + g6.xlarge (stage 1)")
    print(f"  Input/Output: {INPUT_LEN}/{OUTPUT_LEN} tokens")
    print(f"  Requests: {NUM_REQUESTS}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Node mapping:")
    for i, (ip, layers) in enumerate(NODE_LAYER_MAPPING):
        print(f"    Stage {i}: {ip or '(empty)'} → {layers} layers (TP={PARALLEL_STRATEGY[i]})")
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
        "pp_layer_partition": ",".join(str(l) for l in PP_LAYER_PARTITION),
        "parallel_strategy": PARALLEL_STRATEGY,
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

    estimated_throughput = 8.70

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
            "system": "interleaved_test",
            "model": MODEL_NAME,
            "total_layers": TOTAL_LAYERS,
            "instance_types": ["g6.12xlarge", "g6.xlarge", "g6.12xlarge"],
            "num_gpus": 5,
            "pp_layer_partition": PP_LAYER_PARTITION,
            "parallel_strategy": PARALLEL_STRATEGY,
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
