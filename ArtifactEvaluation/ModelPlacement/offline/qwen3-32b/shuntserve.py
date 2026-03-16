"""
Offline benchmark — ShuntServe (Qwen3-32B)
All pipelines loaded from predicted JSON, Azure Trace with time_scale=0.0 (offline).
"""
import asyncio
import concurrent.futures
import json
import logging
import sys
import os

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_REPO_ROOT = _d
del _d

from global_server import GlobalServer
from benchmark_utils import (
    print_benchmark_results, run_trace_benchmark, DEFAULT_DATASET_PATH
)
from save_results import save_benchmark_results
from nodes import *

# ─── Load config from optimizer results ──────────────────────────────

SYSTEM = "shuntserve"
STAGE_LAYER_COUNT_IDX = 1

PREDICTED_DIR = os.path.join(
    _REPO_ROOT, "ArtifactEvaluation", "ModelPlacement",
    "optimizer", "results", "qwen3-32b", "estimated"
)
OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "ArtifactEvaluation", "ModelPlacement",
    "optimizer", "results", "qwen3-32b", "measured"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTED_FILE = [f for f in os.listdir(PREDICTED_DIR) if f.startswith(f"predicted_{SYSTEM}_")][0]
with open(os.path.join(PREDICTED_DIR, PREDICTED_FILE)) as f:
    _data = json.load(f)

S3_BUCKET = "hetero-spot-llm-serve-models"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"offline_{SYSTEM}.json")

# ─── Node assignments per pipeline ───────────────────────────────────
# SS-P1: g6e.xl#1-4                       (TP=1 each)
# SS-P2: g6.12xl#1, g6.12xl#2, g6.12xl#3  (TP=4 each)
# SS-P3: g5.12xl#1                         (TP=4)
# SS-P4: g5.12xl#2                         (TP=4)

NODE_MAPPINGS = [
    # P1: 4 stages
    [
        (g6e_xlarge_node_ip_1, 0),
        (g6e_xlarge_node_ip_2, 1),
        (g6e_xlarge_node_ip_3, 2),
        (g6e_xlarge_node_ip_4, 3),
    ],
    # P2: 3 stages
    [
        (g6_12xlarge_node_ip_1, 0),
        (g6_12xlarge_node_ip_2, 1),
        (g6_12xlarge_node_ip_3, 2),
    ],
    # P3: 1 stage
    [
        (g5_12xlarge_node_ip_1, 0),
    ],
    # P4: 1 stage
    [
        (g5_12xlarge_node_ip_2, 0),
    ],
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
    pipelines = _data["pipelines"]

    print("=" * 70)
    print(f"Offline Benchmark — ShuntServe — {PREDICTED_FILE}")
    print(f"  Pipelines: {len(pipelines)}")
    for i, p in enumerate(pipelines):
        print(f"  P{i+1}: TP={p['parallel_strategy']}  layers={p['pp_layer_partition']}"
              f"  mbs={p['max_batch_size']}  tput={p['predicted_throughput_rps']:.3f}")
    print(f"  Total predicted: {_data['total_throughput_rps']:.3f} req/s")
    print("=" * 70)

    # Validate layer counts
    for i, p in enumerate(pipelines):
        total_layers = sum(int(s[STAGE_LAYER_COUNT_IDX]) for s in p["stages"])
        assert total_layers == sum(int(s[STAGE_LAYER_COUNT_IDX]) for s in pipelines[0]["stages"]), (
            f"P{i+1} total layers ({total_layers}) != "
            f"P1 total layers ({sum(int(s[STAGE_LAYER_COUNT_IDX]) for s in pipelines[0]['stages'])})"
        )
    expected_total_layers = sum(int(s[STAGE_LAYER_COUNT_IDX]) for s in pipelines[0]["stages"])
    print(f"  Layer check passed: all pipelines have {expected_total_layers} layers")

    global_server = GlobalServer()

    async def create_pipeline_async(config, node_layer_mapping, throughput):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, global_server.create_pipeline,
                node_layer_mapping, config, throughput
            )
        logger.info("Pipeline creation completed")

    pipeline_tasks = []
    for i, p in enumerate(pipelines):
        config = {
            "model_name": model_name,
            "total_num_layers": sum(int(s[STAGE_LAYER_COUNT_IDX]) for s in p["stages"]),
            "gpu_memory_utilization": p.get("gpu_memory_utilization", 0.85),
            "pp_layer_partition": p["pp_layer_partition"],
            "parallel_strategy": p["parallel_strategy"],
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 1024,
            "model_source": "s3",
            "s3_path": f"s3://{S3_BUCKET}/{model_name}",
            "num_gpu_blocks": p["num_blocks"],
            "max_batch_size": int(p["max_batch_size"]),
        }

        node_layer_mapping = [
            (ip, int(p["stages"][stage_idx][STAGE_LAYER_COUNT_IDX]))
            for ip, stage_idx in NODE_MAPPINGS[i]
        ]

        task = asyncio.create_task(
            create_pipeline_async(config, node_layer_mapping, p["predicted_throughput_rps"])
        )
        pipeline_tasks.append(task)

    server_task = asyncio.create_task(global_server.run_global_server())

    try:
        logger.info("Waiting for pipeline creation to complete...")
        for task in pipeline_tasks:
            await task
        logger.info("All pipelines are ready!")

        metrics = await run_trace_benchmark(
            global_server=global_server,
            dataset_path=DEFAULT_DATASET_PATH,
            trace_output_prefix=f"modelplacement_offline_{SYSTEM}",
            num_requests=None,
            time_scale=0.0,
            model_name=model_name,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False,
            run_initial_test=False,
            start_time=0,
            end_time=30 * 60,  # 30 minutes
        )

        print_benchmark_results(metrics)

        save_benchmark_results(metrics, OUTPUT_PATH, extra={
            "system": "ShuntServe",
            "benchmark_type": "offline",
            "num_pipelines": len(pipelines),
            "predicted_total_throughput_rps": _data["total_throughput_rps"],
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
        for task in pipeline_tasks:
            task.cancel()
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
