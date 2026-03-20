"""
Offline benchmark — Warmup baseline (Llama-3.1-70B, Scenario B)

Provisions ALL available nodes into pipelines, no spot events.
ShuntServe pipelines (P1, P2) + extra pipeline for remaining nodes.

Remaining nodes (not in ShuntServe config):
  - spot_g6_12xlarge 1,2,3 + spot_g6e_xlarge 3,4 → P3 (20+20+20+10+10, pp=[4,4,4,1,1])
"""
import asyncio
import concurrent.futures
import json
import logging
import os
import sys

# ─── Path setup ──────────────────────────────────────────────────────
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
sys.path.insert(0, os.path.join(_d, "ArtifactEvaluation", "ModelPlacement"))
_REPO_ROOT = _d
del _d

from global_server import GlobalServer
from benchmark_utils import (
    print_benchmark_results, run_trace_benchmark, DEFAULT_DATASET_PATH
)
from save_results import save_benchmark_results

# ─── Load JSON configs ───────────────────────────────────────────────

SCENARIO = "B"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # llama3-70b/
SPOT_TOLERANCE_DIR = os.path.dirname(MODEL_DIR)              # SpotTolerance/

with open(os.path.join(MODEL_DIR, f"pipelines_llama3_70b_scenario_{SCENARIO}.json")) as f:
    pipelines_data = json.load(f)
with open(os.path.join(SPOT_TOLERANCE_DIR, f"nodes_scenario_{SCENARIO}.json")) as f:
    nodes_map = json.load(f)

OUTPUT_DIR = os.path.join(SPOT_TOLERANCE_DIR, "results", "llama3-70b", f"scenario_{SCENARIO}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "offline_warmup.json")

logger = logging.getLogger(__name__)

# ─── Extra pipelines for remaining nodes ─────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
S3_PATH = "s3://hetero-spot-llm-serve-models/meta-llama/Llama-3.1-70B-Instruct"

EXTRA_PIPELINES = [
    {
        "label": "WU-P3",
        "predicted_throughput_rps": 0.8,
        "config": {
            "model_name": MODEL_NAME,
            "total_num_layers": 80,
            "gpu_memory_utilization": 0.85,
            "pp_layer_partition": "20,20,20,10,10",
            "parallel_strategy": [4, 4, 4, 1, 1],
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 512,
            "model_source": "s3",
            "s3_path": S3_PATH,
            "num_gpu_blocks": 1181,
            "max_batch_size": 19,
        },
        "node_layer_mapping": [
            ["spot_g6_12xlarge_node_ip_1", 20],
            ["spot_g6_12xlarge_node_ip_2", 20],
            ["spot_g6_12xlarge_node_ip_3", 20],
            ["spot_g6e_xlarge_node_ip_3", 10],
            ["spot_g6e_xlarge_node_ip_4", 10],
        ],
    },
]


# ─── Benchmark ───────────────────────────────────────────────────────

async def test_benchmark():
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

    model_name = pipelines_data["model"]
    all_pipelines = pipelines_data["pipelines"] + EXTRA_PIPELINES

    total_tput = sum(p["predicted_throughput_rps"] for p in all_pipelines)

    print("=" * 70)
    print(f"Offline Benchmark — Warmup (All Nodes) — Scenario {SCENARIO}")
    print(f"  Model: {model_name}")
    print(f"  Pipelines: {len(all_pipelines)}")
    for i, p in enumerate(all_pipelines):
        mapping = " → ".join(f"{n}:{l}" for n, l in p["node_layer_mapping"])
        print(f"  {p['label']}: tput={p['predicted_throughput_rps']:.3f}  {mapping}")
    print(f"  Total predicted: {total_tput:.3f} req/s")
    print("=" * 70)

    global_server = GlobalServer()

    # ── Create pipelines ──────────────────────────────────────────────

    async def create_pipeline_async(config, node_layer_mapping, throughput):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, global_server.create_pipeline,
                node_layer_mapping, config, throughput
            )
        logger.info("Pipeline creation completed")

    pipeline_tasks = []
    for p in all_pipelines:
        config = p["config"]
        node_layer_mapping = [
            (nodes_map[name], layers)
            for name, layers in p["node_layer_mapping"]
        ]
        task = asyncio.create_task(
            create_pipeline_async(config, node_layer_mapping, p["predicted_throughput_rps"])
        )
        pipeline_tasks.append(task)

    server_task = asyncio.create_task(global_server.run_global_server())

    # ── No spot events — warmup run ───────────────────────────────────

    try:
        logger.info("Waiting for pipeline creation to complete...")
        for task in pipeline_tasks:
            await task
        logger.info("All pipelines are ready!")

        metrics = await run_trace_benchmark(
            global_server=global_server,
            dataset_path=DEFAULT_DATASET_PATH,
            trace_output_prefix=f"spottolerance_offline_warmup_llama3_70b_scenario_{SCENARIO}",
            trace_base_dir=OUTPUT_DIR,
            num_requests=None,
            time_scale=0.0,
            model_name=model_name,
            percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99],
            disable_tqdm=False,
            run_initial_test=False,
            start_time=0,
            end_time=60 * 60,
            max_duration=3 * 60,
        )

        print_benchmark_results(metrics)

        save_benchmark_results(metrics, OUTPUT_PATH, extra={
            "system": "Warmup",
            "benchmark_type": "offline",
            "scenario": SCENARIO,
            "num_pipelines": len(all_pipelines),
            "predicted_total_throughput_rps": total_tput,
            "percentiles": [1, 5, 10, 25, 50, 75, 90, 95, 99],
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
