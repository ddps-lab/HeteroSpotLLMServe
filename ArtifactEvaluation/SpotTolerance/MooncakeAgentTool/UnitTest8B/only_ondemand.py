"""
Offline benchmark — On-Demand Only baseline unit test (Llama-3.1-8B-Instruct)

Baseline: runs on on-demand instances only with no spot interruption events.
Uses the same pipeline config but with on-demand node IPs.
"""
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
from typing import List

# ─── Path setup ──────────────────────────────────────────────────────
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
sys.path.insert(0, os.path.join(_d, "ArtifactEvaluation", "ModelPlacement"))
_REPO_ROOT = _d
DATASET_PATH = os.path.join(_REPO_ROOT, "ArtifactEvaluation", "Datasets", "MooncakeToolAgentTrace_pruned_16384.csv")  # long-context Mooncake tool-agent trace
del _d

from global_server import GlobalServer
from benchmark_utils import (
    print_benchmark_results, run_trace_benchmark, DEFAULT_DATASET_PATH
)
from save_results import save_benchmark_results

# ─── Load JSON configs ───────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPOT_TOLERANCE_DIR = os.path.dirname(SCRIPT_DIR)  # SpotTolerance/

with open(os.path.join(SCRIPT_DIR, "pipelines_8b.json")) as f:
    pipelines_data = json.load(f)
with open(os.path.join(SCRIPT_DIR, "nodes.json")) as f:
    nodes_map = json.load(f)

OUTPUT_DIR = os.path.join(SPOT_TOLERANCE_DIR, "results", "UnitTest8B")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "offline_only_ondemand.json")

logger = logging.getLogger(__name__)


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
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, os.path.basename(__file__).replace(".py", ".log"))
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

    model_name = pipelines_data["model"]
    pipelines = pipelines_data["pipelines"]

    logger.info("=" * 70)
    logger.info(f"Offline Benchmark — On-Demand Only Baseline — UnitTest8B")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Pipelines: {len(pipelines)}")
    for i, p in enumerate(pipelines):
        mapping = " → ".join(f"{n}:{l}" for n, l in p["node_layer_mapping"])
        logger.info(f"  {p['label']}: tput={p['predicted_throughput_rps']:.3f}  {mapping}")
    logger.info(f"  Total predicted: {pipelines_data['total_throughput_rps']:.3f} req/s")
    logger.info("=" * 70)

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
    for p in pipelines:
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

    # ── No spot events — baseline run ─────────────────────────────────

    try:
        logger.info("Waiting for pipeline creation to complete...")
        for task in pipeline_tasks:
            await task
        logger.info("All pipelines are ready!")

        metrics = await run_trace_benchmark(
            global_server=global_server,
            dataset_path=DATASET_PATH,
            trace_output_prefix="spottolerance_offline_only_ondemand_unittest8b",
            trace_base_dir=OUTPUT_DIR,
            num_requests=None,
            time_scale=1.0,
            model_name=model_name,
            percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99],
            disable_tqdm=False,
            run_initial_test=False,
            max_duration=12 * 60,
            logger=logger,
        )

        print_benchmark_results(metrics, logger=logger)

        save_benchmark_results(metrics, OUTPUT_PATH, extra={
            "system": "OnlyOnDemand",
            "benchmark_type": "offline",
            "scenario": "UnitTest8B",
            "num_pipelines": len(pipelines),
            "predicted_total_throughput_rps": pipelines_data["total_throughput_rps"],
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
