"""
Offline benchmark — ShuntServe spot tolerance (Llama-3.1-70B, Scenario A)

Pipelines, nodes, and spot-trace events are loaded from JSON config files.
On interruption: spot nodes are replaced with on-demand counterparts via switch_nodes().
On restore: on-demand nodes are switched back to recovered spot nodes.
"""
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import List

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

SCENARIO = "A"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # llama3-70b/
SPOT_TOLERANCE_DIR = os.path.dirname(MODEL_DIR)              # SpotTolerance/

with open(os.path.join(MODEL_DIR, f"pipelines_llama3_70b_scenario_{SCENARIO}.json")) as f:
    pipelines_data = json.load(f)
with open(os.path.join(SPOT_TOLERANCE_DIR, f"nodes_scenario_{SCENARIO}.json")) as f:
    nodes_map = json.load(f)
with open(os.path.join(SPOT_TOLERANCE_DIR, f"spot_trace_events_scenario_{SCENARIO}.json")) as f:
    events_data = json.load(f)

OUTPUT_DIR = os.path.join(SPOT_TOLERANCE_DIR, "results", "llama3-70b", "offline", f"scenario_{SCENARIO}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "offline_shuntserve.json")

# ─── Benchmark time parameters (minutes) ─────────────────────────────
START_TIME_MIN = 0
END_TIME_MIN = 60        # 1 hour of trace data
MAX_DURATION_MIN = 60    # 1 hour wall-clock limit

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────

def get_counterpart_name(name: str) -> str:
    """spot_xxx → on_demand_xxx, on_demand_xxx → spot_xxx"""
    if name.startswith("spot_"):
        return "on_demand_" + name[len("spot_"):]
    elif name.startswith("on_demand_"):
        return "spot_" + name[len("on_demand_"):]
    raise ValueError(f"Unknown prefix: {name}")


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
    logger.info(f"Offline Benchmark — ShuntServe Spot Tolerance — Scenario {SCENARIO}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Pipelines: {len(pipelines)}")
    for i, p in enumerate(pipelines):
        mapping = " → ".join(f"{n}:{l}" for n, l in p["node_layer_mapping"])
        logger.info(f"  {p['label']}: tput={p['predicted_throughput_rps']:.3f}  {mapping}")
    logger.info(f"  Total predicted: {pipelines_data['total_throughput_rps']:.3f} req/s")
    logger.info(f"  Events: {len(events_data['events'])}")
    for ev in events_data["events"]:
        logger.info(f"    t={ev['time_min']}min  {ev['type']}  {ev['instances']}")
    logger.info("=" * 70)

    global_server = GlobalServer(request_handler_mode="migration")

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

    # ── Schedule spot events ──────────────────────────────────────────

    async def switch_node_after_delay(event_time: float, old_ips: List[str], new_ips: List[str]):
        """Switch nodes after a specified delay."""
        await asyncio.sleep(event_time)
        try:
            logger.info(f"Starting node switch: {old_ips} -> {new_ips}")
            switch_start = time.time()
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor, global_server.switch_nodes, old_ips, new_ips
                )
            duration = time.time() - switch_start
            logger.info(f"Node switch completed in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Node switch failed: {e}")

    # Group events by time_min so simultaneous events become a single switch_nodes call
    grouped_events = defaultdict(list)
    for event in events_data["events"]:
        grouped_events[event["time_min"]].append(event)

    # Track which spot nodes are currently interrupted → on_demand replacement
    # Pre-populate from initial pipeline state: on_demand nodes imply spot counterpart unavailable
    interrupted_spots = {}
    for p in pipelines:
        for node_name, _ in p["node_layer_mapping"]:
            if node_name.startswith("on_demand_"):
                spot_name = get_counterpart_name(node_name)
                interrupted_spots[spot_name] = node_name

    event_schedule = []
    for time_min in sorted(grouped_events.keys()):
        event_time = time_min * 60
        all_old_names = []
        all_new_names = []

        for event in grouped_events[time_min]:
            if event["type"] == "interruption":
                old_names = event["instances"]
                new_names = [get_counterpart_name(n) for n in old_names]
                for spot, od in zip(old_names, new_names):
                    interrupted_spots[spot] = od
                all_old_names.extend(old_names)
                all_new_names.extend(new_names)

            elif event["type"] == "restore":
                old_names = [interrupted_spots[n] for n in event["instances"]]
                new_names = event["instances"]
                for n in new_names:
                    del interrupted_spots[n]
                all_old_names.extend(old_names)
                all_new_names.extend(new_names)

        assert len(all_old_names) == len(all_new_names), (
            f"t={time_min}min: old/new node count mismatch: "
            f"{len(all_old_names)} old vs {len(all_new_names)} new"
        )

        event_schedule.append((
            event_time,
            [nodes_map[n] for n in all_old_names],
            [nodes_map[n] for n in all_new_names],
        ))

    # ── Run benchmark ─────────────────────────────────────────────────

    try:
        logger.info("Waiting for pipeline creation to complete...")
        for task in pipeline_tasks:
            await task
        logger.info("All pipelines are ready!")
        # Schedule spot events (after pipelines ready)
        for event_args in event_schedule:
            asyncio.create_task(switch_node_after_delay(*event_args))

        metrics = await run_trace_benchmark(
            global_server=global_server,
            dataset_path=DEFAULT_DATASET_PATH,
            trace_output_prefix=f"spottolerance_offline_shuntserve_scenario_{SCENARIO}",
            trace_base_dir=OUTPUT_DIR,
            num_requests=None,
            time_scale=0.0,
            model_name=model_name,
            percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99],
            disable_tqdm=False,
            run_initial_test=False,
            start_time=START_TIME_MIN * 60,
            end_time=END_TIME_MIN * 60,
            max_duration=MAX_DURATION_MIN * 60,
            logger=logger,
        )

        print_benchmark_results(metrics, logger=logger)

        save_benchmark_results(metrics, OUTPUT_PATH, extra={
            "system": "ShuntServe",
            "benchmark_type": "offline",
            "scenario": SCENARIO,
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
