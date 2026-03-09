"""
Generate benchmark p files from estimation results.

For each estimation JSON in {model}/results/est_*.json,
generates a corresponding p file at {model}/{instance_dir}/{strategy}.py

Usage:
    python generate_p_files.py                      # all models
    python generate_p_files.py --model llama3-3b    # single model
"""
import argparse
import json
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = '''"""
Performance Estimation Benchmark: {model_name}
Instance: {instance_type} ({gpu_type} ×{gpu_count})
Strategy: TP={tp_size}, PP={pp_size} ({strategy_label})
Generated from estimation results.

Sweeps batch sizes: 1, 2, 4, 8, ..., MAX_BATCH_SIZE
Pipeline is created once; benchmark runs for each batch size.
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
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_latency_benchmark

# ─── Node IP (from shared nodes.py) ──────────────────────────────────
_pe_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _pe_dir)
from nodes import get_node_ip
NODE_IP = get_node_ip("{instance_type}")

# ─── Load estimation config ──────────────────────────────────────────
EST_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "data", "estimated",
                        "est_{instance_dir}_{strategy_label}.json")
with open(EST_FILE) as f:
    EST = json.load(f)

MODEL_NAME = EST["model"]
PP_LAYER_PARTITION = EST["pp_layer_partition_str"]
PARALLEL_STRATEGY = EST["parallel_strategy"]
MAX_BATCH_SIZE = EST["max_batch_size"]
NUM_GPU_BLOCKS = EST["num_blocks"]

S3_BUCKET = "hetero-spot-llm-serve-models"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data", "measured")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "bench_{instance_dir}_{strategy_label}.json")

# ─── Batch sweep: 1, 2, 4, 8, ..., MAX_BATCH_SIZE ────────────────────
SWEEP_BATCH_SIZES = []
bs = 1
while bs < MAX_BATCH_SIZE:
    SWEEP_BATCH_SIZES.append(bs)
    bs *= 2
SWEEP_BATCH_SIZES.append(MAX_BATCH_SIZE)

# ─── Node layer mapping ──────────────────────────────────────────────
NODE_LAYER_MAPPING = [
{node_mapping_block}
]


def extract_metrics(metrics, batch_size):
    """Extract key metrics from a benchmark run."""
    return {{
        "batch_size": batch_size,
        "completed": metrics.completed,
        "total_input": metrics.total_input,
        "total_output": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "mean_e2el_ms": metrics.mean_e2el_ms,
        "median_e2el_ms": metrics.median_e2el_ms,
        "benchmark_duration": metrics.benchmark_duration,
    }}


def save_results(all_results, output_path, meta):
    """Save all batch sweep results to a single JSON."""
    output = {{**meta, "batch_sweep": all_results}}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\\nResults saved to {{output_path}}")


async def test_benchmark():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)
    logger.propagate = False

    print("=" * 70)
    print(f"Performance Estimation Benchmark (Batch Sweep)")
    print(f"  Model: {{MODEL_NAME}}")
    print(f"  Instance: {instance_type} ({gpu_type} ×{gpu_count})")
    print(f"  Strategy: TP={tp_size}, PP={pp_size} → {{PARALLEL_STRATEGY}}")
    print(f"  PP partition: {{PP_LAYER_PARTITION}}")
    print(f"  Max batch: {{MAX_BATCH_SIZE}}, Blocks: {{NUM_GPU_BLOCKS}}")
    print(f"  Estimated throughput: {{EST['estimated_throughput_rps']:.4f}} req/s")
    print(f"  Batch sizes: {{SWEEP_BATCH_SIZES}}")
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

    config = {{
        "model_name": MODEL_NAME,
        "total_num_layers": sum(EST["pp_layer_partition"]),
        "gpu_memory_utilization": EST["gpu_memory_utilization"],
        "pp_layer_partition": PP_LAYER_PARTITION,
        "parallel_strategy": PARALLEL_STRATEGY,
        "max_model_len": EST["workload"]["max_model_len"],
        "max_num_batched_tokens": EST["workload"]["max_model_len"],
        "max_num_seqs": MAX_BATCH_SIZE,
        "model_source": "s3",
        "s3_path": f"s3://{{S3_BUCKET}}/{{MODEL_NAME}}",
        "num_gpu_blocks": NUM_GPU_BLOCKS,
        "max_batch_size": MAX_BATCH_SIZE,
    }}

    server_task = None
    all_results = []

    try:
        for idx, current_batch_size in enumerate(SWEEP_BATCH_SIZES):
            num_requests = current_batch_size * 10

            # (Re)build pipeline with current batch size
            # num_gpu_blocks stays the same; only max_batch_size changes
            config["max_batch_size"] = current_batch_size
            config["max_num_seqs"] = current_batch_size

            if idx > 0:
                logger.info("Stopping previous pipeline...")
                global_server.cluster.stop_all_pipelines()

            logger.info(f"Creating pipeline with max_batch_size={{current_batch_size}}...")
            pipeline_task = asyncio.create_task(
                create_pipeline_async(config, NODE_LAYER_MAPPING, EST["estimated_throughput_rps"])
            )
            if idx == 0:
                server_task = asyncio.create_task(global_server.run_global_server())

            await pipeline_task
            logger.info("Pipeline ready!")

            print(f"\\n{{'-' * 60}}")
            print(f"  Batch size: {{current_batch_size}} ({{num_requests}} requests)")
            print(f"{{'-' * 60}}")

            metrics = await run_latency_benchmark(
                global_server=global_server,
                num_requests=num_requests,
                input_len=EST["workload"]["input_len"],
                output_len=EST["workload"]["output_len"],
                request_rate=float('inf'),
                model_name=MODEL_NAME,
                max_concurrency=float('inf'),
                percentiles=[10, 25, 50, 75, 90, 99],
                disable_tqdm=False,
                run_initial_test=(idx == 0),
                test_requests_per_pipeline=2,
            )

            print_benchmark_results(metrics)
            all_results.append(extract_metrics(metrics, current_batch_size))

            # Save after every batch size (incremental, prevents data loss)
            save_results(all_results, OUTPUT_PATH, meta={{
            "model": MODEL_NAME,
            "instance_type": "{instance_type}",
            "gpu_type": "{gpu_type}",
            "gpu_count": {gpu_count},
            "tp_size": {tp_size},
            "pp_size": {pp_size},
            "strategy_label": "{strategy_label}",
            "parallel_strategy": PARALLEL_STRATEGY,
            "pp_layer_partition": PP_LAYER_PARTITION,
            "max_batch_size": MAX_BATCH_SIZE,
            "num_gpu_blocks": NUM_GPU_BLOCKS,
            "estimated_throughput_rps": EST["estimated_throughput_rps"],
            "input_len": EST["workload"]["input_len"],
            "output_len": EST["workload"]["output_len"],
        }})

    except KeyboardInterrupt:
        logger.info("\\nBenchmark interrupted")
        if all_results:
            logger.info(f"Saving {{len(all_results)}} completed batch results...")
            save_results(all_results, OUTPUT_PATH, meta={{
                "model": MODEL_NAME,
                "instance_type": "{instance_type}",
                "strategy_label": "{strategy_label}",
                "partial": True,
            }})
    except Exception as e:
        logger.error(f"Benchmark failed: {{e}}")
        raise
    finally:
        logger.info("Cleaning up...")
        server_task.cancel()
        pipeline_task.cancel()
        try:
            await asyncio.gather(server_task, return_exceptions=True)
        except:
            pass
        try:
            global_server.cluster.stop_all_pipelines()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_benchmark())
'''


def generate_for_model(model_key: str):
    results_dir = os.path.join(BASE_DIR, model_key, "results", "data", "estimated")
    est_files = sorted(glob.glob(os.path.join(results_dir, "est_*.json")))

    if not est_files:
        print(f"  No estimation results for {model_key}. Run estimate.py first.")
        return

    print(f"\n{'='*60}")
    print(f"  Generating p files for {model_key}")
    print(f"{'='*60}")

    for est_file in est_files:
        with open(est_file) as f:
            est = json.load(f)

        if not est["feasible"]:
            print(f"  [skip] {est['instance_type']} {est['strategy_label']} — infeasible")
            continue

        instance_dir = est["instance_type"].replace(".", "_")
        strategy_label = est["strategy_label"]
        pp_partition = est["pp_layer_partition"]

        # Build node mapping block
        # All stages are on the same physical node (same IP)
        node_lines = []
        for i, layers in enumerate(pp_partition):
            node_lines.append(f"    # stage[{i}]: {layers} layers (TP={est['parallel_strategy'][i]})")
            node_lines.append(f"    (NODE_IP, {layers}),")
        node_mapping_block = "\n".join(node_lines)

        content = TEMPLATE.format(
            model_name=est["model"],
            instance_type=est["instance_type"],
            instance_dir=instance_dir,
            gpu_type=est["gpu_type"],
            gpu_count=est["gpu_count"],
            tp_size=est["tp_size"],
            pp_size=est["pp_size"],
            strategy_label=strategy_label,
            node_mapping_block=node_mapping_block,
        )

        out_dir = os.path.join(BASE_DIR, model_key, instance_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{strategy_label}.py")
        with open(out_path, "w") as f:
            f.write(content)
        print(f"  [created] {model_key}/{instance_dir}/{strategy_label}.py")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark p files from estimation results")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to generate for (default: all)")
    args = parser.parse_args()

    model_dirs = [args.model] if args.model else [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d, "results"))
    ]

    for model_key in sorted(model_dirs):
        generate_for_model(model_key)


if __name__ == "__main__":
    main()
