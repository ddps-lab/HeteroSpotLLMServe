"""
Measure actual throughput for varying batch sizes and TP/PP configurations.
Runs offline benchmark through GlobalServer with controlled max_num_seqs.

For multi-GPU instances (e.g., g6e.12xlarge = L40S×4), measures all TP/PP
combinations: TP=4/PP=1, TP=2/PP=2, TP=1/PP=4.

Usage:
    python measure.py --config 70B_L40S
    python measure.py --config 70B_L40S --batch-sizes 1,2,4,8
    python measure.py --config 32B_L4

Prerequisites:
    - nodes.py에 인스턴스 IP 채울 것
    - 해당 인스턴스에 SSH 접근 가능할 것
    - S3에 모델 weight 업로드되어 있을 것
"""
import asyncio
import argparse
import concurrent.futures
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

# Add GlobalServer to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
del _d

from global_server import GlobalServer
from benchmark_utils import (
    print_benchmark_results,
    run_trace_benchmark,
    DEFAULT_DATASET_PATH,
)
from nodes import *

S3_BUCKET = "hetero-spot-llm-serve-models"

# ─── Experiment Configurations ───────────────────────────────────────────────
#
# Each config defines:
#   - model, instance, node IP
#   - variations: list of TP/PP combinations to measure
#   - Each variation defines how to partition the model on the same physical instance
#
# For a 4-GPU instance (e.g., g6e.12xlarge):
#   TP=4 PP=1 → single vLLM engine with tp=4
#   TP=2 PP=2 → 2-stage pipeline, each stage tp=2 (using 2 GPUs)
#   TP=1 PP=4 → 4-stage pipeline, each stage tp=1 (using 1 GPU each)
#
# In practice, all variations run on the SAME physical instance.
# The pipeline is configured differently for each variation.

CONFIGS = {
    "70B_L40S": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "total_num_layers": 80,
        "node_ip": lambda: g6e_12xlarge_node_ip,
        "instance_type": "g6e.12xlarge",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        # num_gpu_blocks: predict.py 결과의 num_blocks 값을 넣을 것 (shared tensor store 할당용)
        # max_batch_size: predict.py 결과의 max_batch_size 값을 넣을 것 (scheduler 제한용)
        "variations": [
            {
                "label": "TP=4, PP=1",
                "tp": 4, "pp": 1,
                "parallel_strategy": [4],
                "pp_layer_partition": "80",
                "num_gpu_blocks": 2052,     # ← predict.py 결과로 채울 것
                "max_batch_size": 33,     # ← predict.py 결과로 채울 것
                "batch_sizes": [1, 2, 4, 8, 16, 32],
            },
            {
                "label": "TP=2, PP=2",
                "tp": 2, "pp": 2,
                "parallel_strategy": [2, 2],
                "pp_layer_partition": "40,40",
                "num_gpu_blocks": 1865,
                "max_batch_size": 30,
                "batch_sizes": [1, 2, 4, 8, 16, 30],
            },
            {
                "label": "TP=1, PP=4",
                "tp": 1, "pp": 4,
                "parallel_strategy": [1, 1, 1, 1],
                "pp_layer_partition": "20,20,20,20",
                "num_gpu_blocks": 684,
                "max_batch_size": 11,
                "batch_sizes": [1, 2, 4, 8, 11],
            },
        ],
    },
    # "32B_L4": {
    #     "model_name": "Qwen/Qwen2.5-32B-Instruct",
    #     "total_num_layers": 64,
    #     "node_ip": lambda: g6_12xlarge_node_ip,
    #     "instance_type": "g6.12xlarge",
    #     "gpu_memory_utilization": 0.85,
    #     "max_model_len": 8192,
    #     "max_num_batched_tokens": 8192,
    #     # num_gpu_blocks / max_batch_size: predict.py 결과로 채울 것
    #     "variations": [
    #         {
    #             "label": "TP=4, PP=1",
    #             "tp": 4, "pp": 1,
    #             "parallel_strategy": [4],
    #             "pp_layer_partition": "64",
    #             "num_gpu_blocks": 0,
    #             "max_batch_size": 0,
    #             "batch_sizes": [1, 2, 4, 8, 16],
    #         },
    #         {
    #             "label": "TP=2, PP=2",
    #             "tp": 2, "pp": 2,
    #             "parallel_strategy": [2, 2],
    #             "pp_layer_partition": "32,32",
    #             "num_gpu_blocks": 0,
    #             "max_batch_size": 0,
    #             "batch_sizes": [1, 2, 4, 8, 16],
    #         },
    #         {
    #             "label": "TP=1, PP=4",
    #             "tp": 1, "pp": 4,
    #             "parallel_strategy": [1, 1, 1, 1],
    #             "pp_layer_partition": "16,16,16,16",
    #             "num_gpu_blocks": 0,
    #             "max_batch_size": 0,
    #             "batch_sizes": [1, 2, 4, 8],
    #         },
    #     ],
    # },
}


async def measure_single_variation(
    config_name: str,
    exp_config: Dict,
    variation: Dict,
    batch_size: int,
    num_requests: int,
    logger: logging.Logger,
):
    """
    Measure throughput for a single TP/PP variation at a specific batch size.
    Creates a pipeline with the given parallel strategy and max_num_seqs = batch_size.
    """
    model_name = exp_config["model_name"]
    node_ip = exp_config["node_ip"]()

    if not node_ip:
        raise ValueError(f"Node IP not set for {config_name}. Edit nodes.py first.")

    pp_count = variation["pp"]
    # For PP > 1, all stages run on the same physical node
    # node_layer_mapping maps each stage to the same IP
    layers = [int(x) for x in variation["pp_layer_partition"].split(",")]
    node_layer_mapping = [(node_ip, l) for l in layers]

    num_gpu_blocks = variation.get("num_gpu_blocks", 0)
    max_batch_size = variation.get("max_batch_size", 0)

    if num_gpu_blocks <= 0:
        raise ValueError(
            f"num_gpu_blocks not set for {label}. "
            f"Run predict.py first and fill in the value from num_blocks."
        )

    pipeline_config = {
        "model_name": model_name,
        "total_num_layers": exp_config["total_num_layers"],
        "gpu_memory_utilization": exp_config["gpu_memory_utilization"],
        "pp_layer_partition": variation["pp_layer_partition"],
        "parallel_strategy": variation["parallel_strategy"],
        "max_model_len": exp_config["max_model_len"],
        "max_num_batched_tokens": exp_config["max_num_batched_tokens"],
        "max_num_seqs": batch_size,
        "num_gpu_blocks": num_gpu_blocks,
        "max_batch_size": max_batch_size,
        "model_source": "s3",
        "s3_path": f"s3://{S3_BUCKET}/{model_name}",
    }

    global_server = GlobalServer()

    async def create_pipeline_async():
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                global_server.create_pipeline,
                node_layer_mapping,
                pipeline_config,
                1.0,
            )

    server_task = asyncio.create_task(global_server.run_global_server())
    label = variation["label"]

    try:
        logger.info(f"[{label} bs={batch_size}] Creating pipeline...")
        await create_pipeline_async()
        logger.info(f"[{label} bs={batch_size}] Pipeline ready. Running benchmark...")

        metrics = await run_trace_benchmark(
            global_server=global_server,
            dataset_path=DEFAULT_DATASET_PATH,
            trace_output_prefix=f"estimation_{config_name}_{label.replace(' ', '').replace(',', '_').replace('=', '')}_bs{batch_size}",
            num_requests=num_requests,
            time_scale=0.0,
            model_name=model_name,
            percentiles=[50, 99],
            disable_tqdm=False,
            run_initial_test=True,
            test_requests_per_pipeline=2,
        )

        return {
            "batch_size": batch_size,
            "throughput_rps": metrics.request_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "completed": metrics.completed,
        }

    finally:
        server_task.cancel()
        try:
            await asyncio.gather(server_task, return_exceptions=True)
        except:
            pass
        try:
            global_server.cluster.stop_all_pipelines()
        except:
            pass


async def run_experiment(config_name: str, num_requests: int, batch_sizes_override: str = None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    exp_config = CONFIGS[config_name]

    print(f"\n{'=' * 90}")
    print(f"Performance Estimation Measurement: {config_name}")
    print(f"Model: {exp_config['model_name']}")
    print(f"Instance: {exp_config['instance_type']}")
    print(f"Requests per measurement: {num_requests}")
    print(f"{'=' * 90}\n")

    all_results = []

    for variation in exp_config["variations"]:
        label = variation["label"]
        batch_sizes = (
            [int(x) for x in batch_sizes_override.split(",")]
            if batch_sizes_override
            else variation["batch_sizes"]
        )

        print(f"\n--- {label} (batch_sizes={batch_sizes}) ---")
        var_results = []

        for bs in batch_sizes:
            print(f"\n  Batch Size: {bs}")
            try:
                result = await measure_single_variation(
                    config_name, exp_config, variation, bs, num_requests, logger
                )
                var_results.append(result)
                print(f"  → Throughput: {result['throughput_rps']:.4f} req/s")
            except Exception as e:
                logger.error(f"  Failed for {label} bs={bs}: {e}")
                var_results.append({
                    "batch_size": bs,
                    "throughput_rps": None,
                    "error": str(e),
                })

        all_results.append({
            "label": label,
            "tp": variation["tp"],
            "pp": variation["pp"],
            "batch_results": var_results,
        })

    # Save
    os.makedirs("results", exist_ok=True)
    output_file = f"results/measured_{config_name}.json"

    output = {
        "config": config_name,
        "model": exp_config["model_name"],
        "instance": exp_config["instance_type"],
        "num_requests_per_bs": num_requests,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary table
    header = f"{'Config':<16} | {'Batch':>6} | {'Throughput (req/s)':>20}"
    print(f"\n{header}")
    print("-" * len(header))
    for var_result in all_results:
        for r in var_result["batch_results"]:
            tp = f"{r['throughput_rps']:.4f}" if r.get("throughput_rps") else "FAILED"
            print(f"{var_result['label']:<16} | {r['batch_size']:>6} | {tp:>20}")
        print("-" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        choices=list(CONFIGS.keys()),
                        help="Experiment configuration")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Override batch sizes for ALL variations (comma-separated)")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests per batch size measurement")
    args = parser.parse_args()

    asyncio.run(run_experiment(args.config, args.num_requests, args.batch_sizes))


if __name__ == "__main__":
    main()
