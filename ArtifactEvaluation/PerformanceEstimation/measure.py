"""
Measure actual throughput for varying batch sizes and TP/PP configurations.
Uses synthetic fixed-length requests (not Azure trace) for controlled measurement.

For multi-GPU instances, measures all TP/PP combinations.

Usage:
    python measure.py --config 70B_L40S
    python measure.py --config 70B_L40S --input-len 763 --output-len 232 --num-requests 50
    python measure.py --config 32B_L4

Prerequisites:
    - nodes.py에 인스턴스 IP 채울 것
    - num_gpu_blocks / max_batch_size를 predict.py 결과로 채울 것
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
from request_handler import generate_random_requests
from benchmark_utils import (
    calculate_benchmark_metrics,
    print_benchmark_results,
    run_benchmark_requests,
)
from nodes import *

S3_BUCKET = "hetero-spot-llm-serve-models"

# ─── Experiment Configurations ───────────────────────────────────────────────

CONFIGS = {
    "70B_L40S": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "total_num_layers": 80,
        "node_ip": lambda: g6e_12xlarge_node_ip,
        "instance_type": "g6e.12xlarge",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        # num_gpu_blocks / max_batch_size: predict.py 결과로 채울 것
        "variations": [
            {
                "label": "TP=4, PP=1",
                "tp": 4, "pp": 1,
                "parallel_strategy": [4],
                "pp_layer_partition": "80",
                "num_gpu_blocks": 2052,     # ← predict.py num_blocks
                "max_batch_size": 33,     # ← predict.py max_batch
                "batch_sizes": [1, 2, 4, 8, 16, 32, 33],
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
    "32B_L4": {
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "total_num_layers": 64,
        "node_ip": lambda: g6_12xlarge_node_ip,
        "instance_type": "g6.12xlarge",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "variations": [
            {
                "label": "TP=4, PP=1",
                "tp": 4, "pp": 1,
                "parallel_strategy": [4],
                "pp_layer_partition": "64",
                "num_gpu_blocks": 0,
                "max_batch_size": 0,
                "batch_sizes": [1, 2, 4, 8, 16],
            },
            {
                "label": "TP=2, PP=2",
                "tp": 2, "pp": 2,
                "parallel_strategy": [2, 2],
                "pp_layer_partition": "32,32",
                "num_gpu_blocks": 0,
                "max_batch_size": 0,
                "batch_sizes": [1, 2, 4, 8, 16],
            },
            {
                "label": "TP=1, PP=4",
                "tp": 1, "pp": 4,
                "parallel_strategy": [1, 1, 1, 1],
                "pp_layer_partition": "16,16,16,16",
                "num_gpu_blocks": 0,
                "max_batch_size": 0,
                "batch_sizes": [1, 2, 4, 8],
            },
        ],
    },
}


async def run_synthetic_benchmark(
    global_server: GlobalServer,
    num_requests: int,
    input_len: int,
    output_len: int,
    model_name: str,
    max_concurrency: int = None,
    percentiles: List[float] = None,
    run_initial_test: bool = True,
    test_requests_per_pipeline: int = 2,
):
    """
    Run benchmark with synthetic fixed-length requests.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 99]

    print(f"  Generating {num_requests} synthetic requests "
          f"(input={input_len}, output={output_len})...")

    request_inputs = generate_random_requests(
        num_prompts=num_requests,
        input_len=input_len,
        output_len=output_len,
        model_name=model_name,
        ignore_eos=True,
    )

    # Initial test
    if run_initial_test:
        num_pipelines = len(global_server.cluster.pipelines)
        test_count = test_requests_per_pipeline * num_pipelines
        test_inputs = request_inputs[:test_count] if test_count <= len(request_inputs) else \
            generate_random_requests(
                num_prompts=test_count, input_len=input_len, output_len=output_len,
                model_name=model_name, ignore_eos=True,
            )
        print(f"  Running {test_count} warmup requests...")
        for test_input in test_inputs:
            request = await global_server.add_request_and_wait(test_input)
            if not (request.output and request.output.success):
                error_msg = request.output.error if request.output else "No output"
                raise ValueError(f"Warmup failed: {error_msg}")
        print(f"  Warmup done.")

    # Run benchmark (offline: send all at once)
    requests, actual_duration = await run_benchmark_requests(
        global_server, request_inputs,
        request_rate=float('inf'),
        max_concurrency=max_concurrency,
        disable_tqdm=False,
    )

    metrics = calculate_benchmark_metrics(
        requests, request_inputs, actual_duration, percentiles
    )

    return metrics


async def measure_single_variation(
    config_name: str,
    exp_config: Dict,
    variation: Dict,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_requests: int,
    logger: logging.Logger,
):
    """
    Measure throughput for a single TP/PP variation at a specific batch size.
    """
    model_name = exp_config["model_name"]
    node_ip = exp_config["node_ip"]()
    label = variation["label"]

    if not node_ip:
        raise ValueError(f"Node IP not set for {config_name}. Edit nodes.py first.")

    num_gpu_blocks = variation.get("num_gpu_blocks", 0)

    if num_gpu_blocks <= 0:
        raise ValueError(
            f"num_gpu_blocks not set for {label}. "
            f"Run predict.py first and fill in the value."
        )

    layers = [int(x) for x in variation["pp_layer_partition"].split(",")]
    node_layer_mapping = [(node_ip, l) for l in layers]

    pipeline_config = {
        "model_name": model_name,
        "total_num_layers": exp_config["total_num_layers"],
        "gpu_memory_utilization": exp_config["gpu_memory_utilization"],
        "pp_layer_partition": variation["pp_layer_partition"],
        "parallel_strategy": variation["parallel_strategy"],
        "max_model_len": exp_config["max_model_len"],
        "max_num_batched_tokens": exp_config["max_num_batched_tokens"],
        "max_num_seqs": 512,  # vLLM engine 쪽 (고정)
        "num_gpu_blocks": num_gpu_blocks,
        "max_batch_size": batch_size,  # GlobalServer scheduler가 batch size 제어
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

    try:
        logger.info(f"[{label} bs={batch_size}] Creating pipeline...")
        await create_pipeline_async()
        logger.info(f"[{label} bs={batch_size}] Pipeline ready.")

        metrics = await run_synthetic_benchmark(
            global_server=global_server,
            num_requests=num_requests,
            input_len=input_len,
            output_len=output_len,
            model_name=model_name,
            max_concurrency=None,  # inf: pipeline의 max_num_seqs가 batch size를 제어
            percentiles=[10, 25, 50, 75, 90, 99],
            run_initial_test=True,
            test_requests_per_pipeline=2,
        )

        return {
            "batch_size": batch_size,
            "throughput_rps": metrics.request_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
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


async def run_experiment(
    config_name: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    batch_sizes_override: str = None,
):
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
    print(f"Synthetic workload: input={input_len}, output={output_len}")
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

        print(f"\n{'─' * 60}")
        print(f"  {label} (batch_sizes={batch_sizes})")
        print(f"{'─' * 60}")
        var_results = []

        for bs in batch_sizes:
            # --num-requests로 override 가능, 아니면 batch_size × 10
            bs_num_requests = num_requests if num_requests else bs * 10
            print(f"\n  ▶ Batch Size: {bs} (num_requests={bs_num_requests})")
            try:
                result = await measure_single_variation(
                    config_name, exp_config, variation,
                    bs, input_len, output_len, bs_num_requests, logger,
                )
                var_results.append(result)
                print(f"  → Throughput: {result['throughput_rps']:.4f} req/s, "
                      f"TTFT: {result['mean_ttft_ms']:.1f}ms, "
                      f"TPOT: {result['mean_tpot_ms']:.1f}ms")
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
        "workload": {"input_len": input_len, "output_len": output_len},
        "num_requests_per_bs": num_requests,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary table
    header = f"{'Config':<16} | {'Batch':>6} | {'Throughput':>12} | {'TTFT(ms)':>10} | {'TPOT(ms)':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for var_result in all_results:
        for r in var_result["batch_results"]:
            if r.get("throughput_rps"):
                print(f"{var_result['label']:<16} | {r['batch_size']:>6} | "
                      f"{r['throughput_rps']:>12.4f} | "
                      f"{r.get('mean_ttft_ms', 0):>10.1f} | "
                      f"{r.get('mean_tpot_ms', 0):>10.1f}")
            else:
                print(f"{var_result['label']:<16} | {r['batch_size']:>6} | "
                      f"{'FAILED':>12} | {'—':>10} | {'—':>10}")
        print("-" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--input-len", type=int, default=763,
                        help="Fixed input token length (default: 763, Azure avg)")
    parser.add_argument("--output-len", type=int, default=232,
                        help="Fixed output token length (default: 232, Azure avg)")
    parser.add_argument("--num-requests", type=int, default=None,
                        help="Override request count (default: batch_size × 10, min 20)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Override batch sizes for ALL variations (comma-separated)")
    args = parser.parse_args()

    asyncio.run(run_experiment(
        args.config, args.num_requests, args.input_len, args.output_len, args.batch_sizes
    ))


if __name__ == "__main__":
    main()
