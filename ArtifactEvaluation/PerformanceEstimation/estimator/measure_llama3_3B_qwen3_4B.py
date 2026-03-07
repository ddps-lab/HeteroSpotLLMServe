"""
Benchmark: Llama-3.2-3B vs Qwen3-4B on g6.xlarge (L4 x1).

Measures actual throughput on a real g6.xlarge instance.
Run predict (test_llama3_3B_qwen3_4B.py) first to get num_gpu_blocks / max_batch_size.

Usage:
    python measure_llama3_3B_qwen3_4B.py --config llama3_3B
    python measure_llama3_3B_qwen3_4B.py --config qwen3_4B
    python measure_llama3_3B_qwen3_4B.py --config llama3_3B --batch-sizes 1,2,4,8
    python measure_llama3_3B_qwen3_4B.py --config qwen3_4B --num-requests 50

Prerequisites:
    - nodes.py에 g6_xlarge_node_ip 설정할 것
    - num_gpu_blocks / max_batch_size를 estimator 결과로 채울 것
    - S3에 모델 weight 업로드되어 있을 것
"""
import asyncio
import argparse
import concurrent.futures
import json
import logging
import os
import sys
from typing import Dict, List

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
    run_benchmark_requests,
)
from nodes import *

S3_BUCKET = "hetero-spot-llm-serve-models"

# ─── Experiment Configurations ───────────────────────────────────────────────

CONFIGS = {
    "llama3_3B": {
        "model_name": "meta-llama/Llama-3.2-3B",
        "total_num_layers": 28,
        "node_ip": lambda: g6_xlarge_node_ip,
        "instance_type": "g6.xlarge",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "variations": [
            {
                "label": "TP=1, PP=1",
                "tp": 1, "pp": 1,
                "parallel_strategy": [1],
                "pp_layer_partition": "28",
                "num_gpu_blocks": 6280,      # ← estimator 결과로 채울 것
                "max_batch_size": 101,      # ← estimator 결과로 채울 것
                "batch_sizes": [1, 2, 4, 8, 16],
            },
        ],
    },
    "qwen3_4B": {
        "model_name": "Qwen/Qwen3-4B",
        "total_num_layers": 36,
        "node_ip": lambda: g6_xlarge_node_ip,
        "instance_type": "g6.xlarge",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "variations": [
            {
                "label": "TP=1, PP=1",
                "tp": 1, "pp": 1,
                "parallel_strategy": [1],
                "pp_layer_partition": "36",
                "num_gpu_blocks": 7213,      # ← estimator 결과로 채울 것
                "max_batch_size": 116,      # ← estimator 결과로 채울 것
                "batch_sizes": [1, 2, 4, 8, 16],
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
    model_name = exp_config["model_name"]
    node_ip = exp_config["node_ip"]()
    label = variation["label"]

    if not node_ip:
        raise ValueError(f"Node IP not set for {config_name}. Edit nodes.py first.")

    num_gpu_blocks = variation.get("num_gpu_blocks", 0)
    if num_gpu_blocks <= 0:
        raise ValueError(
            f"num_gpu_blocks not set for {label}. "
            f"Run test_llama3_3B_qwen3_4B.py first and fill in the value."
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
        "max_num_seqs": 512,
        "num_gpu_blocks": num_gpu_blocks,
        "max_batch_size": batch_size,
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
            max_concurrency=None,
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
            "percentiles_ttft_ms": {str(int(p)): v for p, v in metrics.percentiles_ttft_ms} if metrics.percentiles_ttft_ms else {},
            "percentiles_tpot_ms": {str(int(p)): v for p, v in metrics.percentiles_tpot_ms} if metrics.percentiles_tpot_ms else {},
            "percentiles_e2el_ms": {str(int(p)): v for p, v in metrics.percentiles_e2el_ms} if metrics.percentiles_e2el_ms else {},
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
    print(f"Benchmark: {config_name}")
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
            bs_num_requests = num_requests if num_requests else max(bs * 10, 20)
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
    output_file = f"results/measured_{config_name}_g6.xlarge.json"

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
    parser = argparse.ArgumentParser(
        description="Benchmark Llama-3.2-3B or Qwen3-4B on g6.xlarge"
    )
    parser.add_argument("--config", type=str, required=True,
                        choices=list(CONFIGS.keys()),
                        help="Model config to benchmark")
    parser.add_argument("--input-len", type=int, default=763,
                        help="Fixed input token length (default: 763)")
    parser.add_argument("--output-len", type=int, default=232,
                        help="Fixed output token length (default: 232)")
    parser.add_argument("--num-requests", type=int, default=None,
                        help="Override request count (default: batch_size × 10)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Override batch sizes (comma-separated)")
    args = parser.parse_args()

    asyncio.run(run_experiment(
        args.config, args.num_requests, args.input_len, args.output_len, args.batch_sizes
    ))


if __name__ == "__main__":
    main()
