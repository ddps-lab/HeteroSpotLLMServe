"""
Performance Estimation — Batch runner

Runs the ShuntServe analytical estimator across all (instance, strategy) combinations
for a given model, sweeping batch sizes. Results are cached to JSON so they are only
computed once.

Usage:
    python estimate.py                          # all models
    python estimate.py --model llama3-70b       # single model
    python estimate.py --model qwen3-4b --force # recompute even if cached
"""
import argparse
import json
import os
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

import torch
import logging
from estimator_utils import get_global_batch_size, get_throughput, get_single_request_latency
from hardware_specs import INSTANCE_SPEC, GPU_SPEC

# ─── Instance types and parallelization strategies ────────────────────
INSTANCES = [
    "g5.48xlarge",
    "g6.48xlarge",
    "g6e.48xlarge",
    "p4d.24xlarge",
    "p5.48xlarge",
]

# Each strategy: (label, parallel_strategy list)
# parallel_strategy[i] = TP degree at stage i; len = PP degree
STRATEGIES = [
    ("tp1_pp8", [1, 1, 1, 1, 1, 1, 1, 1]),
    ("tp2_pp4", [2, 2, 2, 2]),
    ("tp4_pp2", [4, 4]),
    ("tp8_pp1", [8]),
]

# ─── Model configurations ────────────────────────────────────────────
MODELS = {
    "llama3-70b": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "head_dim": 128,
    },
    "qwen3-32b": {
        "model_name": "Qwen/Qwen3-32B",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "head_dim": 128,
    },
}

WORKLOAD = {
    "input_len": 763,
    "output_len": 232,
    "max_model_len": 8192,
}


def workload_dirname(workload: dict) -> str:
    """Generate directory name from workload config: in763-out232"""
    return f"in{workload['input_len']}-out{workload['output_len']}"


def instance_to_dirname(instance_type: str) -> str:
    """Convert instance type to directory-safe name: g5.48xlarge -> g5_48xlarge"""
    return instance_type.replace(".", "_")


def build_node_layer_comb(instance_type: str, num_layers: int, strategy: list) -> list:
    """
    Build node_layer_comb for the estimator.

    Each entry represents one pipeline stage:
        (instance_type, availability_zone, num_layers_in_stage)

    Layers are divided as evenly as possible across PP stages.
    """
    pp_size = len(strategy)
    base_layers = num_layers // pp_size
    remainder = num_layers % pp_size

    comb = []
    for i in range(pp_size):
        # Distribute remainder to earlier stages
        stage_layers = base_layers + (1 if i < remainder else 0)
        comb.append((instance_type, "us-east-1a", stage_layers))

    return comb


def run_estimation(model_key: str, instance_type: str, strategy_label: str,
                   strategy: list, force: bool = False) -> dict:
    """
    Run estimation for one (model, instance, strategy) combination.
    Returns result dict and saves to JSON.
    """
    model = MODELS[model_key]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wl_dir = workload_dirname(WORKLOAD)
    results_dir = os.path.join(base_dir, model_key, wl_dir, "results", "data", "estimated")
    os.makedirs(results_dir, exist_ok=True)

    # Also output to trtllm/predicted/ directory
    trtllm_dir = os.path.join(base_dir, "trtllm", model_key, wl_dir, "predicted")
    os.makedirs(trtllm_dir, exist_ok=True)
    trtllm_file = os.path.join(trtllm_dir, f"est_{instance_to_dirname(instance_type)}_{strategy_label}.json")

    output_file = os.path.join(results_dir, f"est_{instance_to_dirname(instance_type)}_{strategy_label}.json")

    # Check cache
    if os.path.exists(output_file) and not force:
        with open(output_file) as f:
            cached = json.load(f)
        # Ensure trtllm/predicted/ also has the cached result
        if not os.path.exists(trtllm_file):
            with open(trtllm_file, "w") as f:
                json.dump(cached, f, indent=2)
        print(f"  [cached] {instance_type} {strategy_label}")
        return cached

    num_layers = model["num_hidden_layers"]
    node_layer_comb = build_node_layer_comb(instance_type, num_layers, strategy)
    pp_layer_partition = [c[2] for c in node_layer_comb]

    gpu_type = INSTANCE_SPEC[instance_type]["gpu_type"]
    gpu_count = INSTANCE_SPEC[instance_type]["gpu_count"]
    tp_size = strategy[0]  # TP degree (same for all stages)
    pp_size = len(strategy)

    # tp_sizes: per-stage TP degree from strategy list
    # This ensures that e.g. tp4_pp2 on g5.48xlarge uses 4 GPUs per stage,
    # not all 8 GPUs of the instance (which would overestimate memory capacity).
    tp_sizes_list = strategy  # e.g. [4, 4] for tp4_pp2

    common_kwargs = dict(
        avg_input_len=WORKLOAD["input_len"],
        avg_output_len=WORKLOAD["output_len"],
        hidden_dim=model["hidden_size"],
        num_attention_head=model["num_attention_heads"],
        num_kv_cache_head=model["num_key_value_heads"],
        total_num_layers=num_layers,
        vocab_size=model["vocab_size"],
        intermediate_dim=model["intermediate_size"],
        dtype=torch.float16,
        head_dim=model["head_dim"],
    )

    # 1) Get max batch size and blocks
    batch_size, num_blocks = get_global_batch_size(
        max_model_len=WORKLOAD["max_model_len"],
        gpu_mem_utilization=0.85,
        node_layer_comb=node_layer_comb,
        tp_sizes=tp_sizes_list,
        **common_kwargs,
    )
    batch_size = int(batch_size)
    num_blocks = int(num_blocks)

    # 2) Get throughput at max batch
    throughput = 0.0
    total_latency = 0.0
    if batch_size > 0:
        throughput, total_latency, _ = get_throughput(
            max_model_len=WORKLOAD["max_model_len"],
            gpu_mem_utilization=0.85,
            node_layer_comb=node_layer_comb,
            tp_sizes=tp_sizes_list,
            **common_kwargs,
        )

    # 3) Get single request latency
    single_latency = 0.0
    if batch_size > 0:
        single_latency = get_single_request_latency(
            node_layer_comb=node_layer_comb,
            tp_sizes=tp_sizes_list,
            **common_kwargs,
        )

    # 4) Sweep batch sizes: 1, 2, 4, 8, 16, 32, ... up to max
    batch_sweep = []
    if batch_size > 0:
        sweep_sizes = []
        bs = 1
        while bs < batch_size:
            sweep_sizes.append(bs)
            bs *= 2
        sweep_sizes.append(batch_size)  # always include max

        for bs in sweep_sizes:
            bs_throughput, bs_latency, _, latency_detail = get_throughput(
                max_model_len=WORKLOAD["max_model_len"],
                gpu_mem_utilization=0.85,
                node_layer_comb=node_layer_comb,
                tp_sizes=tp_sizes_list,
                batch_override=(bs, 0),
                detail=True,
                **common_kwargs,
            )
            prefill_ms = latency_detail["prefill_latency_ms"]
            decode_ms = latency_detail["decode_latency_ms"]

            # TPOT = decode_latency / (output_len * batch_size)
            # (total decode time spread across all tokens of all requests)
            tpot_ms = decode_ms / (WORKLOAD["output_len"] * bs) if bs > 0 else 0

            batch_sweep.append({
                "batch_size": bs,
                "throughput_rps": bs_throughput,
                "batch_latency_ms": bs_latency,
                "ttft_ms": prefill_ms,
                "tpot_ms": tpot_ms,
                "prefill_latency_ms": prefill_ms,
                "decode_latency_ms": decode_ms,
            })

    result = {
        "model": model["model_name"],
        "model_key": model_key,
        "instance_type": instance_type,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "strategy_label": strategy_label,
        "parallel_strategy": strategy,
        "pp_layer_partition": pp_layer_partition,
        "pp_layer_partition_str": ",".join(str(l) for l in pp_layer_partition),
        "workload": WORKLOAD,
        "max_batch_size": batch_size,
        "num_blocks": num_blocks,
        "estimated_throughput_rps": throughput,
        "estimated_batch_latency_ms": total_latency,
        "estimated_single_latency_ms": single_latency,
        "gpu_memory_utilization": 0.85,
        "batch_sweep": batch_sweep,
        "feasible": batch_size > 0,
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    # Copy to trtllm/predicted/
    with open(trtllm_file, "w") as f:
        json.dump(result, f, indent=2)

    status = "✓" if batch_size > 0 else "✗ infeasible"
    print(f"  [{status}] {instance_type} {strategy_label}: "
          f"batch={batch_size}, throughput={throughput:.4f} req/s, "
          f"latency={total_latency:.1f}ms")

    return result


def run_all_for_model(model_key: str, force: bool = False):
    """Run estimation for all (instance, strategy) combinations for a model."""
    model = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"  {model_key}: {model['model_name']}")
    print(f"  Layers={model['num_hidden_layers']}, Hidden={model['hidden_size']}, "
          f"HeadDim={model['head_dim']}")
    print(f"  Workload: input={WORKLOAD['input_len']}, output={WORKLOAD['output_len']}")
    print(f"{'='*70}")

    all_results = []
    for instance in INSTANCES:
        print(f"\n  --- {instance} ({INSTANCE_SPEC[instance]['gpu_type']} ×{INSTANCE_SPEC[instance]['gpu_count']}) ---")
        for label, strategy in STRATEGIES:
            logging.disable(logging.DEBUG)
            result = run_estimation(model_key, instance, label, strategy, force=force)
            all_results.append(result)

    # Save combined summary
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wl_dir = workload_dirname(WORKLOAD)
    summary_file = os.path.join(base_dir, model_key, wl_dir, "results", "data", "estimated", "estimation_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "model": model["model_name"],
            "model_key": model_key,
            "workload": WORKLOAD,
            "instances": INSTANCES,
            "strategies": [(l, s) for l, s in STRATEGIES],
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Summary saved to {summary_file}")

    # Print summary table
    print(f"\n  {'Instance':<18} {'Strategy':<10} {'Feasible':<10} {'MaxBatch':<10} "
          f"{'Throughput':<14} {'BatchLat(ms)':<14} {'SingleLat(ms)':<14}")
    print(f"  {'-'*94}")
    for r in all_results:
        print(f"  {r['instance_type']:<18} {r['strategy_label']:<10} "
              f"{'✓' if r['feasible'] else '✗':<10} "
              f"{r['max_batch_size']:<10} "
              f"{r['estimated_throughput_rps']:<14.4f} "
              f"{r['estimated_batch_latency_ms']:<14.1f} "
              f"{r['estimated_single_latency_ms']:<14.1f}")


def main():
    parser = argparse.ArgumentParser(description="Run performance estimation")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODELS.keys()),
                        help="Model to estimate (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cached results exist")
    args = parser.parse_args()

    if args.model:
        run_all_for_model(args.model, force=args.force)
    else:
        for model_key in MODELS:
            run_all_for_model(model_key, force=args.force)


if __name__ == "__main__":
    main()
