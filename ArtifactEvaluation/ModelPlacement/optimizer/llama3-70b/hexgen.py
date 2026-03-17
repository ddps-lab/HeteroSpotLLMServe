"""
HexGen Evaluate (Pure): HexGen GA determines model placement (which nodes, TP degrees),
then layers are partitioned proportionally to each stage's GPU memory capacity.

This follows the HexGen paper's original approach:
  1. HexGen GA → node placement + TP degrees
  2. Memory-proportional layer partitioning (not ShuntServe DP)

Throughput is estimated using ShuntServe's estimator (get_throughput / get_global_batch_size).

Run: python3 hexgen.py
"""
import sys
import os
import json
import logging
import math

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

import torch
from transformers import AutoConfig
from hexgen_optimizer import main as hexgen_main
from hardware_specs import INSTANCE_SPEC, GPU_SPEC
from estimator_utils import get_throughput, get_global_batch_size


def resolve_stage_mapping(genes_pipeline, plan_pipeline, pp_partition, cluster_config_flatten):
    """Map HexGen GA output to per-stage (instance_type, tp_degree, layer_count)."""
    node_gpus = []
    for node_idx, gpu_count in enumerate(genes_pipeline):
        if gpu_count > 0:
            node_gpus.append({
                "node_idx": node_idx,
                "instance_type": cluster_config_flatten[node_idx],
                "remaining": gpu_count,
            })

    stages = []
    node_ptr = 0
    for stage_idx, (tp_degree, layer_count) in enumerate(zip(plan_pipeline, pp_partition)):
        while node_ptr < len(node_gpus) and node_gpus[node_ptr]["remaining"] <= 0:
            node_ptr += 1
        if node_ptr >= len(node_gpus):
            raise ValueError(f"Stage {stage_idx}: ran out of nodes")
        node = node_gpus[node_ptr]
        node["remaining"] -= tp_degree
        stages.append({
            "node_idx": node["node_idx"],
            "instance_type": node["instance_type"],
            "tp_degree": tp_degree,
            "layer_count": layer_count,
        })
    return stages


def memory_proportional_partition(stages, total_layers, gpu_mem_utilization=0.85):
    """
    Distribute layers across stages proportionally to each stage's GPU memory capacity.

    Each stage's capacity = GPU memory (MB) × tp_degree × gpu_mem_utilization.
    Layers are allocated proportionally and rounded, with remainder correction
    applied to the stage with the largest fractional part.
    """
    capacities = []
    for s in stages:
        gpu_type = INSTANCE_SPEC[s["instance_type"]]["gpu_type"]
        mem_mb = GPU_SPEC[gpu_type]["memory_size"]
        cap = mem_mb * s["tp_degree"] * gpu_mem_utilization
        capacities.append(cap)

    total_cap = sum(capacities)
    ratios = [c / total_cap for c in capacities]

    # Compute fractional layer counts
    raw = [r * total_layers for r in ratios]
    floored = [math.floor(v) for v in raw]
    remainders = [v - f for v, f in zip(raw, floored)]

    # Distribute the remaining layers to stages with largest fractional parts
    deficit = total_layers - sum(floored)
    indices_by_remainder = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)
    for i in range(deficit):
        floored[indices_by_remainder[i]] += 1

    # Ensure no stage has 0 layers
    for i in range(len(floored)):
        if floored[i] == 0:
            # Steal from the stage with the most layers
            donor = max(range(len(floored)), key=lambda j: floored[j])
            floored[donor] -= 1
            floored[i] = 1

    assert sum(floored) == total_layers, f"Layer sum mismatch: {sum(floored)} != {total_layers}"
    return floored


def estimate_pipeline_throughput(stages, layer_partition, model_config, head_dim=None):
    """
    Estimate throughput for a pipeline using ShuntServe's estimator functions.

    Builds node_layer_comb from stages + layer_partition, then calls
    get_global_batch_size and get_throughput.
    """
    node_layer_comb = []
    tp_sizes = []
    for s, layers in zip(stages, layer_partition):
        node_layer_comb.append((s["instance_type"], "us-east-1a", layers))
        tp_sizes.append(s["tp_degree"])

    input_len = 763
    output_len = 232

    try:
        mbs, num_blocks = get_global_batch_size(
            avg_input_len=input_len,
            avg_output_len=output_len,
            max_model_len=8192,
            hidden_dim=model_config.hidden_size,
            num_attention_head=model_config.num_attention_heads,
            num_kv_cache_head=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
            total_num_layers=model_config.num_hidden_layers,
            vocab_size=model_config.vocab_size,
            intermediate_dim=model_config.intermediate_size,
            gpu_mem_utilization=0.85,
            node_layer_comb=node_layer_comb,
            dtype=torch.float16,
            tp_sizes=tp_sizes,
            head_dim=head_dim,
        )
    except Exception as e:
        print(f"  ❌ get_global_batch_size failed: {e}")
        return None

    if mbs <= 0:
        print(f"  ❌ max_batch_size = {mbs} (infeasible)")
        return None

    try:
        tput, latency, _ = get_throughput(
            avg_input_len=input_len,
            avg_output_len=output_len,
            max_model_len=8192,
            hidden_dim=model_config.hidden_size,
            num_attention_head=model_config.num_attention_heads,
            num_kv_cache_head=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
            total_num_layers=model_config.num_hidden_layers,
            vocab_size=model_config.vocab_size,
            intermediate_dim=model_config.intermediate_size,
            gpu_mem_utilization=0.85,
            node_layer_comb=node_layer_comb,
            dtype=torch.float16,
            tp_sizes=tp_sizes,
            batch_override=(mbs, num_blocks),
            head_dim=head_dim,
        )
    except Exception as e:
        print(f"  ❌ get_throughput failed: {e}")
        return None

    return {
        "throughput": tput,
        "latency_ms": latency,
        "max_batch_size": int(mbs),
        "num_blocks": num_blocks,
        "tp_sizes": tp_sizes,
        "layer_partition": layer_partition,
    }


def main():
    logging.basicConfig(level=logging.WARNING, format='%(message)s', force=True)

    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)
    head_dim = getattr(model_config, "head_dim", None)

    print("=" * 80)
    print("HexGen Optimizer (Pure: HexGen placement + memory-proportional partition)")
    print(f"Model: {model_name}")
    print("Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    # Step 1: Run HexGen GA
    print("\n[Step 1] Running HexGen GA optimizer...")
    best_ind, cluster, cluster_config_flatten = hexgen_main()

    genes_all = best_ind.genes[0]
    plans_all = best_ind.plan[1][0]
    pp_partitions_all = best_ind.pp_partition[0]

    # Step 2: Memory-proportional layer partition + throughput estimation
    print(f"\n{'=' * 80}")
    print("[Step 2] Memory-proportional layer partition + throughput estimation")
    print(f"{'=' * 80}")

    total_throughput = 0
    output_pipelines = []
    pipeline_count = 0

    for pipeline_idx in range(len(genes_all)):
        stages = resolve_stage_mapping(
            genes_all[pipeline_idx], plans_all[pipeline_idx],
            pp_partitions_all[pipeline_idx], cluster_config_flatten
        )

        print(f"\n{'─' * 60}")
        print(f"Pipeline {pipeline_idx + 1} (from HexGen GA):")
        print(f"  HexGen stages:")
        for j, s in enumerate(stages):
            print(f"    Stage {j}: {s['instance_type']} (Node {s['node_idx']}), "
                  f"TP={s['tp_degree']}, GA layers={s['layer_count']}")

        # Memory-proportional layer partition
        layer_partition = memory_proportional_partition(
            stages, model_config.num_hidden_layers
        )

        print(f"  Memory-proportional partition: {layer_partition}")
        print(f"  TP strategy: {[s['tp_degree'] for s in stages]}")

        # Estimate throughput
        result = estimate_pipeline_throughput(stages, layer_partition, model_config, head_dim)

        if result is None or result["throughput"] <= 0:
            print(f"  ❌ Infeasible pipeline (OOM or negative throughput)")
            continue

        pipeline_count += 1
        total_throughput += result["throughput"]

        print(f"\n  Estimated result:")
        print(f"    Layer partition: {result['layer_partition']}")
        print(f"    Throughput: {result['throughput']:.4f} req/s")
        print(f"    Latency: {result['latency_ms']:.2f} ms")
        print(f"    Max batch size: {result['max_batch_size']}")
        print(f"    Num GPU blocks: {result['num_blocks']}")

        # Build stages list for JSON output
        stages_list = []
        for s, layers in zip(stages, layer_partition):
            stages_list.append([s["instance_type"], int(layers)])

        output_pipelines.append({
            "label": f"HX-P{pipeline_count}",
            "system": "HexGen",
            "mode": "hexgen",
            "stages": stages_list,
            "parallel_strategy": result["tp_sizes"],
            "pp_layer_partition": ",".join(str(l) for l in layer_partition),
            "gpu_memory_utilization": 0.85,
            "predicted_throughput_rps": result["throughput"],
            "predicted_total_latency_ms": result["latency_ms"],
            "max_batch_size": result["max_batch_size"],
            "num_blocks": result["num_blocks"],
            "hexgen_original_stages": [
                {
                    "node_idx": s["node_idx"],
                    "instance_type": s["instance_type"],
                    "tp_degree": s["tp_degree"],
                    "layer_count": s["layer_count"],
                }
                for s in stages
            ],
        })

    print(f"\n{'=' * 60}")
    print(f"Total system throughput: {total_throughput:.4f} req/s")
    print(f"Pipelines: {pipeline_count}")

    # Save results
    output = {
        "model": model_name,
        "workload": {"input_len": 763, "output_len": 232},
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "llama3-70b", "estimated")
    os.makedirs(results_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = os.path.join(results_dir, f"predicted_hexgen_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
