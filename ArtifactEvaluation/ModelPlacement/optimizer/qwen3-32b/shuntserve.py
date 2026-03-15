"""
ShuntServe Optimizer for Qwen3-32B: iteratively finds pipelines until cluster is exhausted.
Each iteration shows top-N candidates, selects the best, subtracts resources, repeats.
Run: python3 optimizer.py
"""
import sys
import os
import json
import logging

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

import torch
from transformers import AutoConfig
from collections import Counter
from shuntserve_optimizer import run_test_case, Pipeline
from cluster_pool import ClusterPool
from hardware_specs import INSTANCE_SPEC


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

    model_name = "Qwen/Qwen3-32B"
    model_config = AutoConfig.from_pretrained(model_name)

    head_dim = getattr(model_config, "head_dim", None) or (model_config.hidden_size // model_config.num_attention_heads)

    config = {
        "expected_input_len": 763,
        "expected_output_len": 232,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        "intermediate_size": model_config.intermediate_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.85,
        "head_dim": head_dim,
    }

    available_nodes = {
        "(spot)g5.12xlarge": 2,
        "(spot)g6.12xlarge": 3,
        "(spot)g6e.xlarge": 4,
    }

    prices = {
        "(spot)g5.12xlarge": 2.2915,
        "(spot)g6.12xlarge": 1.9445,
        "(spot)g6e.xlarge": 0.7040,
    }

    TOP_K = 5

    print("=" * 80)
    print("ShuntServe Optimizer — Iterative Pipeline Search")
    print(f"Model: {model_name}")
    print(f"  hidden_size={model_config.hidden_size}, num_layers={model_config.num_hidden_layers}, "
          f"num_heads={model_config.num_attention_heads}, head_dim={head_dim}")
    print(f"Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print(f"Beam width (top_k): {TOP_K}")
    print("=" * 80)

    remaining_nodes = dict(available_nodes)
    all_pipelines = []
    iteration = 0

    while True:
        iteration += 1
        if all(v == 0 for v in remaining_nodes.values()):
            print(f"\n✓ All resources exhausted.")
            break

        print(f"\n{'━' * 70}")
        print(f"Iteration {iteration}: Remaining resources:")
        for inst, count in remaining_nodes.items():
            if count > 0:
                print(f"  {inst}: {count}")
        print(f"{'━' * 70}")

        cluster_pool = ClusterPool(
            available_spot_nodes=remaining_nodes,
            spot_prices=prices
        )

        results, optimizer, opt_time = run_test_case(
            config, budget=9999, latency_slo=99999999,
            cluster_pool=cluster_pool, max_stages=13, top_k=TOP_K,
            optimization_mode="soft_slo"
        )

        if not results:
            print(f"  No feasible pipeline found with remaining resources.")
            break

        print(f"\n  Found {len(results)} candidate(s) (optimization: {opt_time:.1f}s):")
        for rank, pipeline in enumerate(results, 1):
            tp_list = [INSTANCE_SPEC[inst]['gpu_count'] for inst in pipeline.stages]
            used = Counter(pipeline.stages)
            instances_str = ", ".join(f"{k}×{v}" for k, v in sorted(used.items()))
            ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
            print(f"\n  {ordinal}:")
            print(f"    Instances: {instances_str}")
            print(f"    TP: {tp_list}, Layers: {list(pipeline.layer_per_stage)}")
            print(f"    Throughput: {pipeline.throughput:.3f} req/s, "
                  f"Batch size: {pipeline.global_batch_size}, "
                  f"GPU blocks: {pipeline.num_blocks}")

        best = results[0]
        all_pipelines.append(best)
        print(f"\n  ✓ Selected 1st → Pipeline {iteration}")

        used = Counter(best.stages)
        for inst, count in used.items():
            if inst in remaining_nodes:
                remaining_nodes[inst] -= count

    # ─── Final Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULT: {len(all_pipelines)} pipeline(s)")
    print(f"{'=' * 70}")

    total_throughput = 0
    for i, pipeline in enumerate(all_pipelines, 1):
        tp_list = [INSTANCE_SPEC[inst]['gpu_count'] for inst in pipeline.stages]
        used = Counter(pipeline.stages)
        instances_str = ", ".join(f"{k}×{v}" for k, v in sorted(used.items()))
        print(f"\nPipeline {i}:")
        print(f"  Instances: {instances_str}")
        print(f"  TP strategy: {tp_list}")
        print(f"  Layer partition: {list(pipeline.layer_per_stage)}")
        print(f"  Throughput: {pipeline.throughput:.3f} req/s")
        print(f"  Global batch size: {pipeline.global_batch_size}")
        print(f"  Num GPU blocks: {pipeline.num_blocks}")
        total_throughput += pipeline.throughput

    print(f"\n{'─' * 40}")
    print(f"Total system throughput: {total_throughput:.3f} req/s")

    unused = {k: v for k, v in remaining_nodes.items() if v > 0}
    if unused:
        print(f"Unused resources: {unused}")

    # ─── Save JSON ────────────────────────────────────────────────────────
    output_pipelines = []
    for i, pipeline in enumerate(all_pipelines, 1):
        tp_list = [INSTANCE_SPEC[inst]['gpu_count'] for inst in pipeline.stages]
        used = Counter(pipeline.stages)
        instances_str = ", ".join(f"{k}×{v}" for k, v in sorted(used.items()))
        output_pipelines.append({
            "label": f"SS-P{i}",
            "system": "ShuntServe",
            "stages": [[inst, int(layers)] for inst, layers in zip(pipeline.stages, pipeline.layer_per_stage)],
            "parallel_strategy": tp_list,
            "pp_layer_partition": ",".join(str(int(l)) for l in pipeline.layer_per_stage),
            "gpu_memory_utilization": config["gpu_mem_utilization"],
            "predicted_throughput_rps": pipeline.throughput,
            "predicted_total_latency_ms": pipeline.latency_per_global_batch,
            "max_batch_size": pipeline.global_batch_size,
            "num_blocks": pipeline.num_blocks,
        })

    model_short = model_name.split("/")[-1]
    output = {
        "model": model_name,
        "workload": {"input_len": config["expected_input_len"], "output_len": config["expected_output_len"]},
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "qwen3-32b", "estimated")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"predicted_shuntserve_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
