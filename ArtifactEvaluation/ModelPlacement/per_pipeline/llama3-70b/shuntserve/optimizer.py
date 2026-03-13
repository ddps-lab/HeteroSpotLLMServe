"""
ShuntServe Optimizer: iteratively finds pipelines until cluster is exhausted.
Each iteration shows top-N candidates, selects the best, subtracts resources, repeats.
Run: python3 optimizer.py
"""
import sys
import os
import copy
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
    # Enable progress logging
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

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


    TOP_K = 5  # Beam width & number of candidates to show per iteration

    print("=" * 80)
    print("ShuntServe Optimizer — Iterative Pipeline Search")
    print(f"Model: {model_name}")
    print(f"Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print(f"Beam width (top_k): {TOP_K}")
    print("=" * 80)

    remaining_nodes = dict(available_nodes)
    all_pipelines = []
    iteration = 0

    while True:
        iteration += 1
        # Check if any nodes remain
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

        # Treat infeasible results (throughput <= 0) as no results
        if results and results[0].throughput <= 0:
            print(f"  Best candidate is infeasible (throughput={results[0].throughput:.3f}). Treating as no feasible pipeline.")
            results = []

        if not results:
            # ─── Residual resource merging ────────────────────────────────
            # Remaining nodes can't form an independent pipeline.
            # Merge them into the last pipeline and re-optimize for throughput.
            if all_pipelines and any(v > 0 for v in remaining_nodes.values()):
                residual = {k: v for k, v in remaining_nodes.items() if v > 0}
                print(f"  No independent pipeline feasible. Merging residual {residual} into Pipeline {iteration - 1}.")

                # Reconstruct the cluster for the last pipeline + residual nodes
                last_pipeline = all_pipelines[-1]
                last_used = Counter(last_pipeline.stages)

                merged_nodes = dict(remaining_nodes)  # start with residual
                for inst, count in last_used.items():
                    merged_nodes[inst] = merged_nodes.get(inst, 0) + count

                print(f"  Merged cluster for re-optimization:")
                for inst, count in merged_nodes.items():
                    if count > 0:
                        print(f"    {inst}: {count}")

                merged_pool = ClusterPool(
                    available_spot_nodes=merged_nodes,
                    spot_prices=prices
                )

                merged_results, _, merged_opt_time = run_test_case(
                    config, budget=9999, latency_slo=99999999,
                    cluster_pool=merged_pool, max_stages=13, top_k=TOP_K,
                    optimization_mode="only_throughput"
                )

                if merged_results:
                    merged_best = merged_results[0]
                    tp_list_old = [INSTANCE_SPEC[inst]['gpu_count'] for inst in last_pipeline.stages]
                    tp_list_new = [INSTANCE_SPEC[inst]['gpu_count'] for inst in merged_best.stages]
                    print(f"\n  Re-optimized Pipeline {iteration - 1} (optimization: {merged_opt_time:.1f}s):")
                    print(f"    Before: TP={tp_list_old}, Layers={list(last_pipeline.layer_per_stage)}, "
                          f"Throughput={last_pipeline.throughput:.3f} req/s")
                    print(f"    After:  TP={tp_list_new}, Layers={list(merged_best.layer_per_stage)}, "
                          f"Throughput={merged_best.throughput:.3f} req/s")

                    # Replace the last pipeline
                    all_pipelines[-1] = merged_best

                    # All residual resources consumed
                    for k in remaining_nodes:
                        remaining_nodes[k] = 0

                    # Also subtract the merged pipeline's usage from the conceptual pool
                    # (already zeroed above, so just for bookkeeping)
                    print(f"  ✓ Residual nodes merged into Pipeline {iteration - 1}")
                else:
                    print(f"  ⚠ Re-optimization with merged resources also failed.")
            else:
                print(f"  No feasible pipeline found with remaining resources.")
            break

        # Show all candidates
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

        # Select the best (1st)
        best = results[0]
        all_pipelines.append(best)

        tp_list = [INSTANCE_SPEC[inst]['gpu_count'] for inst in best.stages]
        print(f"\n  ✓ Selected 1st → Pipeline {iteration}")

        # Subtract used resources
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
        print(f"  pp_layer_partition: \"{','.join(str(l) for l in pipeline.layer_per_stage)}\"")
        print(f"  parallel_strategy: {tp_list}")
        print(f"  Throughput: {pipeline.throughput:.3f} req/s")
        print(f"  Global batch size: {pipeline.global_batch_size}")
        print(f"  Num GPU blocks: {pipeline.num_blocks}")
        total_throughput += pipeline.throughput

    print(f"\n{'─' * 40}")
    print(f"Total system throughput: {total_throughput:.3f} req/s")

    unused = {k: v for k, v in remaining_nodes.items() if v > 0}
    if unused:
        print(f"Unused resources: {unused}")

    # ─── Save results to JSON ────────────────────────────────────────────
    output_pipelines = []
    for i, pipeline in enumerate(all_pipelines, 1):
        tp_list = [INSTANCE_SPEC[inst]['gpu_count'] for inst in pipeline.stages]
        stages_list = [[inst, int(layers)] for inst, layers in zip(pipeline.stages, pipeline.layer_per_stage)]
        output_pipelines.append({
            "label": f"SS-P{i}",
            "system": "ShuntServe",
            "stages": stages_list,
            "parallel_strategy": tp_list,
            "pp_layer_partition": ",".join(str(int(l)) for l in pipeline.layer_per_stage),
            "gpu_memory_utilization": config["gpu_mem_utilization"],
            "predicted_throughput_rps": pipeline.throughput,
            "predicted_total_latency_ms": pipeline.latency_per_global_batch,
            "max_batch_size": pipeline.global_batch_size,
            "num_blocks": pipeline.num_blocks,
        })

    output = {
        "model": model_name,
        "workload": {
            "input_len": config["expected_input_len"],
            "output_len": config["expected_output_len"],
        },
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = os.path.join(results_dir, f"predicted_shuntserve_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
