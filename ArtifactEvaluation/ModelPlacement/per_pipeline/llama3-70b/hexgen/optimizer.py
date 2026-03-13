"""
HexGen Evaluate: HexGen GA determines model placement (which nodes, TP degrees),
then ShuntServe's DP optimizer determines the optimal layer partition.

HexGen GA determines: stage definitions (which GPU types, TP degrees per pipeline)
ShuntServe optimizer: finds optimal layer partition for the given nodes (only_throughput mode)

Run: python3 optimizer.py
"""
import sys
import os
import json
import logging
from collections import Counter

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

import torch
from transformers import AutoConfig
from hexgen_optimizer import main as hexgen_main
from hardware_specs import INSTANCE_SPEC
from shuntserve_optimizer import run_test_case
from cluster_pool import ClusterPool


"""
Map HexGen TP degree to an equivalent ShuntServe instance type.
"""
_TP_TO_INSTANCE = {
    ("g5.12xlarge", 4): "(spot)g5.12xlarge",
    ("g5.12xlarge", 2): "(spot)g5.12xlarge(half)",
    ("g5.12xlarge", 1): "(spot)g5.xlarge",
    ("g6.12xlarge", 4): "(spot)g6.12xlarge",
    ("g6.12xlarge", 2): "(spot)g6.12xlarge(half)",
    ("g6.12xlarge", 1): "(spot)g6.xlarge",
    ("g6e.xlarge", 1): "(spot)g6e.xlarge",
    ("g6e.12xlarge", 4): "(spot)g6e.12xlarge",
    ("g6e.12xlarge", 2): "(spot)g6e.12xlarge(half)",
    ("g6e.12xlarge", 1): "(spot)g6e.xlarge",
}

PRICES = {
    "(spot)g5.12xlarge": 2.2915,
    "(spot)g5.12xlarge(half)": 2.2915 / 2,
    "(spot)g5.xlarge": 2.2915 / 4,
    "(spot)g6.12xlarge": 1.9445,
    "(spot)g6.12xlarge(half)": 1.9445 / 2,
    "(spot)g6.xlarge": 1.9445 / 4,
    "(spot)g6e.xlarge": 0.7040,
    "(spot)g6e.12xlarge": 0.7040 * 4,
    "(spot)g6e.12xlarge(half)": 0.7040 * 2,
}


def resolve_stage_mapping(genes_pipeline, plan_pipeline, pp_partition, cluster_config_flatten):
    """Map HexGen stages to (instance_type, tp_degree, layer_count)."""
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


def main():
    logging.basicConfig(level=logging.WARNING, format='%(message)s', force=True)

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

    print("=" * 80)
    print("HexGen Optimizer (HexGen placement + ShuntServe layer partition)")
    print(f"Model: {model_name}")
    print("Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    # Step 1: Run HexGen GA
    print("\n[Step 1] Running HexGen GA optimizer...")
    best_ind, cluster, cluster_config_flatten = hexgen_main()

    genes_all = best_ind.genes[0]
    plans_all = best_ind.plan[1][0]
    pp_partitions_all = best_ind.pp_partition[0]

    # Step 2: For each pipeline, extract nodes and use ShuntServe's optimizer
    print(f"\n{'=' * 80}")
    print("[Step 2] Apply ShuntServe layer optimization to HexGen's placement")
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
                  f"TP={s['tp_degree']}, layers={s['layer_count']}")
        print(f"  TP strategy: {[s['tp_degree'] for s in stages]}")
        print(f"  Layer partition (GA original): {[s['layer_count'] for s in stages]}")

        # Convert HexGen stages to ShuntServe instance types and count them
        ss_nodes = Counter()
        for s in stages:
            key = (s["instance_type"], s["tp_degree"])
            mapped_inst = _TP_TO_INSTANCE.get(key)
            if mapped_inst is None:
                print(f"  ❌ No instance mapping for {s['instance_type']} TP={s['tp_degree']}")
                continue
            ss_nodes[mapped_inst] += 1

        print(f"  ShuntServe equivalent nodes: {dict(ss_nodes)}")

        # Create cluster pool with these specific nodes
        cluster_pool = ClusterPool(
            available_spot_nodes=dict(ss_nodes),
            spot_prices=PRICES
        )

        # Run ShuntServe optimizer (only_throughput mode) for layer partitioning
        results, optimizer, opt_time = run_test_case(
            config, budget=9999, latency_slo=99999999,
            cluster_pool=cluster_pool, max_stages=len(stages) + 2,
            top_k=3, optimization_mode="only_throughput"
        )

        if not results or results[0].throughput <= 0:
            print(f"  ❌ ShuntServe optimizer found no feasible partition for these nodes")
            continue

        best = results[0]
        tp_list = [INSTANCE_SPEC[inst]['gpu_count'] for inst in best.stages]

        print(f"\n  ShuntServe optimized partition (optimization: {opt_time:.1f}s):")
        print(f"    Stages: {list(best.stages)}")
        print(f"    TP strategy: {tp_list}")
        print(f"    Layer partition: {list(best.layer_per_stage)}")
        print(f"    Throughput: {best.throughput:.4f} req/s")
        print(f"    Global batch size: {best.global_batch_size}")
        print(f"    Num GPU blocks: {best.num_blocks}")

        total_throughput += best.throughput
        pipeline_count += 1

        stages_list = [[inst, int(layers)] for inst, layers in zip(best.stages, best.layer_per_stage)]
        output_pipelines.append({
            "label": f"HX-P{pipeline_count}",
            "system": "HexGen",
            "stages": stages_list,
            "parallel_strategy": tp_list,
            "pp_layer_partition": ",".join(str(int(l)) for l in best.layer_per_stage),
            "gpu_memory_utilization": 0.85,
            "predicted_throughput_rps": best.throughput,
            "predicted_total_latency_ms": best.latency_per_global_batch,
            "max_batch_size": best.global_batch_size,
            "num_blocks": best.num_blocks,
            "hexgen_original_stages": [
                {"instance_type": s["instance_type"], "tp_degree": s["tp_degree"], "layer_count": s["layer_count"]}
                for s in stages
            ],
        })

    print(f"\n{'=' * 60}")
    print(f"Total system throughput: {total_throughput:.4f} req/s")

    # Save results
    output = {
        "model": model_name,
        "workload": {"input_len": 763, "output_len": 232},
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = os.path.join(results_dir, f"predicted_hexgen_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
