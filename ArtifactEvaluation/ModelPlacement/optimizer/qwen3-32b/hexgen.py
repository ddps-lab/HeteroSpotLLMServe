"""
HexGen Evaluate: Parse HexGen GA results, then use ShuntServe's estimator
to find optimal layer partition and predict throughput for each pipeline.

HexGen GA determines: stage definitions (which GPU types, TP degrees).
ShuntServe optimizer determines: optimal layer partition + batched throughput.

This gives HexGen the benefit of ShuntServe's estimator while keeping
HexGen's stage/TP decisions. Uses only_throughput mode to ensure all
nodes in each pipeline are used.

Run: python3 evaluate.py
"""
import sys
import os
import json

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

import torch
import logging
from transformers import AutoConfig
from hexgen_optimizer import main as hexgen_main
from hardware_specs import INSTANCE_SPEC
from shuntserve_optimizer import run_test_case
from cluster_pool import ClusterPool


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


"""
Map HexGen TP degree to an equivalent ShuntServe instance type.

ShuntServe requires TP = instance gpu_count, so we map:
  TP=4 on g5.12xlarge  → (spot)g5.12xlarge       (4 GPU)
  TP=2 on g5.12xlarge  → (spot)g5.12xlarge(half)  (2 GPU)
  TP=1 on g5.12xlarge  → (spot)g5.xlarge           (1 GPU)
  TP=1 on g6e.xlarge   → (spot)g6e.xlarge          (1 GPU, already 1)
"""
_TP_TO_INSTANCE = {
    # g5.12xlarge (A10G, 4 GPU base)
    ("g5.12xlarge", 4): "(spot)g5.12xlarge",
    ("g5.12xlarge", 2): "(spot)g5.12xlarge(half)",
    ("g5.12xlarge", 1): "(spot)g5.xlarge",
    # g6.12xlarge (L4, 4 GPU base)
    ("g6.12xlarge", 4): "(spot)g6.12xlarge",
    ("g6.12xlarge", 2): "(spot)g6.12xlarge(half)",
    ("g6.12xlarge", 1): "(spot)g6.xlarge",
    # g6e.xlarge (L40S, 1 GPU base) — TP can only be 1
    ("g6e.xlarge", 1): "(spot)g6e.xlarge",
    # g6e.12xlarge (L40S, 4 GPU base)
    ("g6e.12xlarge", 4): "(spot)g6e.12xlarge",
    ("g6e.12xlarge", 2): "(spot)g6e.12xlarge(half)",
    ("g6e.12xlarge", 1): "(spot)g6e.xlarge",
}


def stages_to_cluster(stages):
    """
    Convert HexGen pipeline stages into a ClusterPool for ShuntServe optimizer.

    Each stage is mapped to a virtual instance whose gpu_count = TP degree.
    ShuntServe optimizer will then determine the optimal layer partition.
    """
    from collections import Counter
    node_counts = Counter()
    for s in stages:
        key = (s["instance_type"], s["tp_degree"])
        mapped = _TP_TO_INSTANCE.get(key)
        if mapped is None:
            raise ValueError(f"No instance mapping for {s['instance_type']} TP={s['tp_degree']}")
        node_counts[mapped] += 1

    available = dict(node_counts)
    # Dummy prices — only_throughput mode ignores cost
    prices = {k: 1.0 for k in available}
    return available, prices


def evaluate_pipeline_with_shuntserve(stages, model_config, head_dim=None):
    """
    Use ShuntServe optimizer to find optimal layer partition for a HexGen pipeline.
    
    The pipeline's stage structure (instance types) is fixed by HexGen GA.
    ShuntServe optimizer finds the best layer partition using only_throughput mode
    (all nodes must be used).
    """
    available, prices = stages_to_cluster(stages)

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

    cluster_pool = ClusterPool(
        available_spot_nodes=available,
        spot_prices=prices
    )

    max_stages = len(stages)

    results, _, opt_time = run_test_case(
        config, budget=9999, latency_slo=99999999,
        cluster_pool=cluster_pool, max_stages=max_stages, top_k=1,
        optimization_mode="only_throughput"
    )

    if not results:
        return None, opt_time

    best = results[0]
    return best, opt_time


def main():
    logging.basicConfig(level=logging.WARNING, format='%(message)s', force=True)

    model_name = "Qwen/Qwen3-32B"
    model_config = AutoConfig.from_pretrained(model_name)
    head_dim = getattr(model_config, "head_dim", None)

    print("=" * 80)
    print("HexGen Optimizer + ShuntServe Estimator Evaluation")
    print(f"Model: {model_name}")
    print(f"  hidden_size={model_config.hidden_size}, num_layers={model_config.num_hidden_layers}, "
          f"num_heads={model_config.num_attention_heads}, head_dim={head_dim}")
    print("Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    # Step 1: Run HexGen GA
    hexgen_model_config = {
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "input_len": 763,
        "output_len": 232,
        "weight_size_gb": 62,
    }
    print("\n[Step 1] Running HexGen GA optimizer...")
    best_ind, cluster, cluster_config_flatten = hexgen_main(model_config=hexgen_model_config)

    genes_all = best_ind.genes[0]
    plans_all = best_ind.plan[1][0]
    pp_partitions_all = best_ind.pp_partition[0]

    # Step 2: For each pipeline, use ShuntServe optimizer
    print(f"\n{'=' * 80}")
    print("[Step 2] ShuntServe estimator — optimal layer partition & throughput")
    print(f"{'=' * 80}")

    total_throughput = 0
    all_pipeline_results = []
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
                  f"TP={s['tp_degree']}, HexGen layers={s['layer_count']}")
        print(f"  HexGen TP strategy: {[s['tp_degree'] for s in stages]}")
        print(f"  HexGen layer partition: {[s['layer_count'] for s in stages]}")

        # Run ShuntServe optimizer on this pipeline's cluster
        best, opt_time = evaluate_pipeline_with_shuntserve(stages, model_config, head_dim=head_dim)

        if best is None:
            print(f"  ❌ No feasible pipeline found by ShuntServe optimizer")
            continue

        tp_list = [INSTANCE_SPEC[inst]['gpu_count'] for inst in best.stages]
        print(f"\n  ShuntServe optimized result ({opt_time:.1f}s):")
        print(f"    Stages: {list(best.stages)}")
        print(f"    TP strategy: {tp_list}")
        print(f"    Layer partition: {list(best.layer_per_stage)}")
        print(f"    Throughput: {best.throughput:.4f} req/s")
        print(f"    Global batch size: {best.global_batch_size}")
        print(f"    Num GPU blocks: {best.num_blocks}")
        print(f"    Single request latency: {best.single_request_latency:.2f} ms")

        total_throughput += best.throughput
        all_pipeline_results.append({
            "best": best,
            "tp_list": tp_list,
            "hexgen_stages": stages,
        })

    print(f"\n{'=' * 60}")
    print(f"Total system throughput (HexGen stages + ShuntServe estimator): {total_throughput:.4f} req/s")

    # ─── Save JSON ────────────────────────────────────────────────────────
    output_pipelines = []
    for i, pr in enumerate(all_pipeline_results, 1):
        best = pr["best"]
        tp_list = pr["tp_list"]
        hexgen_stages = pr["hexgen_stages"]
        ss_stages = [[inst, int(layers)] for inst, layers in zip(best.stages, best.layer_per_stage)]
        output_pipelines.append({
            "label": f"HX-P{i}",
            "system": "HexGen",
            "stages": ss_stages,
            "parallel_strategy": tp_list,
            "pp_layer_partition": ",".join(str(int(l)) for l in best.layer_per_stage),
            "gpu_memory_utilization": 0.85,
            "predicted_throughput_rps": best.throughput,
            "predicted_total_latency_ms": best.single_request_latency,
            "max_batch_size": best.global_batch_size,
            "num_blocks": best.num_blocks,
            "hexgen_original_stages": [
                {"instance_type": s["instance_type"], "tp_degree": s["tp_degree"], "layer_count": s["layer_count"]}
                for s in hexgen_stages
            ],
        })

    model_short = model_name.split("/")[-1]
    output = {
        "model": model_name,
        "workload": {"input_len": 763, "output_len": 232},
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "qwen3-32b", "estimated")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"predicted_hexgen_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
