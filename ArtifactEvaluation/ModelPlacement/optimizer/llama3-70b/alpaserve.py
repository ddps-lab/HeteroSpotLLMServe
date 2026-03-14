"""
AlpaServe Optimizer: uses AlpaServe's original placement algorithm to decide
the optimal pipeline configuration for each homogeneous instance group.
Run: python3 optimizer.py
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
from transformers import AutoConfig
from alpaserve_optimizer import AlpaServeOptimizer
from hardware_specs import INSTANCE_SPEC
from estimator_utils import get_throughput, get_global_batch_size


def run_alpaserve_optimizer(instance_type, num_instances, model_name, model_config,
                            head_dim=None, gpu_mem_utilization=0.85):
    """
    Run AlpaServe placement for a homogeneous group.
    AlpaServe may choose PP < num_instances (replication).
    Returns a list of pipeline results (one per replica).
    """
    gpu_type = INSTANCE_SPEC[instance_type]['gpu_type']
    tp_size = INSTANCE_SPEC[instance_type]['gpu_count']
    interconnect_bandwidth = INSTANCE_SPEC[instance_type]['interconnect_bandwidth']

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
        "gpu_mem_utilization": gpu_mem_utilization,
        "tp_size": tp_size,
        "p2p_bandwidth": interconnect_bandwidth,
    }
    if head_dim is not None:
        config["head_dim"] = head_dim

    optimizer = AlpaServeOptimizer(gpu_type, num_instances, config)
    optimal_latency, layers_per_stage = optimizer.optimize()

    # AlpaServe may choose PP < num_instances (model fits on fewer GPUs → replication)
    actual_pp = len(layers_per_stage)
    num_replicas = num_instances // actual_pp
    unused_instances = num_instances % actual_pp

    # Compute throughput for one pipeline
    node_layer_comb = [(instance_type, "us-east-1a", int(l)) for l in layers_per_stage]
    throughput_kwargs = dict(
        avg_input_len=763, avg_output_len=232, max_model_len=8192,
        hidden_dim=model_config.hidden_size,
        num_attention_head=model_config.num_attention_heads,
        num_kv_cache_head=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        total_num_layers=model_config.num_hidden_layers,
        vocab_size=model_config.vocab_size,
        intermediate_dim=model_config.intermediate_size,
        gpu_mem_utilization=gpu_mem_utilization,
        node_layer_comb=node_layer_comb,
        dtype=torch.float16,
    )
    if head_dim is not None:
        throughput_kwargs["head_dim"] = head_dim

    throughput, single_latency, num_blocks = get_throughput(**throughput_kwargs)

    gbs = 0
    if throughput > 0:
        gbs, _ = get_global_batch_size(**throughput_kwargs)

    return {
        "instance_type": instance_type,
        "num_instances": num_instances,
        "gpu_type": gpu_type,
        "tp_size": tp_size,
        "actual_pp": actual_pp,
        "num_replicas": num_replicas,
        "unused_instances": unused_instances,
        "layers_per_stage": [int(l) for l in layers_per_stage],
        "optimal_latency_ms": optimal_latency,
        "throughput_per_replica": throughput,
        "total_throughput": throughput * num_replicas,
        "single_latency_ms": single_latency,
        "num_blocks": num_blocks,
        "max_batch_size": gbs,
        "gpu_mem_utilization": gpu_mem_utilization,
    }


def main():
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    print("=" * 80)
    print("AlpaServe Optimizer Result (original AlpaServe placement algorithm)")
    print(f"Model: {model_name}")
    print("Cluster: g6.12xlarge×2, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    groups = [
        ("g6.12xlarge", 3, 0.85),
        ("g5.12xlarge", 2, 0.85),
        ("g6e.xlarge", 4, 0.85),
    ]

    total_throughput = 0
    all_results = []
    pipeline_idx = 1

    for instance_type, num_instances, gpu_util in groups:
        result = run_alpaserve_optimizer(
            instance_type, num_instances, model_name, model_config,
            gpu_mem_utilization=gpu_util)
        all_results.append(result)

        print(f"\n{'─' * 60}")
        print(f"Instance group: {instance_type} × {num_instances}")
        print(f"  GPU: {result['gpu_type']} (TP={result['tp_size']})")
        print(f"  AlpaServe chose: PP={result['actual_pp']}, "
              f"{result['num_replicas']} replica(s)"
              + (f', {result["unused_instances"]} unused' if result['unused_instances'] > 0 else ''))
        print(f"  Layer partition: {result['layers_per_stage']}")
        print(f"  Optimal stage latency: {result['optimal_latency_ms']:.2f} ms")
        print(f"  Per-replica throughput: {result['throughput_per_replica']:.4f} req/s")
        print(f"  Total throughput ({result['num_replicas']} replicas): "
              f"{result['total_throughput']:.4f} req/s")
        print(f"  Single request latency: {result['single_latency_ms']:.2f} ms")
        print(f"  Num GPU blocks: {result['num_blocks']}")
        if result['max_batch_size'] > 0:
            print(f"  Max batch size: {result['max_batch_size']}")

        total_throughput += result['total_throughput']

    print(f"\n{'=' * 60}")
    print(f"Total system throughput: {total_throughput:.3f} req/s")

    # ─── Save JSON ────────────────────────────────────────────────────────
    output_pipelines = []
    pipeline_idx = 1
    for result in all_results:
        for replica in range(result['num_replicas']):
            stages = [[result['instance_type'], l] for l in result['layers_per_stage']]
            tp_list = [result['tp_size']] * result['actual_pp']
            output_pipelines.append({
                "label": f"AP-P{pipeline_idx}",
                "system": "AlpaServe",
                "stages": stages,
                "parallel_strategy": tp_list,
                "pp_layer_partition": ",".join(str(l) for l in result['layers_per_stage']),
                "gpu_memory_utilization": result['gpu_mem_utilization'],
                "predicted_throughput_rps": result['throughput_per_replica'],
                "predicted_total_latency_ms": result['single_latency_ms'],
                "optimal_stage_latency_ms": result['optimal_latency_ms'],
                "max_batch_size": result['max_batch_size'],
                "num_blocks": result['num_blocks'],
            })
            pipeline_idx += 1

    output = {
        "model": model_name,
        "workload": {"input_len": 763, "output_len": 232},
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "llama3-70b", "estimated")
    os.makedirs(results_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = os.path.join(results_dir, f"predicted_alpaserve_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
