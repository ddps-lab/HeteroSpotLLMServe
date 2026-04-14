"""
AlpaServe Optimizer for Qwen3-32B: uses AlpaServe's original placement algorithm
to decide the optimal pipeline configuration for each homogeneous instance group.
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

# Import the shared runner from llama3-70b (same logic, just different model)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llama3-70b"))
from alpaserve import run_alpaserve_optimizer


def main():
    model_name = "Qwen/Qwen3-32B"
    model_config = AutoConfig.from_pretrained(model_name)
    head_dim = getattr(model_config, "head_dim", None) or (model_config.hidden_size // model_config.num_attention_heads)

    print("=" * 80)
    print("AlpaServe Optimizer Result (original AlpaServe placement algorithm)")
    print(f"Model: {model_name}")
    print(f"  hidden_size={model_config.hidden_size}, num_layers={model_config.num_hidden_layers}, "
          f"num_heads={model_config.num_attention_heads}, head_dim={head_dim}")
    print("Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    groups = [
        ("g6.12xlarge", 3, 0.85),
        ("g5.12xlarge", 2, 0.85),
        ("g6e.xlarge", 4, 0.85),
    ]

    total_throughput = 0
    all_results = []

    for instance_type, num_instances, gpu_util in groups:
        result = run_alpaserve_optimizer(
            instance_type, num_instances, model_name, model_config,
            head_dim=head_dim, gpu_mem_utilization=gpu_util)
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
            spot_name = f"(spot){result['instance_type']}"
            stages = [[spot_name, l] for l in result['layers_per_stage']]
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

    model_short = model_name.split("/")[-1]
    output = {
        "model": model_name,
        "workload": {"input_len": 763, "output_len": 232},
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "qwen3-32b", "estimated")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"predicted_alpaserve_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
