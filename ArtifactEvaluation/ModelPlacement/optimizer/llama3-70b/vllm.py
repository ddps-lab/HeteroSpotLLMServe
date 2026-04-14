"""
vLLM Optimizer: shows homogeneous pipeline configs (even layer partition).
vLLM doesn't have a placement optimizer — it uses simple even partitioning
within each homogeneous instance group.
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
from hardware_specs import INSTANCE_SPEC
from estimator_utils import get_throughput, get_global_batch_size, get_single_request_latency


def even_partition(num_layers, num_stages):
    """Even layer partition (vLLM default)."""
    base = num_layers // num_stages
    remainder = num_layers % num_stages
    layers = []
    for i in range(num_stages):
        layers.append(base + (1 if i < remainder else 0))
    return layers


def run_vllm_pipeline(instance_type, num_instances, model_config, gpu_mem_utilization=0.85):
    """Compute vLLM pipeline config with even partitioning."""
    num_layers = model_config.num_hidden_layers
    layers_per_stage = even_partition(num_layers, num_instances)
    tp_size = INSTANCE_SPEC[instance_type]['gpu_count']

    node_layer_comb = [(instance_type, "us-east-1a", l) for l in layers_per_stage]
    throughput, batch_latency, num_blocks = get_throughput(
        avg_input_len=763,
        avg_output_len=232,
        max_model_len=8192,
        hidden_dim=model_config.hidden_size,
        num_attention_head=model_config.num_attention_heads,
        num_kv_cache_head=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        total_num_layers=num_layers,
        vocab_size=model_config.vocab_size,
        intermediate_dim=model_config.intermediate_size,
        gpu_mem_utilization=gpu_mem_utilization,
        node_layer_comb=node_layer_comb,
        dtype=torch.float16,
    )

    single_latency = get_single_request_latency(
        avg_input_len=763,
        avg_output_len=232,
        hidden_dim=model_config.hidden_size,
        num_attention_head=model_config.num_attention_heads,
        num_kv_cache_head=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        total_num_layers=num_layers,
        intermediate_dim=model_config.intermediate_size,
        vocab_size=model_config.vocab_size,
        node_layer_comb=node_layer_comb,
        dtype=torch.float16,
    )

    gbs, _ = get_global_batch_size(
        avg_input_len=763, avg_output_len=232, max_model_len=8192,
        hidden_dim=model_config.hidden_size,
        num_attention_head=model_config.num_attention_heads,
        num_kv_cache_head=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        total_num_layers=num_layers,
        vocab_size=model_config.vocab_size,
        intermediate_dim=model_config.intermediate_size,
        gpu_mem_utilization=gpu_mem_utilization,
        node_layer_comb=node_layer_comb,
        dtype=torch.float16,
    )

    return {
        "instance_type": instance_type,
        "num_instances": num_instances,
        "gpu_type": INSTANCE_SPEC[instance_type]['gpu_type'],
        "tp_size": tp_size,
        "pp_size": num_instances,
        "layers_per_stage": layers_per_stage,
        "throughput": throughput,
        "single_latency_ms": single_latency,
        "batch_latency_ms": batch_latency,
        "num_blocks": num_blocks,
        "max_batch_size": gbs,
    }


def main():
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    print("=" * 80)
    print("vLLM Pipeline Config (even partition per homogeneous group)")
    print(f"Model: {model_name}")
    print("Cluster: g6.12xlarge×2, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    groups = [
        ("g6.12xlarge", 3, 0.85),   # Pipeline 1: L4×4 × 2
        ("g5.12xlarge", 2, 0.85),   # Pipeline 2: A10G×4 × 2
        ("g6e.xlarge", 4, 0.85),     # Pipeline 3: L40S×1 × 4
    ]

    total_throughput = 0
    all_results = []
    for i, (instance_type, num_instances, gpu_util) in enumerate(groups, 1):
        result = run_vllm_pipeline(instance_type, num_instances, model_config, gpu_util)
        all_results.append(result)

        print(f"\n{'─' * 60}")
        print(f"Pipeline {i}: {instance_type} × {num_instances}")
        print(f"  GPU: {result['gpu_type']} (TP={result['tp_size']})")
        print(f"  PP={result['pp_size']}, Layer partition: {result['layers_per_stage']}")
        print(f"  Parallel strategy: {[result['tp_size']] * result['pp_size']}")
        print(f"  Throughput: {result['throughput']:.4f} req/s")
        print(f"  Single request latency: {result['single_latency_ms']:.2f} ms")
        print(f"  Batch latency (global batch={result['max_batch_size']}): {result['batch_latency_ms']:.2f} ms")
        print(f"  Num GPU blocks: {result['num_blocks']}")
        print(f"  Max batch size: {result['max_batch_size']}")

        total_throughput += max(result['throughput'], 0)

    print(f"\n{'=' * 60}")
    print(f"Total system throughput: {total_throughput:.3f} req/s")

    # ─── Save results to JSON ────────────────────────────────────────────
    output_pipelines = []
    for i, result in enumerate(all_results, 1):
        tp_list = [result['tp_size']] * result['pp_size']
        stages_list = [[result['instance_type'], l] for l in result['layers_per_stage']]
        output_pipelines.append({
            "label": f"vL-P{i}",
            "system": "vLLM",
            "stages": stages_list,
            "parallel_strategy": tp_list,
            "pp_layer_partition": ",".join(str(l) for l in result['layers_per_stage']),
            "gpu_memory_utilization": groups[i-1][2],
            "predicted_throughput_rps": result['throughput'],
            "predicted_total_latency_ms": result['batch_latency_ms'],
            "predicted_single_latency_ms": result['single_latency_ms'],
            "max_batch_size": result['max_batch_size'],
            "num_blocks": result['num_blocks'],
        })

    output = {
        "model": model_name,
        "workload": {"input_len": 763, "output_len": 232},
        "pipelines": output_pipelines,
        "total_throughput_rps": total_throughput,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "llama3-70b", "estimated")
    os.makedirs(results_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = os.path.join(results_dir, f"predicted_vllm_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
