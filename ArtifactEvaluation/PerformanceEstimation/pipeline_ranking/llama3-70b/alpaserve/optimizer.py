"""
AlpaServe Optimizer: shows what pipeline config the DP optimizer produces
for each homogeneous instance group.
Run: python3 optimizer.py
"""
import sys
import os

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

import torch
import numpy as np
from transformers import AutoConfig
from alpaserve_optimizer import AlpaServeOptimizer
from hardware_specs import INSTANCE_SPEC, INTERCONNECT_SPEC
from estimator_utils import get_throughput


def run_alpaserve_optimizer(instance_type, num_instances, model_name, model_config, gpu_mem_utilization=0.85):
    """Run AlpaServe DP optimizer for a homogeneous group."""
    gpu_type = INSTANCE_SPEC[instance_type]['gpu_type']
    tp_size = INSTANCE_SPEC[instance_type]['gpu_count']
    interconnect_bandwidth = INTERCONNECT_SPEC[INSTANCE_SPEC[instance_type]['interconnect']]['bandwidth']
    num_stage = num_instances  # PP = number of instances (each instance = 1 PP stage with TP)

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

    optimizer = AlpaServeOptimizer(gpu_type, num_stage, config)
    optimal_latency, layers_per_stage = optimizer.optimize()

    # Get throughput via estimator
    node_layer_comb = [(instance_type, "us-east-1a", int(l)) for l in layers_per_stage]
    throughput, single_latency, num_blocks = get_throughput(
        avg_input_len=763,
        avg_output_len=232,
        max_model_len=8192,
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

    return {
        "instance_type": instance_type,
        "num_instances": num_instances,
        "gpu_type": gpu_type,
        "tp_size": tp_size,
        "pp_size": num_stage,
        "layers_per_stage": [int(l) for l in layers_per_stage],
        "optimal_latency_ms": optimal_latency,
        "throughput": throughput,
        "single_latency_ms": single_latency,
        "num_blocks": num_blocks,
    }


def main():
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    print("=" * 80)
    print("AlpaServe Optimizer Result (DP per homogeneous group)")
    print(f"Model: {model_name}")
    print("Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    groups = [
        ("g6.12xlarge", 3, 0.85),   # Pipeline 1: L4×4 × 3 instances
        ("g5.12xlarge", 2, 0.85),   # Pipeline 2: A10G×4 × 2 instances
        ("g6e.xlarge", 4, 0.9),     # Pipeline 3: L40S×1 × 4 instances
    ]

    total_throughput = 0
    for i, (instance_type, num_instances, gpu_util) in enumerate(groups, 1):
        result = run_alpaserve_optimizer(instance_type, num_instances, model_name, model_config, gpu_util)

        print(f"\n{'─' * 60}")
        print(f"Pipeline {i}: {instance_type} × {num_instances}")
        print(f"  GPU: {result['gpu_type']} (TP={result['tp_size']})")
        print(f"  PP={result['pp_size']}, Layer partition: {result['layers_per_stage']}")
        print(f"  Parallel strategy: {[result['tp_size']] * result['pp_size']}")
        print(f"  Optimal stage latency: {result['optimal_latency_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput']:.4f} req/s")
        print(f"  Single request latency: {result['single_latency_ms']:.2f} ms")
        print(f"  Num GPU blocks: {result['num_blocks']}")

        if result['throughput'] > 0:
            from estimator_utils import get_global_batch_size
            gbs, _ = get_global_batch_size(
                avg_input_len=763, avg_output_len=232, max_model_len=8192,
                hidden_dim=model_config.hidden_size,
                num_attention_head=model_config.num_attention_heads,
                num_kv_cache_head=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
                total_num_layers=model_config.num_hidden_layers,
                vocab_size=model_config.vocab_size,
                intermediate_dim=model_config.intermediate_size,
                gpu_mem_utilization=gpu_util,
                node_layer_comb=[(instance_type, "us-east-1a", l) for l in result['layers_per_stage']],
                dtype=torch.float16,
            )
            print(f"  Max batch size: {gbs}")

        total_throughput += max(result['throughput'], 0)

    print(f"\n{'=' * 60}")
    print(f"Total system throughput: {total_throughput:.3f} req/s")


if __name__ == "__main__":
    main()
