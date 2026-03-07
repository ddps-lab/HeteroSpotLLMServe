"""
ShuntServe Optimizer: shows what pipeline config the optimizer produces.
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
from transformers import AutoConfig
from shuntserve_optimizer import run_test_case, Pipeline
from cluster_pool import ClusterPool


def main():
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

    cluster_pool = ClusterPool(available_spot_nodes=available_nodes, spot_prices=prices)

    print("=" * 80)
    print("ShuntServe Optimizer Result")
    print(f"Model: {model_name}")
    print(f"Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    results, optimizer, opt_time = run_test_case(
        config, budget=9999, latency_slo=99999999,
        cluster_pool=cluster_pool, max_stages=13, top_k=5,
        optimization_mode="soft_slo"
    )

    if results:
        print(f"\nFound {len(results)} pipeline(s):")
        total_throughput = 0
        for i, pipeline in enumerate(results, 1):
            print(f"\n{'─' * 60}")
            print(f"Pipeline {i}:")
            print(f"  {pipeline}")
            
            # Extract parallel strategy
            from hardware_specs import INSTANCE_SPEC
            tp_list = []
            for inst in pipeline.stages:
                tp_list.append(INSTANCE_SPEC[inst]['gpu_count'])
            print(f"  TP strategy: {tp_list}")
            print(f"  Layer partition: {list(pipeline.layer_per_stage)}")
            total_throughput += pipeline.throughput

        print(f"\n{'=' * 60}")
        print(f"Total system throughput: {total_throughput:.3f} req/s")
        print(f"Optimization time: {opt_time:.3f}s")
    else:
        print("No feasible pipeline found.")


if __name__ == "__main__":
    main()
