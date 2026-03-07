"""
HexGen Optimizer for Qwen3-32B: shows what pipeline config the GA optimizer produces.
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

from transformers import AutoConfig
from hexgen_optimizer import run_hexgen_ga, main as hexgen_main


def main():
    model_name = "Qwen/Qwen3-32B"
    model_config = AutoConfig.from_pretrained(model_name)
    head_dim = getattr(model_config, "head_dim", None) or (model_config.hidden_size // model_config.num_attention_heads)

    # FP16 weight size in GB (approximate)
    weight_size_gb = 62

    print("=" * 80)
    print("HexGen Optimizer Result (Genetic Algorithm)")
    print(f"Model: {model_name}")
    print(f"  hidden_size={model_config.hidden_size}, num_layers={model_config.num_hidden_layers}, "
          f"num_heads={model_config.num_attention_heads}, head_dim={head_dim}")
    print("Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    # Pass model_config to configure HexGen's cost model for Qwen3
    hexgen_model_config = {
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "input_len": 763,
        "output_len": 232,
        "weight_size_gb": weight_size_gb,
    }

    best_ind, cluster, cluster_config_flatten = hexgen_main(model_config=hexgen_model_config)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Best fitness: {best_ind.fitness.values[0]:.6f}")

    genes_all = best_ind.genes[0]
    plans_all = best_ind.plan[1][0]
    pp_partitions_all = best_ind.pp_partition[0]

    for pipeline_idx in range(len(genes_all)):
        plan = plans_all[pipeline_idx]
        pp_partition = pp_partitions_all[pipeline_idx]
        print(f"\n  Pipeline {pipeline_idx+1}:")
        print(f"    TP strategy: {plan}")
        print(f"    PP partition: {pp_partition}")

        gene = genes_all[pipeline_idx]
        for node_idx, gpu_count in enumerate(gene):
            if gpu_count > 0:
                print(f"    Node {node_idx} ({cluster_config_flatten[node_idx]}): {gpu_count} GPUs")


if __name__ == "__main__":
    main()
