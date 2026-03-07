"""
HexGen Optimizer: shows what pipeline config the GA optimizer produces.
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

from hexgen_optimizer import run_hexgen_ga, main as hexgen_main


def main():
    print("=" * 80)
    print("HexGen Optimizer Result (Genetic Algorithm)")
    print("Model: meta-llama/Llama-3.1-70B-Instruct")
    print("Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4")
    print("=" * 80)

    # Uses the main() from hexgen_optimizer which sets up the cluster
    # and runs the GA, printing the best individual
    best_ind, cluster, cluster_config_flatten = hexgen_main()

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Best fitness: {best_ind.fitness.values[0]:.6f}")

    genes_all = best_ind.genes[0]
    plans_all = best_ind.plan[1][0]
    pp_partitions_all = best_ind.pp_partition[0]

    total_throughput = 0
    for pipeline_idx in range(len(genes_all)):
        plan = plans_all[pipeline_idx]
        pp_partition = pp_partitions_all[pipeline_idx]
        print(f"\n  Pipeline {pipeline_idx+1}:")
        print(f"    TP strategy: {plan}")
        print(f"    PP partition: {pp_partition}")

        # Count instances used
        gene = genes_all[pipeline_idx]
        for node_idx, gpu_count in enumerate(gene):
            if gpu_count > 0:
                print(f"    Node {node_idx} ({cluster_config_flatten[node_idx]}): {gpu_count} GPUs")


if __name__ == "__main__":
    main()
