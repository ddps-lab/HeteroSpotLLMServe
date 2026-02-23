import argparse
import sys
import os

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d


def build_model_config(model_name="meta-llama/Llama-3.1-70B-Instruct"):
    import torch
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_name)
    return {
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
    }


def parse_hexgen_pipelines(best_ind, cluster, cluster_config_flatten):
    """Parse hexgen GA output into per-pipeline stage lists (instance types)."""
    genes_all = best_ind.genes[0]       # per-pipeline [GPU usage per node]
    plans_all = best_ind.plan[1][0]     # per-pipeline [TP degree per stage]

    gpu_per_node = []
    for node_info in cluster:
        for _ in range(node_info["num_instances"]):
            gpu_per_node.append(node_info["num_gpu_per_instance"])

    pipelines_stages = []
    for gene, plan in zip(genes_all, plans_all):
        stages = []
        plan_offset = 0
        for node_idx, gpu_used in enumerate(gene):
            if gpu_used <= 0:
                continue
            base_type = cluster_config_flatten[node_idx]  # e.g., "g5.12xlarge"
            node_gpus = gpu_per_node[node_idx]             # e.g., 4

            consumed = 0
            while consumed < gpu_used:
                tp_degree = plan[plan_offset]
                plan_offset += 1
                consumed += tp_degree

                if tp_degree == node_gpus:
                    stage_type = f"(spot){base_type}"
                elif node_gpus > 1 and tp_degree == node_gpus // 2:
                    stage_type = f"{base_type}(half)"
                elif tp_degree == 1 and node_gpus > 1:
                    family = base_type.split(".")[0]  # "g5" or "g6"
                    stage_type = f"(spot){family}.xlarge"
                else:
                    stage_type = f"(spot){base_type}"

                stages.append(stage_type)

        pipelines_stages.append(stages)
    return pipelines_stages


def run_hexgen(config):
    """Run HEXGEN GA, parse output, and re-optimize layer partitioning with ShuntServe."""
    from collections import Counter
    from hexgen_optimizer import main as hexgen_main
    from cluster_pool import ClusterPool
    from shuntserve_optimizer import run_test_case

    hexgen_config = {
        **config,
        "gpu_mem_utilization": 0.85,
    }

    # Run hexgen GA (cluster is hardcoded inside)
    best_ind, cluster_def, cluster_config_flatten = hexgen_main()

    # Parse hexgen output into per-pipeline instance lists
    pipelines_stages = parse_hexgen_pipelines(best_ind, cluster_def, cluster_config_flatten)

    genes_all = best_ind.genes[0]
    plans_all = best_ind.plan[1][0]

    # Re-optimize layer partitioning per pipeline via ShuntServe (only_throughput)
    optimized_pipelines = []
    for i, stages in enumerate(pipelines_stages):
        instance_counts = Counter(stages)
        dummy_prices = {inst: 1.0 for inst in instance_counts}

        cluster_pool = ClusterPool(
            available_spot_nodes=instance_counts,
            spot_prices=dummy_prices,
        )

        results, _, _ = run_test_case(
            hexgen_config, budget=9999, latency_slo=99999999,
            cluster_pool=cluster_pool, max_stages=len(stages), top_k=1,
            optimization_mode="only_throughput",
        )

        optimized_pipelines.append(results[0] if results else None)

    # Output in hexgen's original format with re-optimized PP Partition
    print(f"\n{'=' * 80}")
    print(f"HEXGEN: {len(pipelines_stages)} pipeline(s)")
    print(f"{'=' * 80}")
    for i, (gene, plan) in enumerate(zip(genes_all, plans_all)):
        print(f"\nPipeline {i + 1}")

        # GPU usage per instance type (same as hexgen's original output)
        instance_usage = [0] * len(cluster_config_flatten)
        for node_idx, node_count in enumerate(gene):
            if node_count > 0 and node_idx < len(cluster_config_flatten):
                instance_usage[node_idx] += node_count

        offset = 0
        for node_info in cluster_def:
            num_instances = node_info["num_instances"]
            instance_type = node_info["instance_type"]
            total_usage = sum(instance_usage[offset:offset + num_instances])
            offset += num_instances
            print(f"  {instance_type}: {total_usage}")

        print(f"  Plan: {plan}")

        pipeline = optimized_pipelines[i]
        if pipeline:
            print(f"  PP Partition: {pipeline.layer_per_stage}")
            print(f"  Throughput: {pipeline.throughput:.2f} req/s")
            print(f"  Global Batch Size: {pipeline.global_batch_size}")
            print(f"  Num Blocks: {pipeline.num_blocks}")
        else:
            print("  (No feasible layer partitioning found)")


def run_alpaserve(config, cluster):
    from hardware_specs import INSTANCE_SPEC, INTERCONNECT_SPEC
    from alpaserve_optimizer import AlpaServeOptimizer
    from shuntserve_optimizer import Pipeline, get_minimum_latency
    from cluster_pool import ClusterPool

    cluster_pool = ClusterPool(
        available_spot_nodes=cluster["available_nodes"],
        spot_prices=cluster["prices"],
    )

    throughput_config = {
        **config,
        "gpu_mem_utilization": 0.85,
    }

    all_pipelines = []

    # AlpaServe is a homogeneous baseline — run layer partitioning per instance type
    for instance_type, num_instances in cluster["available_nodes"].items():
        if num_instances <= 0:
            continue

        base_instance_type = instance_type.replace("(spot)", "")
        num_stage = num_instances

        gpu_type = INSTANCE_SPEC[base_instance_type]['gpu_type']
        tp_size = INSTANCE_SPEC[base_instance_type]['gpu_count']
        interconnect_bandwidth = INTERCONNECT_SPEC[
            INSTANCE_SPEC[base_instance_type]['interconnect']
        ]['bandwidth']

        alpaserve_config = {
            **config,
            "tp_size": tp_size,
            "p2p_bandwidth": interconnect_bandwidth,
        }

        optimizer = AlpaServeOptimizer(gpu_type, num_stage, alpaserve_config)
        optimal_latency, layers_per_stage = optimizer.optimize()

        # Pack into Pipeline
        pipeline = Pipeline()
        for i in range(num_stage):
            pipeline.stages.append(instance_type)
            pipeline.azs.append("dummy-az")
            pipeline.layer_per_stage.append(int(layers_per_stage[i]))

        total_cost = sum(cluster_pool.get_instance_price(instance_type) for _ in range(num_stage))
        pipeline.set_cost(total_cost)
        pipeline.calculate_throughput(throughput_config)
        pipeline.single_request_latency = get_minimum_latency(pipeline, throughput_config)

        all_pipelines.append(pipeline)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"AlpaServe: Found {len(all_pipelines)} pipeline(s)")
    print(f"{'=' * 80}")
    for i, pipeline in enumerate(all_pipelines, 1):
        print(f"\nPipeline {i}:")
        print(f"  {pipeline}")
        print(f"  - Throughput: {pipeline.throughput:.2f} req/s")
        print(f"  - Cost: ${pipeline.cost:.2f} USD/h")


def run_shuntserve(config, cluster):
    import logging
    from collections import Counter
    from cluster_pool import ClusterPool
    from shuntserve_optimizer import run_test_case

    logger = logging.getLogger(__name__)

    shuntserve_config = {
        **config,
        "gpu_mem_utilization": 0.85,
    }

    remaining_nodes = dict(cluster["available_nodes"])
    all_pipelines = []
    pipeline_idx = 0

    while True:
        # Check if any instances remain
        if all(v <= 0 for v in remaining_nodes.values()):
            break

        cluster_pool = ClusterPool(
            available_spot_nodes=remaining_nodes,
            spot_prices=cluster["prices"],
        )

        logger.info(f"\n--- ShuntServe Iteration {pipeline_idx + 1} ---")
        logger.info(f"Remaining nodes: {remaining_nodes}")

        results, optimizer, optimization_time = run_test_case(
            shuntserve_config, budget=9999, latency_slo=99999999,
            cluster_pool=cluster_pool, max_stages=13, top_k=1,
            optimization_mode="soft_slo",
        )

        if not results:
            logger.info("No more feasible pipelines found. Stopping.")
            break

        best_pipeline = results[0]
        all_pipelines.append(best_pipeline)
        pipeline_idx += 1

        # Deduct used instances from remaining pool
        used_instances = Counter(best_pipeline.stages)
        for instance_type, count in used_instances.items():
            remaining_nodes[instance_type] = remaining_nodes.get(instance_type, 0) - count

    # Summary
    print(f"\n{'=' * 80}")
    print(f"ShuntServe: Found {len(all_pipelines)} pipeline(s)")
    print(f"{'=' * 80}")
    for i, pipeline in enumerate(all_pipelines, 1):
        print(f"\nPipeline {i}:")
        print(f"  {pipeline}")
        print(f"  - Throughput: {pipeline.throughput:.2f} req/s")
        print(f"  - Cost: ${pipeline.cost:.2f} USD/h")


def main():
    parser = argparse.ArgumentParser(description="Run baseline model placement optimizers")
    parser.add_argument(
        "--baseline",
        nargs="+",
        choices=["hexgen", "alpaserve", "shuntserve"],
        required=True,
        help="Baseline optimizer(s) to run",
    )
    args = parser.parse_args()

    # --- Shared model config ---
    config = None
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    needs_config = {"hexgen", "alpaserve", "shuntserve"}
    if needs_config & set(args.baseline):
        config = build_model_config(model_name=model_name)

    # --- Cluster definition ---
    cluster = {
        "available_nodes": {
            "(spot)g5.12xlarge": 2,
            "(spot)g6.12xlarge": 3,
            "(spot)g6e.xlarge": 4,
        },
        "prices": {
            "(spot)g5.12xlarge": 2.2915,
            "(spot)g6.12xlarge": 1.9445,
            "(spot)g6e.xlarge": 0.7040,
        },
    }

    # --- Run baselines ---
    for baseline in args.baseline:
        print(f"\n{'=' * 80}")
        print(f"Running baseline: {baseline}")
        print(f"{'=' * 80}\n")

        if baseline == "hexgen":
            run_hexgen(config)
        elif baseline == "alpaserve":
            run_alpaserve(config, cluster)
        elif baseline == "shuntserve":
            run_shuntserve(config, cluster)


if __name__ == "__main__":
    main()
