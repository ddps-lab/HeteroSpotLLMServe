#!/usr/bin/env python3
"""
Generate pipelines.json from ShuntServe optimizer predicted results.

Reads predicted_shuntserve_*.json files and produces:
  - pipelines_{model}.json  (pipeline configs + node_layer_mapping)
  - nodes_{model}.json      (node name -> ip mapping, to be filled)

Usage:
    python generate_pipelines.py
    python generate_pipelines.py --model llama3-70b
    python generate_pipelines.py --model qwen3-32b
    python generate_pipelines.py --model all  (default)
"""

import json
import argparse
import os
from collections import defaultdict

S3_BUCKET = "hetero-spot-llm-serve-models"

# ── Paths to optimizer results ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_RESULTS = os.path.join(SCRIPT_DIR, "../ModelPlacement/optimizer/results")

MODEL_CONFIGS = {
    "llama3-70b": {
        "predicted_path": os.path.join(
            OPTIMIZER_RESULTS,
            "llama3-70b/estimated/predicted_shuntserve_Llama-3.1-70B-Instruct.json"
        ),
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "total_num_layers": 80,
    },
    "qwen3-32b": {
        "predicted_path": os.path.join(
            OPTIMIZER_RESULTS,
            "qwen3-32b/estimated/predicted_shuntserve_Qwen3-32B.json"
        ),
        "model_name": "Qwen/Qwen3-32B",
        "total_num_layers": 64,
    },
}


def normalize_instance_type(raw: str) -> tuple:
    """Parse '(spot)g6.12xlarge' -> ('spot', 'g6.12xlarge')"""
    raw = raw.strip()
    if raw.startswith("(spot)"):
        return "spot", raw[len("(spot)"):]
    elif raw.startswith("(on-demand)") or raw.startswith("(ondemand)"):
        tag = "on-demand" if raw.startswith("(on-demand)") else "ondemand"
        return "on_demand", raw[len(f"({tag})"):]
    else:
        return "spot", raw


def generate(model_key: str) -> tuple:
    """Generate pipelines.json and nodes.json for a single model.
    Returns (pipelines_dict, nodes_dict) or (None, None)."""
    cfg = MODEL_CONFIGS[model_key]
    predicted_path = os.path.normpath(cfg["predicted_path"])

    if not os.path.exists(predicted_path):
        print(f"⚠️  Not found: {predicted_path}")
        return None, None

    with open(predicted_path) as f:
        data = json.load(f)

    pipelines_data = data["pipelines"]
    model_name = cfg["model_name"]
    total_num_layers = cfg["total_num_layers"]

    # Count total nodes needed per (pricing, instance_type)
    total_nodes = defaultdict(int)
    for pipe in pipelines_data:
        for stage_raw, _ in pipe["stages"]:
            pricing, inst = normalize_instance_type(stage_raw)
            total_nodes[(pricing, inst)] += 1

    # Build name pool: spot_g6_12xlarge_node_ip_1, ...
    name_pool = {}
    for (pricing, inst), count in sorted(total_nodes.items()):
        prefix = f"{pricing}_{inst.replace('.', '_').replace('-', '_')}"
        name_pool[(pricing, inst)] = [f"{prefix}_node_ip_{i+1}" for i in range(count)]

    # Assign names to stages
    assign_counter = defaultdict(int)
    result_pipelines = []

    for pipe in pipelines_data:
        stages = pipe["stages"]
        parallel_strategy = pipe["parallel_strategy"]

        node_layer_mapping = []
        for (stage_raw, num_layers) in stages:
            pricing, inst = normalize_instance_type(stage_raw)
            key = (pricing, inst)
            idx = assign_counter[key]
            assign_counter[key] += 1
            node_name = name_pool[key][idx]
            node_layer_mapping.append([node_name, num_layers])

        pipeline_config = {
            "model_name": model_name,
            "total_num_layers": total_num_layers,
            "gpu_memory_utilization": pipe.get("gpu_memory_utilization", 0.85),
            "pp_layer_partition": pipe["pp_layer_partition"],
            "parallel_strategy": parallel_strategy,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 512,
            "model_source": "s3",
            "s3_path": f"s3://{S3_BUCKET}/{model_name}",
        }
        if pipe.get("num_blocks") is not None:
            pipeline_config["num_gpu_blocks"] = pipe["num_blocks"]
        if pipe.get("max_batch_size") is not None:
            pipeline_config["max_batch_size"] = pipe["max_batch_size"]

        result_pipelines.append({
            "label": pipe["label"],
            "predicted_throughput_rps": pipe.get("predicted_throughput_rps", 0),
            "config": pipeline_config,
            "node_layer_mapping": node_layer_mapping,
        })

    # nodes.json: flat mapping of node_name -> ""
    nodes = {}
    for names in name_pool.values():
        for name in names:
            nodes[name] = ""

    pipelines_out = {
        "model": model_name,
        "model_key": model_key,
        "total_num_layers": total_num_layers,
        "total_throughput_rps": data.get("total_throughput_rps", 0),
        "pipelines": result_pipelines,
    }

    return pipelines_out, nodes


def main():
    parser = argparse.ArgumentParser(
        description="Generate pipelines.json from ShuntServe optimizer results"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which model to generate (default: all)"
    )
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    for model_key in models:
        pipelines_out, nodes_out = generate(model_key)
        if pipelines_out is None:
            continue

        suffix = model_key.replace("-", "_")
        pipelines_path = os.path.join(SCRIPT_DIR, f"pipelines_{suffix}.json")
        nodes_path = os.path.join(SCRIPT_DIR, f"nodes_{suffix}.json")

        with open(pipelines_path, "w") as f:
            json.dump(pipelines_out, f, indent=2)
        with open(nodes_path, "w") as f:
            json.dump(nodes_out, f, indent=2)

        # Summary
        print(f"✅ {model_key}")
        print(f"   {pipelines_path}")
        print(f"   {nodes_path}")
        print(f"   Pipelines: {len(pipelines_out['pipelines'])}  |  "
              f"Nodes: {len(nodes_out)}  |  "
              f"Total: {pipelines_out['total_throughput_rps']:.3f} req/s")
        for pipe in pipelines_out["pipelines"]:
            mapping = "  →  ".join(
                f"{n}:{l}" for n, l in pipe["node_layer_mapping"]
            )
            print(f"   {pipe['label']} ({pipe['predicted_throughput_rps']:.3f} req/s): {mapping}")
        print()


if __name__ == "__main__":
    main()
