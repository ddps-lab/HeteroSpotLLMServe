#!/usr/bin/env python3
"""
Generate the MooncakeAgentTool (long-context) SpotTolerance configs for Llama-3.1-70B.

This is the long-context sibling of ``generate_pipelines.py``. It is a SEPARATE file
(the original generator is left untouched) and differs in three ways:

  1. It reads the LONG-CONTEXT optimizer output
     (``predicted_shuntserve_Llama-3.1-70B-Instruct-long-context.json``), produced by
     ``ModelPlacement/optimizer/llama3-70b/shuntserve-long-context.py`` with the Mooncake
     workload (input 5745 / output 154, ``max_model_len`` 16384).
  2. The emitted pipeline ``config`` uses the RUNTIME ``max_model_len`` =
     ``int(16384 * 1.1)`` = 18022 (vLLM adds a few tokens per sequence during serving, so
     the served engine needs ~10% headroom over the value the optimizer sized blocks for).
  3. It emits the SCENARIO-SPECIFIC pipelines directly: every spot node whose first event
     in ``spot_trace_events_scenario_{S}.json`` is a ``restore`` starts as ``on_demand_``
     (it is interrupted at t=0 and only comes back later), matching how the AzureConversation
     scenario pipelines were derived.

Outputs (for scenario A):
  - MooncakeAgentTool/llama3-70b/pipelines_llama3_70b_scenario_A.json
  - MooncakeAgentTool/nodes_scenario_A.json

The scenario events file is expected to already exist at
``MooncakeAgentTool/spot_trace_events_scenario_{S}.json`` (an identical copy of the
AzureConversation one — the spot scenario is shared, only the dataset differs).

Usage:
    python3 generate_pipelines_long_context.py            # scenario A (default)
    python3 generate_pipelines_long_context.py --scenario A
"""

import argparse
import json
import os
from collections import defaultdict

S3_BUCKET = "hetero-spot-llm-serve-models"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_RESULTS = os.path.join(SCRIPT_DIR, "../ModelPlacement/optimizer/results")

MODEL_KEY = "llama3-70b"
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
TOTAL_NUM_LAYERS = 80
PREDICTED_PATH = os.path.join(
    OPTIMIZER_RESULTS,
    "llama3-70b/estimated/predicted_shuntserve_Llama-3.1-70B-Instruct-long-context.json",
)

# Runtime max_model_len: optimizer sizes blocks at 16384, the served engine gets +10% headroom.
OPTIMIZER_MAX_MODEL_LEN = 16384
RUNTIME_MAX_MODEL_LEN = int(OPTIMIZER_MAX_MODEL_LEN * 1.1)  # 18022

MOONCAKE_DIR = os.path.join(SCRIPT_DIR, "MooncakeAgentTool")


def normalize_instance_type(raw: str) -> tuple:
    """Parse '(spot)g6.12xlarge' -> ('spot', 'g6.12xlarge')  (same as generate_pipelines.py)."""
    raw = raw.strip()
    if raw.startswith("(spot)"):
        return "spot", raw[len("(spot)"):]
    elif raw.startswith("(on-demand)") or raw.startswith("(ondemand)"):
        tag = "on-demand" if raw.startswith("(on-demand)") else "ondemand"
        return "on_demand", raw[len(f"({tag})"):]
    else:
        return "spot", raw


def counterpart(name: str) -> str:
    """spot_xxx -> on_demand_xxx, on_demand_xxx -> spot_xxx."""
    if name.startswith("spot_"):
        return "on_demand_" + name[len("spot_"):]
    if name.startswith("on_demand_"):
        return "spot_" + name[len("on_demand_"):]
    raise ValueError(f"Unknown prefix: {name}")


def first_event_type_per_node(events: list) -> dict:
    """Map each spot node name to the type ('interruption'/'restore') of its first event."""
    first = {}
    for ev in sorted(events, key=lambda e: e["time_min"]):
        for node in ev["instances"]:
            first.setdefault(node, ev["type"])
    return first


def build_base_node_layer_mapping(pipelines_data: list) -> list:
    """All-spot node_layer_mapping, identical naming to generate_pipelines.py."""
    total_nodes = defaultdict(int)
    for pipe in pipelines_data:
        for stage_raw, _ in pipe["stages"]:
            pricing, inst = normalize_instance_type(stage_raw)
            total_nodes[(pricing, inst)] += 1

    name_pool = {}
    for (pricing, inst), count in sorted(total_nodes.items()):
        prefix = f"{pricing}_{inst.replace('.', '_').replace('-', '_')}"
        name_pool[(pricing, inst)] = [f"{prefix}_node_ip_{i + 1}" for i in range(count)]

    assign_counter = defaultdict(int)
    out = []
    for pipe in pipelines_data:
        nlm = []
        for (stage_raw, num_layers) in pipe["stages"]:
            pricing, inst = normalize_instance_type(stage_raw)
            key = (pricing, inst)
            idx = assign_counter[key]
            assign_counter[key] += 1
            nlm.append([name_pool[key][idx], num_layers])
        out.append(nlm)
    return out


def generate(scenario: str):
    predicted_path = os.path.normpath(PREDICTED_PATH)
    if not os.path.exists(predicted_path):
        raise FileNotFoundError(
            f"Long-context optimizer output not found: {predicted_path}\n"
            f"Run ModelPlacement/optimizer/llama3-70b/shuntserve-long-context.py first."
        )

    events_path = os.path.join(MOONCAKE_DIR, f"spot_trace_events_scenario_{scenario}.json")
    if not os.path.exists(events_path):
        raise FileNotFoundError(
            f"Scenario events not found: {events_path}\n"
            f"Copy it from AzureConversation/spot_trace_events_scenario_{scenario}.json first."
        )

    with open(predicted_path) as f:
        predicted = json.load(f)
    with open(events_path) as f:
        events = json.load(f)["events"]

    pipelines_data = predicted["pipelines"]
    base_mappings = build_base_node_layer_mapping(pipelines_data)

    # Which spot nodes start as on_demand (first event is a restore => interrupted at t=0).
    first_type = first_event_type_per_node(events)
    initial_on_demand = {n for n, t in first_type.items() if t == "restore"}

    result_pipelines = []
    all_pipeline_names = set()
    for pipe, base_nlm in zip(pipelines_data, base_mappings):
        node_layer_mapping = []
        for name, num_layers in base_nlm:
            if name in initial_on_demand:
                name = counterpart(name)  # spot_ -> on_demand_
            node_layer_mapping.append([name, num_layers])
            all_pipeline_names.add(name)

        config = {
            "model_name": MODEL_NAME,
            "total_num_layers": TOTAL_NUM_LAYERS,
            "gpu_memory_utilization": pipe.get("gpu_memory_utilization", 0.85),
            "pp_layer_partition": pipe["pp_layer_partition"],
            "parallel_strategy": pipe["parallel_strategy"],
            "max_model_len": RUNTIME_MAX_MODEL_LEN,
            "max_num_batched_tokens": RUNTIME_MAX_MODEL_LEN,
            "max_num_seqs": 512,
            "model_source": "s3",
            "s3_path": f"s3://{S3_BUCKET}/{MODEL_NAME}",
        }
        if pipe.get("num_blocks") is not None:
            config["num_gpu_blocks"] = pipe["num_blocks"]
        if pipe.get("max_batch_size") is not None:
            config["max_batch_size"] = pipe["max_batch_size"]

        result_pipelines.append({
            "label": pipe["label"],
            "predicted_throughput_rps": pipe.get("predicted_throughput_rps", 0),
            "config": config,
            "node_layer_mapping": node_layer_mapping,
        })

    # nodes.json = names in pipelines  ∪  every event spot node and its on_demand counterpart.
    node_names = set(all_pipeline_names)
    for ev in events:
        for spot_name in ev["instances"]:
            node_names.add(spot_name)
            node_names.add(counterpart(spot_name))
    nodes = {name: "" for name in sorted(node_names)}

    # ── Runtime-consistency assertions (R1) ──────────────────────────────────
    for spot_name, t in first_type.items():
        if t == "restore":
            od = counterpart(spot_name)
            assert od in all_pipeline_names, (
                f"Scenario {scenario}: restore-first node {spot_name} has no initial "
                f"on_demand counterpart {od} in the placement — events incompatible with "
                f"the long-context placement."
            )
        else:  # interruption-first
            assert spot_name in all_pipeline_names, (
                f"Scenario {scenario}: interruption-first node {spot_name} is not initially "
                f"active (spot) in the placement — events incompatible with the placement."
            )

    pipelines_out = {
        "model": MODEL_NAME,
        "model_key": MODEL_KEY,
        "total_num_layers": TOTAL_NUM_LAYERS,
        "total_throughput_rps": predicted.get("total_throughput_rps", 0),
        "pipelines": result_pipelines,
    }
    return pipelines_out, nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", "-s", default="A", help="Scenario letter (default: A)")
    args = parser.parse_args()
    scenario = args.scenario

    pipelines_out, nodes_out = generate(scenario)

    model_out_dir = os.path.join(MOONCAKE_DIR, MODEL_KEY)
    os.makedirs(model_out_dir, exist_ok=True)
    pipelines_path = os.path.join(model_out_dir, f"pipelines_llama3_70b_scenario_{scenario}.json")
    nodes_path = os.path.join(MOONCAKE_DIR, f"nodes_scenario_{scenario}.json")

    with open(pipelines_path, "w") as f:
        json.dump(pipelines_out, f, indent=2)
    with open(nodes_path, "w") as f:
        json.dump(nodes_out, f, indent=2)

    print(f"✅ MooncakeAgentTool long-context — scenario {scenario}")
    print(f"   {pipelines_path}")
    print(f"   {nodes_path}")
    print(f"   max_model_len (runtime): {RUNTIME_MAX_MODEL_LEN}  |  "
          f"Total predicted: {pipelines_out['total_throughput_rps']:.3f} req/s")
    for pipe in pipelines_out["pipelines"]:
        mapping = "  →  ".join(f"{n}:{l}" for n, l in pipe["node_layer_mapping"])
        print(f"   {pipe['label']} ({pipe['predicted_throughput_rps']:.3f} req/s): {mapping}")
    print(f"   nodes ({len(nodes_out)}): {', '.join(sorted(nodes_out))}")


if __name__ == "__main__":
    main()
