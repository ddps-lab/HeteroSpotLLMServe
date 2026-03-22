#!/usr/bin/env python3
"""
Debug tool — show how spot trace events map to switch_nodes calls (Qwen3-32B, Scenario B).

Prints:
  1. Initial pipeline node layout
  2. Per-timepoint: raw events, merged switch_nodes call, resulting pipeline state
"""
import json
import os
from collections import defaultdict

SCENARIO = "B"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SPOT_TOLERANCE_DIR = os.path.dirname(MODEL_DIR)

with open(os.path.join(MODEL_DIR, f"pipelines_qwen3_32b_scenario_{SCENARIO}.json")) as f:
    pipelines_data = json.load(f)
with open(os.path.join(SPOT_TOLERANCE_DIR, f"spot_trace_events_scenario_{SCENARIO}.json")) as f:
    events_data = json.load(f)


def get_counterpart_name(name: str) -> str:
    if name.startswith("spot_"):
        return "on_demand_" + name[len("spot_"):]
    elif name.startswith("on_demand_"):
        return "spot_" + name[len("on_demand_"):]
    raise ValueError(f"Unknown prefix: {name}")


def short(name: str) -> str:
    """Shorten node name for display."""
    return name.replace("_node_ip_", "#").replace("_xlarge", "").replace("_12", ".12")


def main():
    pipelines = pipelines_data["pipelines"]

    # Build mutable pipeline state: pipeline_idx → [(node_name, layers), ...]
    pipeline_states = []
    for p in pipelines:
        pipeline_states.append([(n, l) for n, l in p["node_layer_mapping"]])

    # ── Initial state ─────────────────────────────────────────────────
    print("=" * 80)
    print(f"  Scenario {SCENARIO} — {pipelines_data['model']}")
    print(f"  Total predicted throughput: {pipelines_data['total_throughput_rps']:.3f} req/s")
    print("=" * 80)

    print("\n[t=0min] Initial Pipeline Layout")
    print("-" * 60)
    for i, p in enumerate(pipelines):
        label = p["label"]
        tput = p["predicted_throughput_rps"]
        stages = " → ".join(f"{short(n)}({l}L)" for n, l in pipeline_states[i])
        print(f"  {label} (tput={tput:.3f}): {stages}")

    # ── Group events by time ──────────────────────────────────────────
    grouped_events = defaultdict(list)
    for event in events_data["events"]:
        grouped_events[event["time_min"]].append(event)

    # Pre-populate interrupted_spots from initial pipeline state
    interrupted_spots = {}
    for state in pipeline_states:
        for node_name, _ in state:
            if node_name.startswith("on_demand_"):
                spot_name = get_counterpart_name(node_name)
                interrupted_spots[spot_name] = node_name

    if interrupted_spots:
        print(f"\n  Initially on-demand (spot unavailable):")
        for spot, od in sorted(interrupted_spots.items()):
            print(f"    {short(spot)} → {short(od)}")

    for time_min in sorted(grouped_events.keys()):
        print(f"\n{'=' * 80}")
        print(f"[t={time_min}min] Events")
        print("-" * 60)

        all_old_names = []
        all_new_names = []

        for event in grouped_events[time_min]:
            etype = event["type"]
            instances = event["instances"]
            print(f"  {etype.upper()}: {[short(n) for n in instances]}")

            if etype == "interruption":
                old_names = instances
                new_names = [get_counterpart_name(n) for n in old_names]
                for spot, od in zip(old_names, new_names):
                    interrupted_spots[spot] = od
                all_old_names.extend(old_names)
                all_new_names.extend(new_names)

            elif etype == "restore":
                old_names = [interrupted_spots[n] for n in instances]
                new_names = instances
                for n in new_names:
                    del interrupted_spots[n]
                all_old_names.extend(old_names)
                all_new_names.extend(new_names)

        assert len(all_old_names) == len(all_new_names), (
            f"t={time_min}min: old/new count mismatch: "
            f"{len(all_old_names)} vs {len(all_new_names)}"
        )

        # Apply to pipeline states and show per-pipeline changes
        swap_map = dict(zip(all_old_names, all_new_names))
        print()
        for i, state in enumerate(pipeline_states):
            label = pipelines[i]["label"]
            new_state = []
            changes = []
            for node_name, layers in state:
                if node_name in swap_map:
                    new_node = swap_map[node_name]
                    new_state.append((new_node, layers))
                    changes.append(f"{short(node_name)} → {short(new_node)}")
                else:
                    new_state.append((node_name, layers))
            pipeline_states[i] = new_state

            if changes:
                stages = " → ".join(f"{short(n)}({l}L)" for n, l in new_state)
                print(f"  {label}:")
                for c in changes:
                    print(f"    {c}")
                print(f"    => {stages}")
            else:
                print(f"  {label}: (no change)")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()
