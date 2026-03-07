"""
Estimator prediction for varying batch sizes and TP/PP configurations.
Generates predicted throughput using ShuntServe's roofline model estimator.

For multi-GPU instances (e.g., g6e.12xlarge = L40S×4), automatically
evaluates all TP/PP combinations (TP=4/PP=1, TP=2/PP=2, TP=1/PP=4).

Usage:
    python predict.py --model meta-llama/Llama-3.1-70B-Instruct --instance g6e.12xlarge
    python predict.py --model meta-llama/Llama-3.1-8B-Instruct --instance g6.xlarge
    python predict.py --model meta-llama/Llama-3.1-70B --instance g6e.12xlarge --batch-sizes 1,2,4,8,16
"""
import argparse
import json
import os
import sys
import torch
from transformers import AutoConfig

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

from estimator_utils import (
    get_throughput,
    get_global_batch_size,
    get_prefill_computation_latency_per_layer,
    get_decoding_computation_latency_per_layer,
    get_prefill_compute_logit_latency,
    get_decoding_compute_logit_latency,
    get_tp_communication_latency_per_layer,
)
from hardware_specs import GPU_SPEC, INSTANCE_SPEC, INTERCONNECT_SPEC


# ─── Instance family → TP/PP variation mapping ──────────────────────────────
# Maps base instance (multi-GPU) → list of sub-instance representations
# for different TP/PP splits.
#
# e.g., g6e.12xlarge (L40S×4) can be split into:
#   TP=4 PP=1 → g6e.12xlarge (1 stage, 4 GPUs)
#   TP=2 PP=2 → g6e.12xlarge(half) × 2 stages (2 GPUs each)
#   TP=1 PP=4 → g6e.xlarge × 4 stages (1 GPU each)

INSTANCE_VARIATIONS = {
    "g6e.12xlarge": [
        {"label": "TP=4, PP=1", "tp": 4, "pp": 1,
         "stage_fn": lambda n: [("g6e.12xlarge", n)]},
        {"label": "TP=2, PP=2", "tp": 2, "pp": 2,
         "stage_fn": lambda n: [("g6e.12xlarge(half)", n // 2),
                                ("g6e.12xlarge(half)", n - n // 2)]},
        {"label": "TP=1, PP=4", "tp": 1, "pp": 4,
         "stage_fn": lambda n: [("g6e.xlarge", n // 4),
                                ("g6e.xlarge", n // 4),
                                ("g6e.xlarge", n // 4),
                                ("g6e.xlarge", n - 3 * (n // 4))]},
    ],
    "g6.12xlarge": [
        {"label": "TP=4, PP=1", "tp": 4, "pp": 1,
         "stage_fn": lambda n: [("g6.12xlarge", n)]},
        {"label": "TP=2, PP=2", "tp": 2, "pp": 2,
         "stage_fn": lambda n: [("g6.12xlarge(half)", n // 2),
                                ("g6.12xlarge(half)", n - n // 2)]},
        {"label": "TP=1, PP=4", "tp": 1, "pp": 4,
         "stage_fn": lambda n: [("g6.xlarge", n // 4),
                                ("g6.xlarge", n // 4),
                                ("g6.xlarge", n // 4),
                                ("g6.xlarge", n - 3 * (n // 4))]},
    ],
    "g5.12xlarge": [
        {"label": "TP=4, PP=1", "tp": 4, "pp": 1,
         "stage_fn": lambda n: [("g5.12xlarge", n)]},
        {"label": "TP=2, PP=2", "tp": 2, "pp": 2,
         "stage_fn": lambda n: [("g5.12xlarge(half)", n // 2),
                                ("g5.12xlarge(half)", n - n // 2)]},
        {"label": "TP=1, PP=4", "tp": 1, "pp": 4,
         "stage_fn": lambda n: [("g5.xlarge", n // 4),
                                ("g5.xlarge", n // 4),
                                ("g5.xlarge", n // 4),
                                ("g5.xlarge", n - 3 * (n // 4))]},
    ],
}


def build_model_config(model_name: str):
    model_config = AutoConfig.from_pretrained(model_name)
    # Use explicit head_dim from config if available (e.g. Qwen3),
    # otherwise fall back to hidden_size // num_attention_heads
    head_dim = getattr(
        model_config, "head_dim",
        model_config.hidden_size // model_config.num_attention_heads
    )
    return {
        "expected_input_len": 763,
        "expected_output_len": 232,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        ),
        "head_dim": head_dim,
        "intermediate_size": model_config.intermediate_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.85,
    }


def get_max_batch_for_pipeline(config, stages):
    """Get max batch size and num blocks for a given pipeline stage configuration."""
    node_layer_comb = [(inst, "dummy-az", layers) for inst, layers in stages]
    max_batch, num_blocks = get_global_batch_size(
        avg_input_len=config["expected_input_len"],
        avg_output_len=config["expected_output_len"],
        max_model_len=config["max_model_len"],
        hidden_dim=config["hidden_size"],
        num_attention_head=config["num_attention_heads"],
        num_kv_cache_head=config["num_key_value_heads"],
        total_num_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
        intermediate_dim=config["intermediate_size"],
        gpu_mem_utilization=config["gpu_mem_utilization"],
        node_layer_comb=node_layer_comb,
        dtype=config["dtype"],
        head_dim=config.get("head_dim"),
    )
    return int(max_batch), int(num_blocks)


def predict_latency_for_pipeline(config, stages, batch_size):
    """
    Predict prefill/decode latency for a pipeline at a specific batch size.
    Computes per-stage latencies and sums them (serial execution for single batch).
    """
    stage_prefill_lats = []
    stage_decode_lats = []
    processed_layers = 0

    for inst, layers in stages:
        gpu_type = INSTANCE_SPEC[inst]["gpu_type"]
        num_gpu = INSTANCE_SPEC[inst]["gpu_count"]
        p2p_bw = INTERCONNECT_SPEC[INSTANCE_SPEC[inst]["interconnect"]]["bandwidth"]
        processed_layers += layers

        prefill = get_prefill_computation_latency_per_layer(
            gpu_type=gpu_type, gpu_count=num_gpu,
            input_len=config["expected_input_len"],
            hidden_dim=config["hidden_size"],
            num_attention_head=config["num_attention_heads"],
            num_kv_cache_head=config["num_key_value_heads"],
            intermediate_dim=config["intermediate_size"],
            batch_size=batch_size, dtype=config["dtype"],
        ) * layers

        decode = get_decoding_computation_latency_per_layer(
            gpu_type=gpu_type, gpu_count=num_gpu,
            input_len=config["expected_input_len"],
            output_len=config["expected_output_len"],
            hidden_dim=config["hidden_size"],
            num_attention_head=config["num_attention_heads"],
            num_kv_cache_head=config["num_key_value_heads"],
            intermediate_dim=config["intermediate_size"],
            batch_size=batch_size, dtype=config["dtype"],
        ) * layers

        # Logit computation on last stage only
        if processed_layers == config["num_layers"]:
            prefill += get_prefill_compute_logit_latency(
                gpu_type=gpu_type, gpu_count=num_gpu,
                input_len=config["expected_input_len"],
                hidden_dim=config["hidden_size"],
                batch_size=batch_size, vocab_size=config["vocab_size"],
                dtype=config["dtype"],
            )
            decode += get_decoding_compute_logit_latency(
                gpu_type=gpu_type, gpu_count=num_gpu,
                output_len=config["expected_output_len"],
                hidden_dim=config["hidden_size"],
                batch_size=batch_size, vocab_size=config["vocab_size"],
                dtype=config["dtype"],
            )

        # TP communication
        tp_prefill = get_tp_communication_latency_per_layer(
            tp_size=num_gpu, batch_size=batch_size,
            sequence_len=config["expected_input_len"],
            hidden_dim=config["hidden_size"],
            p2p_bandwidth=p2p_bw, dtype=config["dtype"],
        ) * layers
        tp_decode = get_tp_communication_latency_per_layer(
            tp_size=num_gpu, batch_size=batch_size,
            sequence_len=1,
            hidden_dim=config["hidden_size"],
            p2p_bandwidth=p2p_bw, dtype=config["dtype"],
        ) * layers * config["expected_output_len"]

        stage_prefill_lats.append(prefill + tp_prefill)
        stage_decode_lats.append(decode + tp_decode)

    total_prefill = sum(stage_prefill_lats)
    total_decode = sum(stage_decode_lats)
    total_latency = total_prefill + total_decode
    throughput = batch_size / (total_latency / 1000)

    return {
        "batch_size": batch_size,
        "prefill_latency_ms": total_prefill,
        "decode_latency_ms": total_decode,
        "total_latency_ms": total_latency,
        "throughput_rps": throughput,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict serving throughput using ShuntServe's roofline estimator"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., meta-llama/Llama-3.1-70B-Instruct)")
    parser.add_argument("--instance", type=str, required=True,
                        help="Instance type (e.g., g6e.12xlarge)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: auto 1,2,4,...,max)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    config = build_model_config(args.model)
    instance_type = args.instance
    gpu_count = INSTANCE_SPEC[instance_type]["gpu_count"]

    print(f"Model: {args.model}")
    print(f"  Layers: {config['num_layers']}, Hidden: {config['hidden_size']}")
    print(f"  Heads: {config['num_attention_heads']} (KV: {config['num_key_value_heads']})")
    print(f"  Intermediate: {config['intermediate_size']}, Vocab: {config['vocab_size']}")
    print(f"Instance: {instance_type}")
    print(f"  GPU: {INSTANCE_SPEC[instance_type]['gpu_type']} × {gpu_count}")
    print(f"Workload: input={config['expected_input_len']}, output={config['expected_output_len']}")
    print()

    # ─── Build variations ────────────────────────────────────────────────
    # Multi-GPU instance → auto-generate TP/PP variations
    # Single-GPU instance → just one config (TP=1, PP=1)
    if instance_type in INSTANCE_VARIATIONS:
        variations = INSTANCE_VARIATIONS[instance_type]
    else:
        # Single GPU or unknown → single variation
        variations = [
            {"label": f"TP={gpu_count}, PP=1", "tp": gpu_count, "pp": 1,
             "stage_fn": lambda n: [(instance_type, n)]},
        ]

    # ─── Determine max batch sizes per variation ─────────────────────────
    var_infos = []
    for var in variations:
        stages = var["stage_fn"](config["num_layers"])
        max_bs, num_blocks = get_max_batch_for_pipeline(config, stages)
        stages_str = " → ".join(f"{inst}:{layers}L" for inst, layers in stages)
        var_infos.append({
            "var": var,
            "stages": stages,
            "stages_str": stages_str,
            "max_batch_size": max_bs,
            "num_blocks": num_blocks,
        })
        print(f"  {var['label']:<16} : {stages_str}  (max_batch={max_bs}, num_blocks={num_blocks})")

    print()

    # ─── Determine batch sizes per variation ─────────────────────────────
    # Each variation gets its own batch size range based on its max_batch_size.
    # --batch-sizes overrides all variations with the same list.

    print()

    # ─── Predict and print table ─────────────────────────────────────────
    header = f"{'Config':<16} | {'Batch':>6} | {'Prefill(ms)':>12} | {'Decode(ms)':>12} | {'Total(ms)':>12} | {'Throughput':>12}"
    print(header)
    print("-" * len(header))

    all_results = []

    for vinfo in var_infos:
        var = vinfo["var"]
        stages = vinfo["stages"]
        max_bs = vinfo["max_batch_size"]
        label = var["label"]
        var_results = []

        if args.batch_sizes:
            batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        else:
            if max_bs <= 0:
                print(f"{label:<16} | {'ALL':>6} | {'OOM':>12} | {'OOM':>12} | {'OOM':>12} | {'OOM':>12}")
                print("-" * len(header))
                all_results.append({
                    "label": label, "tp": var["tp"], "pp": var["pp"],
                    "stages": [(inst, layers) for inst, layers in stages],
                    "max_batch_size": 0, "num_blocks": vinfo["num_blocks"],
                    "batch_results": [],
                })
                continue
            batch_sizes = []
            b = 1
            while b <= max_bs:
                batch_sizes.append(b)
                b *= 2
            if batch_sizes[-1] != max_bs:
                batch_sizes.append(max_bs)

        for bs in batch_sizes:
            if max_bs <= 0 or bs > max_bs:
                print(f"{label:<16} | {bs:>6} | {'OOM':>12} | {'OOM':>12} | {'OOM':>12} | {'OOM':>12}")
                var_results.append({"batch_size": bs, "status": "OOM"})
                continue

            result = predict_latency_for_pipeline(config, stages, bs)
            var_results.append(result)
            print(f"{label:<16} | {bs:>6} | {result['prefill_latency_ms']:>12.2f} | "
                  f"{result['decode_latency_ms']:>12.2f} | {result['total_latency_ms']:>12.2f} | "
                  f"{result['throughput_rps']:>12.4f}")

        all_results.append({
            "label": label,
            "tp": var["tp"],
            "pp": var["pp"],
            "stages": [(inst, layers) for inst, layers in stages],
            "max_batch_size": max_bs,
            "num_blocks": vinfo["num_blocks"],
            "batch_results": var_results,
        })

        print("-" * len(header))

    # ─── Save ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1]
    output_file = os.path.join(args.output_dir, f"predicted_{model_short}_{instance_type}.json")

    output = {
        "model": args.model,
        "instance": instance_type,
        "gpu_type": INSTANCE_SPEC[instance_type]["gpu_type"],
        "gpu_count": gpu_count,
        "workload": {
            "input_len": config["expected_input_len"],
            "output_len": config["expected_output_len"],
        },
        "batch_sizes": batch_sizes,
        "results": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
