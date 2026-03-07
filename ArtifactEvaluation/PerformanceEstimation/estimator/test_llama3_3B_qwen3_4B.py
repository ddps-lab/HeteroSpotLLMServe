"""
Estimator comparison: Llama-3.2-3B vs Qwen3-4B on g6.xlarge (L4 x1).

Runs the roofline model estimator for both models and compares:
  - Per-layer latency (prefill / decode)
  - Memory capacity (max batch size)
  - Throughput estimates

Usage:
    python test_llama3_3B_qwen3_4B.py
    python test_llama3_3B_qwen3_4B.py --batch-sizes 1,2,4,8,16
    python test_llama3_3B_qwen3_4B.py --output-dir results
"""
import argparse
import json
import os
import sys
import torch
import logging

# Add ModelPlacement to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "ModelPlacement"))
del _d

from estimator_utils import (
    get_throughput,
    get_global_batch_size,
    get_memory_size_decoder_layer_weight_bytes,
    get_memory_size_embedding_or_lm_head_weight_bytes,
    get_prefill_computation_latency_per_layer,
    get_decoding_computation_latency_per_layer,
    get_prefill_compute_logit_latency,
    get_decoding_compute_logit_latency,
)
from hardware_specs import GPU_SPEC, INSTANCE_SPEC


# ─── Model Configs (hardcoded to avoid HuggingFace auth issues) ──────────────

MODELS = {
    "Llama-3.2-3B": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "hidden_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": True,
    },
    "Qwen3-4B": {
        "model_name": "Qwen/Qwen3-4B",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": True,
    },
}

INSTANCE_TYPE = "g6.xlarge"  # L4 x1
DTYPE = torch.float16
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.85
INPUT_LEN = 763   # Azure trace avg
OUTPUT_LEN = 232  # Azure trace avg


def estimate_model(model_name: str, model_cfg: dict, batch_sizes: list, verbose: bool = False):
    """Run estimator for a single model and return results."""
    gpu_type = INSTANCE_SPEC[INSTANCE_TYPE]["gpu_type"]
    gpu_count = INSTANCE_SPEC[INSTANCE_TYPE]["gpu_count"]
    gpu_spec = GPU_SPEC[gpu_type]

    print(f"\n{'=' * 70}")
    print(f"  {model_name}")
    print(f"  Layers: {model_cfg['num_hidden_layers']}, Hidden: {model_cfg['hidden_size']}, "
          f"Intermediate: {model_cfg['intermediate_size']}")
    print(f"  Heads: {model_cfg['num_attention_heads']} (KV: {model_cfg['num_key_value_heads']}), "
          f"Vocab: {model_cfg['vocab_size']}")
    print(f"  Instance: {INSTANCE_TYPE} ({gpu_type} x{gpu_count})")
    print(f"{'=' * 70}")

    # ── Weight memory ────────────────────────────────────────────────────
    layer_weight_bytes = get_memory_size_decoder_layer_weight_bytes(
        hidden_dim=model_cfg["hidden_size"],
        num_attention_head=model_cfg["num_attention_heads"],
        num_key_value_head=model_cfg["num_key_value_heads"],
        intermediate_dim=model_cfg["intermediate_size"],
        dtype=DTYPE,
    )
    embed_weight_bytes = get_memory_size_embedding_or_lm_head_weight_bytes(
        hidden_dim=model_cfg["hidden_size"],
        vocab_size=model_cfg["vocab_size"],
        dtype=DTYPE,
    )
    total_weight_bytes = (
        layer_weight_bytes * model_cfg["num_hidden_layers"]
        + embed_weight_bytes  # embedding
        + (0 if model_cfg["tie_word_embeddings"] else embed_weight_bytes)  # lm_head
    )
    print(f"\n  Weight Memory:")
    print(f"    Per layer:  {layer_weight_bytes / 1e6:.2f} MB")
    print(f"    Embedding:  {embed_weight_bytes / 1e6:.2f} MB")
    print(f"    Total:      {total_weight_bytes / 1e9:.4f} GB")
    print(f"    GPU Memory: {gpu_spec['memory_size']} MB")

    # ── Max batch size ───────────────────────────────────────────────────
    node_layer_comb = [(INSTANCE_TYPE, "dummy-az", model_cfg["num_hidden_layers"])]
    max_batch, num_blocks = get_global_batch_size(
        avg_input_len=INPUT_LEN,
        avg_output_len=OUTPUT_LEN,
        max_model_len=MAX_MODEL_LEN,
        hidden_dim=model_cfg["hidden_size"],
        num_attention_head=model_cfg["num_attention_heads"],
        num_kv_cache_head=model_cfg["num_key_value_heads"],
        total_num_layers=model_cfg["num_hidden_layers"],
        vocab_size=model_cfg["vocab_size"],
        intermediate_dim=model_cfg["intermediate_size"],
        gpu_mem_utilization=GPU_MEM_UTIL,
        node_layer_comb=node_layer_comb,
        dtype=DTYPE,
    )
    print(f"\n  Max Batch Size: {max_batch}")
    print(f"  Num KV Blocks: {num_blocks}")

    if max_batch <= 0:
        print(f"  ⚠ OOM: Cannot fit model on {INSTANCE_TYPE}")
        return None

    # ── Per-layer latency (batch=1) ──────────────────────────────────────
    prefill_per_layer = get_prefill_computation_latency_per_layer(
        gpu_type=gpu_type, gpu_count=gpu_count,
        input_len=INPUT_LEN, hidden_dim=model_cfg["hidden_size"],
        num_attention_head=model_cfg["num_attention_heads"],
        num_kv_cache_head=model_cfg["num_key_value_heads"],
        intermediate_dim=model_cfg["intermediate_size"],
        batch_size=1, dtype=DTYPE,
    )
    decode_per_layer = get_decoding_computation_latency_per_layer(
        gpu_type=gpu_type, gpu_count=gpu_count,
        input_len=INPUT_LEN, output_len=OUTPUT_LEN,
        hidden_dim=model_cfg["hidden_size"],
        num_attention_head=model_cfg["num_attention_heads"],
        num_kv_cache_head=model_cfg["num_key_value_heads"],
        intermediate_dim=model_cfg["intermediate_size"],
        batch_size=1, dtype=DTYPE,
    )
    print(f"\n  Per-layer Latency (batch=1):")
    print(f"    Prefill: {prefill_per_layer:.4f} ms")
    print(f"    Decode:  {decode_per_layer:.4f} ms")
    print(f"    Prefill total ({model_cfg['num_hidden_layers']}L): "
          f"{prefill_per_layer * model_cfg['num_hidden_layers']:.2f} ms")
    print(f"    Decode total ({model_cfg['num_hidden_layers']}L):  "
          f"{decode_per_layer * model_cfg['num_hidden_layers']:.2f} ms")

    # ── Throughput sweep ─────────────────────────────────────────────────
    if not batch_sizes:
        batch_sizes = []
        b = 1
        while b <= max_batch:
            batch_sizes.append(b)
            b *= 2
        if batch_sizes[-1] != max_batch:
            batch_sizes.append(int(max_batch))

    header = f"  {'Batch':>6} | {'Throughput':>12} | {'Latency(ms)':>12} | {'Blocks':>8}"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    results = []
    for bs in batch_sizes:
        if bs > max_batch:
            print(f"  {bs:>6} | {'OOM':>12} | {'OOM':>12} | {'OOM':>8}")
            results.append({"batch_size": bs, "status": "OOM"})
            continue

        throughput, latency, n_blocks = get_throughput(
            avg_input_len=INPUT_LEN,
            avg_output_len=OUTPUT_LEN,
            max_model_len=MAX_MODEL_LEN,
            hidden_dim=model_cfg["hidden_size"],
            num_attention_head=model_cfg["num_attention_heads"],
            num_kv_cache_head=model_cfg["num_key_value_heads"],
            total_num_layers=model_cfg["num_hidden_layers"],
            vocab_size=model_cfg["vocab_size"],
            intermediate_dim=model_cfg["intermediate_size"],
            gpu_mem_utilization=GPU_MEM_UTIL,
            node_layer_comb=node_layer_comb,
            dtype=DTYPE,
        )

        print(f"  {bs:>6} | {throughput:>12.4f} | {latency:>12.2f} | {n_blocks:>8}")
        results.append({
            "batch_size": bs,
            "throughput_rps": throughput,
            "total_latency_ms": latency,
            "num_blocks": n_blocks,
        })

    return {
        "model": model_name,
        "model_config": {k: v for k, v in model_cfg.items() if k != "model_name"},
        "instance": INSTANCE_TYPE,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "weight_memory_gb": total_weight_bytes / 1e9,
        "max_batch_size": int(max_batch),
        "num_blocks": int(num_blocks),
        "prefill_per_layer_ms": prefill_per_layer,
        "decode_per_layer_ms": decode_per_layer,
        "batch_results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Llama-3.2-3B vs Qwen3-4B estimator predictions on g6.xlarge"
    )
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: auto)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else None

    all_results = {}

    for model_name, model_cfg in MODELS.items():
        result = estimate_model(model_name, model_cfg, batch_sizes, args.verbose)
        if result:
            all_results[model_name] = result

    # ── Comparison Summary ───────────────────────────────────────────────
    if len(all_results) == 2:
        print(f"\n\n{'=' * 70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'=' * 70}")

        models = list(all_results.values())
        header = f"  {'Metric':<30} | {models[0]['model']:>18} | {models[1]['model']:>18}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        comparisons = [
            ("Layers", lambda m: m["model_config"]["num_hidden_layers"]),
            ("Hidden Size", lambda m: m["model_config"]["hidden_size"]),
            ("Intermediate Size", lambda m: m["model_config"]["intermediate_size"]),
            ("Vocab Size", lambda m: m["model_config"]["vocab_size"]),
            ("Weight Memory (GB)", lambda m: f"{m['weight_memory_gb']:.3f}"),
            ("Max Batch Size", lambda m: m["max_batch_size"]),
            ("Num KV Blocks", lambda m: m["num_blocks"]),
            ("Prefill/layer (ms)", lambda m: f"{m['prefill_per_layer_ms']:.4f}"),
            ("Decode/layer (ms)", lambda m: f"{m['decode_per_layer_ms']:.4f}"),
        ]

        for label, fn in comparisons:
            v0 = fn(models[0])
            v1 = fn(models[1])
            print(f"  {label:<30} | {str(v0):>18} | {str(v1):>18}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "estimated_llama3_3B_vs_qwen3_4B_g6.xlarge.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
