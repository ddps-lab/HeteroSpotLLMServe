#!/usr/bin/env python3
"""
Benchmark: ShuntServe optimizer performance vs beam search top-k.

Measures wall-clock algorithm time and predicted throughput for each
(model, cluster, top_k) combination.

Uses ProcessPoolExecutor with max_workers = cpu_count - 1 to run
experiments in parallel while isolating each run in its own process.

Logs are written to both the terminal and a timestamped log file
automatically via Python logging (no need for `tee`).

Usage:
    python3 benchmark_topk.py
"""

import sys
import os
import json
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta

# ── Path setup ────────────────────────────────────────────────────────
_root = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_root, ".git")):
    _root = os.path.dirname(_root)
sys.path.insert(0, os.path.join(_root, "ModelPlacement"))

import torch
import logging

# ── Logger setup (terminal handler first; file handler added in main) ─
_script_dir = os.path.dirname(os.path.abspath(__file__))
_kst = timezone(timedelta(hours=9))
_log_dir = os.path.join(_script_dir, "logs")
os.makedirs(_log_dir, exist_ok=True)

log = logging.getLogger("topk_benchmark")
log.setLevel(logging.INFO)
log.propagate = False

_fmt = logging.Formatter("%(asctime)s │ %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
log.addHandler(_sh)

# Suppress optimizer internal logging in workers
logging.getLogger("shuntserve_optimizer").setLevel(logging.CRITICAL)

from transformers import AutoConfig
from collections import Counter
from shuntserve_optimizer import run_test_case, Pipeline
from cluster_pool import ClusterPool
from hardware_specs import INSTANCE_SPEC


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

MODELS = {
    "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen3-32b": "Qwen/Qwen3-32B",
}

K_VALUES = list(range(1, 9)) + [16, 32, 64, 128]

CLUSTERS = {
    "eval_cluster": {
        "description": "Evaluation cluster (3type, 9 nodes, 24 GPUs)",
        "nodes": {
            "(spot)g5.12xlarge": 2,
            "(spot)g6.12xlarge": 3,
            "(spot)g6e.xlarge": 4,
        },
        "prices": {
            "(spot)g5.12xlarge": 2.2915,
            "(spot)g6.12xlarge": 1.9445,
            "(spot)g6e.xlarge": 0.7040,
        },
    },
    "large_hetero_cluster": {
        "description": "Large heterogeneous cluster (7 types, 15 nodes, 76 GPUs)",
        "nodes": {
            "(spot)g4dn.xlarge": 1,
            "(spot)g4dn.12xlarge": 1,
            "(spot)g4dn.metal": 1,
            "(spot)g5.xlarge": 1,
            "(spot)g5.12xlarge": 1,
            "(spot)g5.48xlarge": 1,
            "(spot)g6.xlarge": 1,
            "(spot)g6.12xlarge": 1,
            "(spot)g6.48xlarge": 1,
            "(spot)g6e.xlarge": 1,
            "(spot)g6e.12xlarge": 1,
            "(spot)g6e.48xlarge": 1,
            "(spot)p4d.24xlarge": 1,
            "(spot)p5.48xlarge": 1,
            "(spot)p6-b200.48xlarge": 1,
        },
        "prices": {
            "(spot)g4dn.xlarge": 0.2104,
            "(spot)g4dn.12xlarge": 1.5648,
            "(spot)g4dn.metal": 3.1296,
            "(spot)g5.xlarge": 0.4024,
            "(spot)g5.12xlarge": 2.2915,
            "(spot)g5.48xlarge": 6.5152,
            "(spot)g6.xlarge": 0.3219,
            "(spot)g6.12xlarge": 1.9445,
            "(spot)g6.48xlarge": 5.3402,
            "(spot)g6e.xlarge": 0.7040,
            "(spot)g6e.12xlarge": 4.1971,
            "(spot)g6e.48xlarge": 12.0525,
            "(spot)p4d.24xlarge": 8.7831,
            "(spot)p5.48xlarge": 22.016,
            "(spot)p6-b200.48xlarge": 45.5731,
        },
    },
}

COMMON = {
    "expected_input_len": 763,
    "expected_output_len": 232,
    "max_model_len": 8192,
    "gpu_mem_utilization": 0.85,
}


def build_model_config(model_key: str) -> Dict[str, Any]:
    model_name = MODELS[model_key]
    hf = AutoConfig.from_pretrained(model_name)
    head_dim = getattr(hf, "head_dim", None) or (hf.hidden_size // hf.num_attention_heads)
    return {
        **COMMON,
        "hidden_size": hf.hidden_size,
        "num_layers": hf.num_hidden_layers,
        "num_attention_heads": hf.num_attention_heads,
        "num_key_value_heads": getattr(hf, "num_key_value_heads", hf.num_attention_heads),
        "intermediate_size": hf.intermediate_size,
        "vocab_size": hf.vocab_size,
        "max_position_embeddings": hf.max_position_embeddings,
        "dtype": torch.float16,
        "head_dim": head_dim,
    }


def run_single_experiment(model_key: str, cluster_key: str, top_k: int) -> Dict[str, Any]:
    """
    Run the full iterative optimizer for one (model, cluster, top_k).
    Executed in a separate process for isolation.
    """
    # Suppress all logging in worker processes to avoid noise
    logging.disable(logging.CRITICAL)

    config = build_model_config(model_key)
    remaining = dict(CLUSTERS[cluster_key]["nodes"])
    prices = CLUSTERS[cluster_key].get("prices")
    pipelines: List[Pipeline] = []
    total_opt_time = 0.0

    wall_start = time.perf_counter()

    while True:
        if all(v == 0 for v in remaining.values()):
            break

        pool = ClusterPool(available_spot_nodes=remaining, spot_prices=prices)
        results, _, opt_time = run_test_case(
            config, budget=9999, latency_slo=99999999,
            cluster_pool=pool, max_stages=None, top_k=top_k,
            optimization_mode="soft_slo",
        )
        total_opt_time += opt_time

        if results and results[0].throughput <= 0:
            results = []

        if not results:
            # Residual merging
            if pipelines and any(v > 0 for v in remaining.values()):
                last_used = Counter(pipelines[-1].stages)
                merged = dict(remaining)
                for inst, cnt in last_used.items():
                    merged[inst] = merged.get(inst, 0) + cnt
                merged_pool = ClusterPool(available_spot_nodes=merged, spot_prices=prices)
                merged_res, _, m_time = run_test_case(
                    config, budget=9999, latency_slo=99999999,
                    cluster_pool=merged_pool, max_stages=None, top_k=top_k,
                    optimization_mode="only_throughput",
                )
                total_opt_time += m_time
                if merged_res:
                    pipelines[-1] = merged_res[0]
                for k in remaining:
                    remaining[k] = 0
            break

        best = results[0]
        pipelines.append(best)
        for inst, cnt in Counter(best.stages).items():
            if inst in remaining:
                remaining[inst] -= cnt

    wall_time = time.perf_counter() - wall_start
    total_thr = sum(p.throughput for p in pipelines)

    pipe_details = []
    for i, p in enumerate(pipelines, 1):
        tp = [INSTANCE_SPEC[inst]["gpu_count"] for inst in p.stages]
        pipe_details.append({
            "label": f"P{i}",
            "stages": list(p.stages),
            "tp_strategy": tp,
            "layer_partition": [int(l) for l in p.layer_per_stage],
            "throughput_rps": round(p.throughput, 4),
            "batch_size": p.global_batch_size,
            "num_blocks": p.num_blocks,
        })

    return {
        "model": model_key,
        "cluster": cluster_key,
        "top_k": top_k,
        "wall_time_sec": round(wall_time, 4),
        "optimizer_time_sec": round(total_opt_time, 4),
        "total_throughput_rps": round(total_thr, 4),
        "num_pipelines": len(pipelines),
        "pipelines": pipe_details,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="ShuntServe Top-K Beam Search Benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        choices=list(MODELS.keys()),
        help=f"Models to benchmark (default: all)\nChoices: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--clusters", nargs="+", default=None, metavar="CLUSTER",
        choices=list(CLUSTERS.keys()),
        help=f"Clusters to benchmark (default: all)\nChoices: {list(CLUSTERS.keys())}",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=None, metavar="K",
        help=f"Top-k values to benchmark (default: {K_VALUES})",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Max parallel workers (default: cpu_count - 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    models = args.models or list(MODELS.keys())
    clusters = args.clusters or list(CLUSTERS.keys())
    k_values = args.k or K_VALUES
    max_workers = args.workers or max(1, mp.cpu_count() - 1)

    # ── Attach file handler with config-based name ──────────────────
    m_tag = "+".join(models)
    c_tag = "+".join(clusters)
    k_tag = f"{k_values[0]}-{k_values[-1]}"
    _ts = datetime.now(_kst).strftime("%Y%m%d_%H%M%S")
    _log_filename = f"benchmark_m={m_tag}_c={c_tag}_k={k_tag}_{_ts}.log"
    _log_path = os.path.join(_log_dir, _log_filename)
    _fh = logging.FileHandler(_log_path, encoding="utf-8")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)

    experiments = [
        (m, c, k)
        for m in models
        for c in clusters
        for k in k_values
    ]
    total = len(experiments)

    log.info("════════════════════════════════════════════════════════")
    log.info("ShuntServe Top-K Beam Search Benchmark")
    log.info("════════════════════════════════════════════════════════")
    log.info(f"Models    : {models}")
    log.info(f"Clusters  : {clusters}")
    log.info(f"K values  : {k_values}")
    log.info(f"Total runs: {total}")
    log.info(f"Workers   : {max_workers} (cpu_count - 1)")
    log.info(f"Log file  : {_log_path}")
    log.info("────────────────────────────────────────────────────────")

    results = []
    completed = 0
    bench_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for m, c, k in experiments:
            f = executor.submit(run_single_experiment, m, c, k)
            future_map[f] = (m, c, k)

        for future in as_completed(future_map):
            spec = future_map[future]
            completed += 1
            try:
                r = future.result()
                results.append(r)
                log.info(
                    f"[{completed:>3d}/{total}] "
                    f"{r['model']:12s} | {r['cluster']:22s} | "
                    f"k={r['top_k']:>3d} | "
                    f"wall={r['wall_time_sec']:>8.2f}s | "
                    f"throughput={r['total_throughput_rps']:>8.3f} req/s | "
                    f"pipelines={r['num_pipelines']}"
                )
            except Exception as e:
                m, c, k = spec
                log.error(f"[{completed:>3d}/{total}] FAILED: {m} | {c} | k={k} | {e}")
                results.append({"model": m, "cluster": c, "top_k": k, "error": str(e)})

    bench_elapsed = time.perf_counter() - bench_start
    results.sort(key=lambda r: (r["model"], r["cluster"], r.get("top_k", 0)))

    # ── Save JSON ─────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")
    os.makedirs(output_dir, exist_ok=True)
    kst_now = datetime.now(_kst).strftime("%Y-%m-%d %H:%M:%S KST")

    output_filename = f"results_m={m_tag}_c={c_tag}_k={k_tag}.json"

    output = {
        "benchmark": "shuntserve_topk_sweep",
        "timestamp": kst_now,
        "total_wall_time_sec": round(bench_elapsed, 2),
        "max_workers": max_workers,
        "k_values": k_values,
        "models": {k: MODELS[k] for k in models},
        "clusters": {k: CLUSTERS[k] for k in clusters},
        "workload": COMMON,
        "results": results,
    }
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary Table ─────────────────────────────────────────────────
    log.info("")
    log.info("════════════════════════════════════════════════════════")
    log.info(f"RESULTS SUMMARY  (total elapsed: {bench_elapsed:.1f}s)")
    log.info("════════════════════════════════════════════════════════")

    for model_key in models:
        for cluster_key in clusters:
            cluster_desc = CLUSTERS[cluster_key]["description"]
            log.info("")
            log.info(f"  [{model_key}] [{cluster_desc}]")
            log.info(f"  {'k':>5s} │ {'wall_time':>10s} │ {'opt_time':>10s} │ {'throughput':>14s} │ {'pipelines':>9s}")
            log.info(f"  {'─' * 5}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 14}─┼─{'─' * 9}")

            subset = sorted(
                [r for r in results
                 if r.get("model") == model_key
                 and r.get("cluster") == cluster_key
                 and "error" not in r],
                key=lambda r: r["top_k"],
            )
            for r in subset:
                log.info(
                    f"  {r['top_k']:>5d} │ {r['wall_time_sec']:>9.2f}s │ "
                    f"{r['optimizer_time_sec']:>9.2f}s │ "
                    f"{r['total_throughput_rps']:>11.3f}    │ "
                    f"{r['num_pipelines']:>9d}"
                )

    log.info("")
    log.info(f"JSON results : {output_file}")
    log.info(f"Log file     : {_log_path}")


if __name__ == "__main__":
    main()
