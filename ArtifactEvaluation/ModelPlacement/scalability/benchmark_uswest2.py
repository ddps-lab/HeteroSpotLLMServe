#!/usr/bin/env python3
"""
Benchmark: ShuntServe optimizer top-k sweep on the full us-west-2 NVIDIA
GPU cluster (every NVIDIA instance type available in the region).

Kept separate from benchmark_topk.py because this sweep is much heavier
(46 instance types vs 15), so it adds:
  - per-experiment worker log files under logs/workers_<ts>/
    (FileHandler flushes per record → `tail -f` shows progress live)
  - per-layer DP progress lines from the optimizer; --debug additionally
    dumps the full DP table after every layer (large files)
  - per-worker address-space cap (--max-mem-gb): an over-budget experiment
    fails alone with MemoryError instead of OOM-killing the machine
  - explicit memory release between pipeline rounds: run_test_case returns
    the optimizer object (full DP table + caches); dropping it and calling
    malloc_trim returns freed pages to the OS so RSS does not ratchet up
    across rounds

────────────────────────────────────────────────────────────────────────
SETUP — copying to another machine
────────────────────────────────────────────────────────────────────────
This script is NOT standalone. On import it walks up the directory tree
until it finds a `.git` marker, then adds `<repo_root>/ModelPlacement` to
sys.path. So either scp the whole repo, or recreate this minimal layout:

    <root>/
    ├── .git/                        # marker only — `mkdir .git` is enough
    ├── ModelPlacement/
    │   ├── shuntserve_optimizer.py
    │   ├── estimator_utils.py
    │   ├── cluster_pool.py
    │   └── hardware_specs.py
    └── anywhere/benchmark_uswest2.py   # depth below <root> doesn't matter

    scp -r host:~/ShuntServe/ModelPlacement <root>/
    scp host:~/ShuntServe/ArtifactEvaluation/ModelPlacement/top_k_beam/benchmark_uswest2.py <root>/anywhere/

Requirements:
  - Linux (uses /proc/self/statm, resource.RLIMIT_AS, libc malloc_trim)
  - python3 with: torch, transformers (qwen3 needs >= 4.51)
  - Hugging Face auth for gated models (meta-llama/Llama-3.1-70B-Instruct):
    `huggingface-cli login` once, or pre-populate ~/.cache/huggingface.
    Only the model *config* is downloaded (a few KB), not the weights.

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────
    # Full sweep: 2 models × 12 k-values (1..8,16,32,64,128) — takes DAYS.
    python3 benchmark_uswest2.py

    # Recommended first run: one model, small k, single worker
    python3 benchmark_uswest2.py --models llama3-70b --k 1 --workers 1

    # Subset sweep with DP-table dumps and a tighter memory cap
    python3 benchmark_uswest2.py --models llama3-70b --k 1 2 4 8 \
                                 --workers 4 --debug --max-mem-gb 60

Options:
  --models M [M ...]   llama3-70b | qwen3-32b          (default: both)
  --k K [K ...]        top-k beam widths               (default: 1-8,16,32,64,128)
  --workers N          parallel worker processes        (default: cpu_count-1)
                       NOTE: each worker may use up to --max-mem-gb; keep
                       workers × max-mem-gb below machine RAM
  --debug              dump the full DP table after every layer into the
                       worker log (can reach GBs at large k)
  --max-mem-gb GB      per-worker address-space cap     (default: total RAM - 1 GB)
  --mem-per-k-gb GB    per-worker cap = GB × k, clamped to --max-mem-gb;
                       budget a parallel sweep as GB × sum(k) <= machine RAM

────────────────────────────────────────────────────────────────────────
MONITORING & OUTPUT
────────────────────────────────────────────────────────────────────────
  Main log     logs/benchmark_m=..._c=..._k=..._<ts>.log   (1 line per finished experiment)
  Worker logs  logs/workers_<ts>/m=<model>_c=<cluster>_k=<k>.log
               — per-round headers, per-layer timing/candidates/RSS/cache
                 sizes, picked pipelines; streamed live:
                     tail -f logs/workers_<ts>/m=llama3-70b_c=uswest2_full_cluster_k=1.log
               — filter just the layer progress:
                     tail -f <worker log> | grep "Layer"
  JSON         json/results_m=..._c=..._k=....json
               — per-experiment wall/opt time, throughput, pipeline details

Rough timing (k=1, llama3-70b, 46-type cluster): round 1 alone ≈ 30 min;
layer time grows ~linearly with layer index; cost scales further with k.
"""

import sys
import os
import gc
import json
import time
import ctypes
import argparse
import resource
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

log = logging.getLogger("uswest2_benchmark")
log.setLevel(logging.INFO)
log.propagate = False

_fmt = logging.Formatter("%(asctime)s │ %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
log.addHandler(_sh)

from transformers import AutoConfig
from collections import Counter
from shuntserve_optimizer import run_test_case, Pipeline, get_rss_gb
from cluster_pool import ClusterPool
from hardware_specs import INSTANCE_SPEC

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

MODELS = {
    "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen3-32b": "Qwen/Qwen3-32B",
}

K_VALUES = list(range(1, 9)) + [16, 32, 64, 128]

# All NVIDIA GPU instance types available in us-west-2, one node per type.
# Spot prices: EC2 API snapshot on 2026-06-12, lowest AZ price (Linux/UNIX).
#
# Included (46 types, 46 nodes, 139 GPUs):
#   - g4dn  (T4 ×1/4/8)        : 7 types, 17 GPUs
#   - g5g   (T4G ×1/2, ARM)    : 6 types,  8 GPUs — T4 silicon on Graviton2;
#                                 estimator only models the GPU, but real
#                                 deployment needs ARM-compatible vLLM
#   - g5    (A10G ×1/4/8)      : 8 types, 21 GPUs
#   - g6    (L4 ×1/4/8)        : 8 types, 21 GPUs
#   - gr6   (L4 ×1, high-RAM)  : 2 types,  2 GPUs — same GPU as g6, cheaper spot
#   - g6e   (L40S ×1/4/8)      : 8 types, 21 GPUs
#   - p4d/p4de (A100 40/80 ×8) : 2 types, 16 GPUs
#   - p5    (H100 ×1/8)        : 2 types,  9 GPUs
#   - p5en  (H200 ×8)          : 1 type,   8 GPUs
#   - p6-b200 (B200 ×8)        : 1 type,   8 GPUs
#   - p6-b300 (B300 ×8)        : 1 type,   8 GPUs — FP16 spec estimated ≈ B200
# Excluded (available in us-west-2 but not usable here):
#   - p5e.48xlarge  : no on-demand price (Capacity Blocks only) → not launchable
#   - g4ad          : AMD Radeon V520, not NVIDIA
#   - g6f / gr6f    : fractional L4 (0.25–0.5 GPU), gpu_count must be integer
#   - p3 / p3dn     : V100, not in GPU_SPEC (previous generation)
CLUSTERS = {
    # 46-type: every NVIDIA instance type in us-west-2, one node per type.
    "uswest2_full_cluster": {
        "description": "us-west-2 full NVIDIA cluster (46 instance types, 46*M nodes, 139 GPUs)",
        "nodes": {
            "(spot)g4dn.xlarge": 1,
            "(spot)g4dn.2xlarge": 1,
            "(spot)g4dn.4xlarge": 1,
            "(spot)g4dn.8xlarge": 1,
            "(spot)g4dn.16xlarge": 1,
            "(spot)g4dn.12xlarge": 1,
            "(spot)g4dn.metal": 1,
            "(spot)g5g.xlarge": 1,
            "(spot)g5g.2xlarge": 1,
            "(spot)g5g.4xlarge": 1,
            "(spot)g5g.8xlarge": 1,
            "(spot)g5g.16xlarge": 1,
            "(spot)g5g.metal": 1,
            "(spot)g5.xlarge": 1,
            "(spot)g5.2xlarge": 1,
            "(spot)g5.4xlarge": 1,
            "(spot)g5.8xlarge": 1,
            "(spot)g5.16xlarge": 1,
            "(spot)g5.12xlarge": 1,
            "(spot)g5.24xlarge": 1,
            "(spot)g5.48xlarge": 1,
            "(spot)g6.xlarge": 1,
            "(spot)g6.2xlarge": 1,
            "(spot)g6.4xlarge": 1,
            "(spot)g6.8xlarge": 1,
            "(spot)g6.16xlarge": 1,
            "(spot)g6.12xlarge": 1,
            "(spot)g6.24xlarge": 1,
            "(spot)g6.48xlarge": 1,
            "(spot)gr6.4xlarge": 1,
            "(spot)gr6.8xlarge": 1,
            "(spot)g6e.xlarge": 1,
            "(spot)g6e.2xlarge": 1,
            "(spot)g6e.4xlarge": 1,
            "(spot)g6e.8xlarge": 1,
            "(spot)g6e.16xlarge": 1,
            "(spot)g6e.12xlarge": 1,
            "(spot)g6e.24xlarge": 1,
            "(spot)g6e.48xlarge": 1,
            "(spot)p4d.24xlarge": 1,
            "(spot)p4de.24xlarge": 1,
            "(spot)p5.4xlarge": 1,
            "(spot)p5.48xlarge": 1,
            "(spot)p5en.48xlarge": 1,
            "(spot)p6-b200.48xlarge": 1,
            "(spot)p6-b300.48xlarge": 1,
        },
        "prices": {
            "(spot)g4dn.xlarge": 0.1998,
            "(spot)g4dn.2xlarge": 0.2577,
            "(spot)g4dn.4xlarge": 0.3834,
            "(spot)g4dn.8xlarge": 0.7623,
            "(spot)g4dn.16xlarge": 1.3833,
            "(spot)g4dn.12xlarge": 1.3584,
            "(spot)g4dn.metal": 3.4908,
            "(spot)g5g.xlarge": 0.122,
            "(spot)g5g.2xlarge": 0.1611,
            "(spot)g5g.4xlarge": 0.2636,
            "(spot)g5g.8xlarge": 0.3896,
            "(spot)g5g.16xlarge": 0.7884,
            "(spot)g5g.metal": 0.398,
            "(spot)g5.xlarge": 0.5501,
            "(spot)g5.2xlarge": 0.5263,
            "(spot)g5.4xlarge": 0.5503,
            "(spot)g5.8xlarge": 1.0524,
            "(spot)g5.16xlarge": 1.287,
            "(spot)g5.12xlarge": 3.5309,
            "(spot)g5.24xlarge": 4.0043,
            "(spot)g5.48xlarge": 6.7089,
            "(spot)g6.xlarge": 0.3561,
            "(spot)g6.2xlarge": 0.5535,
            "(spot)g6.4xlarge": 0.5812,
            "(spot)g6.8xlarge": 0.8312,
            "(spot)g6.16xlarge": 1.4345,
            "(spot)g6.12xlarge": 1.6448,
            "(spot)g6.24xlarge": 2.3192,
            "(spot)g6.48xlarge": 5.2547,
            "(spot)gr6.4xlarge": 0.4824,
            "(spot)gr6.8xlarge": 0.7193,
            "(spot)g6e.xlarge": 0.9048,
            "(spot)g6e.2xlarge": 1.0629,
            "(spot)g6e.4xlarge": 1.4975,
            "(spot)g6e.8xlarge": 1.3962,
            "(spot)g6e.16xlarge": 2.6257,
            "(spot)g6e.12xlarge": 3.0612,
            "(spot)g6e.24xlarge": 4.1323,
            "(spot)g6e.48xlarge": 8.3206,
            "(spot)p4d.24xlarge": 11.642,
            "(spot)p4de.24xlarge": 15.4059,
            "(spot)p5.4xlarge": 1.6634,
            "(spot)p5.48xlarge": 14.083,
            "(spot)p5en.48xlarge": 17.037,
            "(spot)p6-b200.48xlarge": 28.0809,
            "(spot)p6-b300.48xlarge": 33.0176,
        },
    },
    # 15-type: the original large_hetero cluster (7 GPU types).
    "large_hetero_cluster": {
        "description": "large heterogeneous cluster (7 GPU types, 15 instance types, 15*M nodes, 76 GPUs)",
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
            "(spot)g6e.xlarge": 0.704,
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


def free_round_memory():
    """Collect garbage and return freed pages to the OS (RSS visibly drops)."""
    gc.collect()
    if _libc is not None:
        _libc.malloc_trim(0)


def default_max_mem_gb() -> float:
    """Total physical RAM minus 1 GB headroom (fallback 150 if undetectable)."""
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
        return max(1.0, round(total - 1, 1))
    except (ValueError, OSError, AttributeError):
        return 150.0


def run_single_experiment(model_key: str, cluster_key: str, top_k: int,
                          debug: bool = False, worker_log_dir: str = None,
                          max_mem_gb: float = 150,
                          node_multiplier: int = 1,
                          first_pipeline_only: bool = False) -> Dict[str, Any]:
    """
    Run the iterative optimizer for one (model, cluster, top_k).
    Executed in a separate process for isolation.
    node_multiplier scales every node count (cluster-SIZE sweep axis).
    first_pipeline_only stops after round 1 (just finding the 1st pipeline) —
    far cheaper, so the 46-type cluster can be swept to much larger M.
    """
    if max_mem_gb:
        mem_limit = int(max_mem_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

    # Route optimizer logs to a per-experiment file; parallel workers would
    # interleave on stdout. DEBUG includes the per-layer DP table dump.
    logging.disable(logging.NOTSET)
    opt_log = logging.getLogger("shuntserve_optimizer")
    opt_log.propagate = False
    for h in opt_log.handlers[:]:
        opt_log.removeHandler(h)
        h.close()
    cluster_tag = cluster_key if node_multiplier == 1 else f"{cluster_key}_x{node_multiplier}"
    if first_pipeline_only:
        cluster_tag += "_p1"
    if worker_log_dir:
        level = logging.DEBUG if debug else logging.INFO
        log_path = os.path.join(worker_log_dir, f"m={model_key}_c={cluster_tag}_k={top_k}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s │ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        opt_log.addHandler(fh)
        opt_log.setLevel(level)
    else:
        logging.disable(logging.CRITICAL)

    config = build_model_config(model_key)
    remaining = {inst: cnt * node_multiplier
                 for inst, cnt in CLUSTERS[cluster_key]["nodes"].items()}
    prices = CLUSTERS[cluster_key].get("prices")
    pipelines: List[Pipeline] = []
    total_opt_time = 0.0
    round_idx = 0

    wall_start = time.perf_counter()

    while True:
        if all(v == 0 for v in remaining.values()):
            break

        round_idx += 1
        remaining_str = ", ".join(f"{i}×{n}" for i, n in remaining.items() if n > 0)
        opt_log.info(f"══ Round {round_idx} │ {model_key} │ {cluster_tag} │ k={top_k} │ remaining: {remaining_str}")
        round_start = time.perf_counter()

        pool = ClusterPool(available_spot_nodes=remaining, spot_prices=prices)
        results, optimizer, opt_time = run_test_case(
            config, budget=9999, latency_slo=99999999,
            cluster_pool=pool, max_stages=None, top_k=top_k,
            optimization_mode="soft_slo",
        )
        # The optimizer holds the full DP table + caches; without dropping it
        # here, the previous round's table stays alive through the next round.
        del optimizer
        free_round_memory()
        total_opt_time += opt_time

        round_wall = time.perf_counter() - round_start
        if results:
            best_str = f"best_thr={results[0].throughput:.3f} req/s"
        else:
            best_str = "no feasible pipeline"
        opt_log.info(
            f"══ Round {round_idx} done │ wall={round_wall:.2f}s │ opt={opt_time:.2f}s │ "
            f"{best_str} │ rss={get_rss_gb():.2f}GB"
        )

        if results and results[0].throughput <= 0:
            results = []

        if not results:
            # Residual merging
            if pipelines and any(v > 0 for v in remaining.values()):
                opt_log.info("══ Residual merge: re-optimizing last pipeline with leftover nodes")
                merge_start = time.perf_counter()
                last_used = Counter(pipelines[-1].stages)
                merged = dict(remaining)
                for inst, cnt in last_used.items():
                    merged[inst] = merged.get(inst, 0) + cnt
                merged_pool = ClusterPool(available_spot_nodes=merged, spot_prices=prices)
                merged_res, merge_optimizer, m_time = run_test_case(
                    config, budget=9999, latency_slo=99999999,
                    cluster_pool=merged_pool, max_stages=None, top_k=top_k,
                    optimization_mode="only_throughput",
                )
                del merge_optimizer
                free_round_memory()
                total_opt_time += m_time
                if merged_res:
                    pipelines[-1] = merged_res[0]
                for k in remaining:
                    remaining[k] = 0
                opt_log.info(
                    f"══ Residual merge done │ wall={time.perf_counter() - merge_start:.2f}s │ "
                    f"opt={m_time:.2f}s │ rss={get_rss_gb():.2f}GB"
                )
            break

        best = results[0]
        pipelines.append(best)
        stage_str = ", ".join(f"{i}:{l}L" for i, l in zip(best.stages, best.layer_per_stage))
        opt_log.info(f"   → picked P{len(pipelines)}: thr={best.throughput:.3f} req/s │ [{stage_str}]")
        if first_pipeline_only:
            opt_log.info("══ first-pipeline-only: stopping after round 1")
            break
        for inst, cnt in Counter(best.stages).items():
            if inst in remaining:
                remaining[inst] -= cnt

    wall_time = time.perf_counter() - wall_start
    total_thr = sum(p.throughput for p in pipelines)
    opt_log.info(
        f"■■ Experiment done │ {model_key} │ k={top_k} │ wall={wall_time:.2f}s │ "
        f"total_thr={total_thr:.3f} req/s │ pipelines={len(pipelines)}"
    )

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
        "node_multiplier": node_multiplier,
        "total_nodes": sum(cnt * node_multiplier
                           for cnt in CLUSTERS[cluster_key]["nodes"].values()),
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
        description="ShuntServe Top-K Benchmark — full us-west-2 NVIDIA cluster",
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
    parser.add_argument(
        "--debug", action="store_true",
        help="Per-experiment logs at DEBUG level (per-layer DP table dump; large files)",
    )
    parser.add_argument(
        "--max-mem-gb", type=float, default=default_max_mem_gb(), metavar="GB",
        help="Address-space limit per worker process; exceeding it fails only that\n"
             "experiment (MemoryError) instead of OOM-killing the machine\n"
             f"(default: total RAM - 1 GB = {default_max_mem_gb():.0f})",
    )
    parser.add_argument(
        "--node-multipliers", nargs="+", type=int, default=[1], metavar="M",
        help="Cluster-SIZE sweep: scale every node count by each M and run all\n"
             "(model, k, M) combinations as parallel workers in ONE invocation\n"
             "(e.g. --node-multipliers 1 2 3 4 6 8). Each (m,k,M) gets its own\n"
             "_xM-tagged worker log; the combined JSON holds the whole sweep.",
    )
    parser.add_argument(
        "--mem-per-k-gb", type=float, default=None, metavar="GB",
        help="If set, each worker's cap is GB × k (clamped to --max-mem-gb).\n"
             "Search memory grows roughly linearly with k, so this gives small-k\n"
             "workers proportionally small caps and lets you budget a parallel\n"
             "sweep as GB × sum(k values) <= machine RAM.",
    )
    parser.add_argument(
        "--first-pipeline-only", action="store_true",
        help="Stop after round 1 (just find the 1st pipeline) instead of the\n"
             "full extraction. Much cheaper, so the 46-type cluster can be swept\n"
             "to larger M. Output/log filenames get a _p1 tag (no overwrite).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    models = args.models or list(MODELS.keys())
    clusters = args.clusters or list(CLUSTERS.keys())
    k_values = args.k or K_VALUES
    multipliers = sorted(set(args.node_multipliers))
    max_workers = args.workers or max(1, mp.cpu_count() - 1)

    # ── Attach file handler with config-based name ──────────────────
    m_tag = "+".join(models)
    c_tag = "+".join(clusters)
    k_tag = f"{k_values[0]}-{k_values[-1]}"
    mult_tag = "M" + "+".join(str(M) for M in multipliers)
    if args.first_pipeline_only:
        mult_tag += "_p1"
    _ts = datetime.now(_kst).strftime("%Y%m%d_%H%M%S")
    _log_filename = f"benchmark_m={m_tag}_c={c_tag}_k={k_tag}_{mult_tag}_{_ts}.log"
    _log_path = os.path.join(_log_dir, _log_filename)
    _fh = logging.FileHandler(_log_path, encoding="utf-8")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)

    # Per-experiment worker logs (round/layer progress; --debug adds DP dumps)
    worker_log_dir = os.path.join(_log_dir, f"workers_{_ts}")
    os.makedirs(worker_log_dir, exist_ok=True)

    experiments = [
        (m, c, k, M)
        for m in models
        for c in clusters
        for k in k_values
        for M in multipliers
    ]
    total = len(experiments)

    def nodes_at(c, M):
        return sum(cnt * M for cnt in CLUSTERS[c]["nodes"].values())

    log.info("════════════════════════════════════════════════════════")
    log.info("ShuntServe Cluster-SIZE Sensitivity Sweep")
    log.info("════════════════════════════════════════════════════════")
    log.info(f"Models    : {models}")
    log.info(f"Clusters  : {clusters}")
    log.info(f"K values  : {k_values}")
    log.info(f"Node mult : {multipliers}  "
             f"(nodes: " + ", ".join(f"M{M}={nodes_at(clusters[0], M)}" for M in multipliers) + ")")
    log.info(f"Total runs: {total}  (models × clusters × k × M)")
    log.info(f"Mode      : {'first-pipeline-only (round 1)' if args.first_pipeline_only else 'full extraction'}")
    log.info(f"Workers   : {max_workers}")
    def worker_mem_gb(k: int) -> float:
        if args.mem_per_k_gb:
            return min(args.max_mem_gb, args.mem_per_k_gb * k)
        return args.max_mem_gb

    log.info(f"Debug     : {args.debug}")
    if args.mem_per_k_gb:
        caps = ", ".join(f"k={k}:{worker_mem_gb(k):.0f}GB" for k in k_values)
        log.info(f"Mem limit : per-k ({args.mem_per_k_gb} GB × k) → {caps}")
    else:
        log.info(f"Mem limit : {args.max_mem_gb} GB per worker process")
    log.info(f"Log file  : {_log_path}")
    log.info(f"Worker logs: {worker_log_dir}/  (tail -f for per-round/per-layer progress)")
    log.info("────────────────────────────────────────────────────────")

    results = []
    completed = 0
    bench_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for m, c, k, M in experiments:
            f = executor.submit(run_single_experiment, m, c, k,
                                args.debug, worker_log_dir, worker_mem_gb(k), M,
                                args.first_pipeline_only)
            future_map[f] = (m, c, k, M)

        for future in as_completed(future_map):
            spec = future_map[future]
            completed += 1
            try:
                r = future.result()
                results.append(r)
                log.info(
                    f"[{completed:>3d}/{total}] "
                    f"{r['model']:12s} | "
                    f"M={r['node_multiplier']:>2d} (N={r['total_nodes']:>3d}) | "
                    f"k={r['top_k']:>3d} | "
                    f"wall={r['wall_time_sec']:>8.2f}s | "
                    f"throughput={r['total_throughput_rps']:>8.3f} req/s | "
                    f"pipelines={r['num_pipelines']}"
                )
            except Exception as e:
                m, c, k, M = spec
                log.error(f"[{completed:>3d}/{total}] FAILED: {m} | {c} | k={k} | M={M} | {e}")
                results.append({"model": m, "cluster": c, "top_k": k,
                                "node_multiplier": M, "error": str(e)})

    bench_elapsed = time.perf_counter() - bench_start
    results.sort(key=lambda r: (r["model"], r["cluster"], r.get("node_multiplier", 0), r.get("top_k", 0)))

    # ── Save JSON ─────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")
    os.makedirs(output_dir, exist_ok=True)
    kst_now = datetime.now(_kst).strftime("%Y-%m-%d %H:%M:%S KST")

    output_filename = f"results_m={m_tag}_c={c_tag}_k={k_tag}_{mult_tag}.json"

    output = {
        "benchmark": "shuntserve_size_sensitivity_sweep",
        "timestamp": kst_now,
        "first_pipeline_only": args.first_pipeline_only,
        "total_wall_time_sec": round(bench_elapsed, 2),
        "max_workers": max_workers,
        "node_multipliers": multipliers,
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
            log.info(f"  {'M':>3s} │ {'N':>4s} │ {'k':>3s} │ {'wall_time':>10s} │ {'opt_time':>10s} │ {'throughput':>14s} │ {'pipelines':>9s}")
            log.info(f"  {'─' * 3}─┼─{'─' * 4}─┼─{'─' * 3}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 14}─┼─{'─' * 9}")

            subset = sorted(
                [r for r in results
                 if r.get("model") == model_key
                 and r.get("cluster") == cluster_key
                 and "error" not in r],
                key=lambda r: (r["node_multiplier"], r["top_k"]),
            )
            for r in subset:
                log.info(
                    f"  {r['node_multiplier']:>3d} │ {r['total_nodes']:>4d} │ {r['top_k']:>3d} │ "
                    f"{r['wall_time_sec']:>9.2f}s │ "
                    f"{r['optimizer_time_sec']:>9.2f}s │ "
                    f"{r['total_throughput_rps']:>11.3f}    │ "
                    f"{r['num_pipelines']:>9d}"
                )

    log.info("")
    log.info(f"JSON results : {output_file}")
    log.info(f"Log file     : {_log_path}")


if __name__ == "__main__":
    main()
