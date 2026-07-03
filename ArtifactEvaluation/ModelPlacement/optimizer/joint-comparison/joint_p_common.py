"""
joint_p_common.py — Shared engine for the "joint vs. greedy pipeline extraction" study.

Motivation (reviewer R2#11):
    "The step where iteratively extract pipelines is critical but unspecified.
     Has greedy extraction been compared against jointly optimizing all K
     pipelines (for K=2, joint optimization is tractable)?"

The deployed ShuntServe optimizer extracts pipelines GREEDILY: it runs the
beam-search DP over the whole cluster, keeps the single best (rank-1) pipeline,
subtracts its nodes, and repeats until the cluster is exhausted. This yields a
variable number K of pipelines.

This module brute-forces the JOINT alternative for a *fixed* number of pipelines
p = K:  enumerate EVERY way to split the cluster's nodes into exactly p
non-empty groups, optimize one pipeline per group with the *same* per-pipeline
optimizer the paper uses, and rank every partition by the total (summed)
system throughput.  The greedy solution is one specific partition; we locate its
rank inside the full joint ranking.

Key efficiency insight: a group's optimal pipeline depends ONLY on (model config,
the group's node multiset, prices) — it is independent of the other groups
(separate node sets, no cross-pipeline interference in the analytical model).
So we MEMOIZE the per-group optimization over the (at most) 59 unique non-empty
sub-clusters and reuse those results across all partitions and all p.  This is
what makes p=4 just as tractable as p=2.

Soundness of memoization: greedy iteration-1 optimizes over the full cluster and
its rank-1 pipeline uses some node subset S.  Optimizing over S alone returns the
same pipeline, because S's feasible set is a subset of the full feasible set and
the chosen pipeline was already the global maximum.  Hence greedy's per-pipeline
throughputs equal our memo values exactly (verified at runtime).
"""

import os
import sys
import json
import time
import logging
from collections import Counter, defaultdict
from itertools import product

# HF config loading uses the ambient environment: network + `hf auth login`, or
# the local cache. To force cache-only (offline), export HF_HUB_OFFLINE=1 yourself.

# ── Locate repo root and add the (root-level) ModelPlacement package to path ──
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, os.path.join(REPO_ROOT, "ModelPlacement"))
del _d

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from shuntserve_optimizer import run_test_case  # noqa: E402
from cluster_pool import ClusterPool  # noqa: E402
from hardware_specs import INSTANCE_SPEC  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Cluster definition (identical to the greedy baseline scripts / the paper)
# ─────────────────────────────────────────────────────────────────────────────
# group/partition tuples are always ordered as (n_g5, n_g6, n_g6e)
INSTANCE_ORDER = ["(spot)g5.12xlarge", "(spot)g6.12xlarge", "(spot)g6e.xlarge"]
SHORT = {"(spot)g5.12xlarge": "g5", "(spot)g6.12xlarge": "g6", "(spot)g6e.xlarge": "g6e"}

CLUSTER = (2, 3, 4)  # g5.12xlarge x2, g6.12xlarge x3, g6e.xlarge x4  (9 nodes, 24 GPUs)

PRICES = {
    "(spot)g5.12xlarge": 2.2915,
    "(spot)g6.12xlarge": 1.9445,
    "(spot)g6e.xlarge": 0.7040,
}
TP_OF = {inst: INSTANCE_SPEC[inst]["gpu_count"] for inst in INSTANCE_ORDER}  # g5=4,g6=4,g6e=1

MODELS = {
    "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen3-32b": "Qwen/Qwen3-32B",
}

# Optimizer hyper-parameters — EXACTLY the ones the greedy baseline scripts use.
TOP_K = 3
MAX_STAGES = 13
BUDGET = 9999
LATENCY_SLO = 99999999
MODES = ("soft_slo", "only_throughput")
PRIMARY_MODE = "soft_slo"  # matches the paper's per-pipeline selection criterion

# ─────────────────────────────────────────────────────────────────────────────
# Greedy reference solutions — produced by the current code and verified to
# reproduce the paper's predicted_total_throughput_rps
# (ReferenceData/.../offline_shuntserve.json: Llama 2.8305, Qwen 9.4241).
# Each pipeline records the node multiset (group tuple) it occupies.
# ─────────────────────────────────────────────────────────────────────────────
GREEDY_REF = {
    "llama3-70b": {
        "K": 2,
        "total": 2.8305132690479464,
        "pipelines": [
            {"group": (0, 2, 4), "tp": [4, 4, 1, 1, 1, 1], "thr": 1.9951945710714936},
            {"group": (2, 1, 0), "tp": [4, 4, 4], "thr": 0.835318697976453},
        ],
    },
    "qwen3-32b": {
        "K": 4,
        "total": 9.424062756635378,
        "pipelines": [
            {"group": (0, 0, 4), "tp": [1, 1, 1, 1], "thr": 5.865183789888193},
            {"group": (0, 3, 0), "tp": [4, 4, 4], "thr": 2.627769793087521},
            {"group": (1, 0, 0), "tp": [4], "thr": 0.46555458682983164},
            {"group": (1, 0, 0), "tp": [4], "thr": 0.46555458682983164},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Config builder (mirrors the greedy baseline scripts exactly)
# ─────────────────────────────────────────────────────────────────────────────
def build_config(model_name):
    mc = AutoConfig.from_pretrained(model_name)
    head_dim = getattr(mc, "head_dim", None) or (mc.hidden_size // mc.num_attention_heads)
    return {
        "expected_input_len": 763,
        "expected_output_len": 232,
        "hidden_size": mc.hidden_size,
        "num_layers": mc.num_hidden_layers,
        "num_attention_heads": mc.num_attention_heads,
        "num_key_value_heads": getattr(mc, "num_key_value_heads", mc.num_attention_heads),
        "intermediate_size": mc.intermediate_size,
        "vocab_size": mc.vocab_size,
        "max_position_embeddings": mc.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.85,
        "head_dim": head_dim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Group / partition helpers
# ─────────────────────────────────────────────────────────────────────────────
def group_to_nodes(group):
    """(n5,n6,n6e) -> {instance: count} (only non-zero), for ClusterPool."""
    return {INSTANCE_ORDER[i]: group[i] for i in range(3) if group[i] > 0}


def group_str(group):
    parts = [f"{SHORT[INSTANCE_ORDER[i]]}x{group[i]}" for i in range(3) if group[i] > 0]
    return "{" + ",".join(parts) + "}" if parts else "{}"


def group_size(group):
    return sum(group)


def all_unique_groups(cluster=CLUSTER):
    """Every non-empty sub-multiset of the cluster (3*4*5 - 1 = 59 groups)."""
    out = []
    for a in range(cluster[0] + 1):
        for b in range(cluster[1] + 1):
            for c in range(cluster[2] + 1):
                if a or b or c:
                    out.append((a, b, c))
    return out


def _compositions(n, parts):
    """All ordered tuples of `parts` non-negative ints summing to n (stars & bars)."""
    if parts == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in _compositions(n - first, parts - 1):
            yield (first,) + rest


def enumerate_partitions(cluster, p):
    """
    All UNORDERED partitions of the cluster node-multiset into EXACTLY p
    non-empty groups.  Returns a sorted list of tuples-of-group-tuples.
    """
    per_type_comps = [list(_compositions(cluster[t], p)) for t in range(3)]
    seen = set()
    out = []
    for combo in product(*per_type_comps):
        # combo[t] = how many of type t go to each of the p bins
        groups = []
        ok = True
        for j in range(p):
            g = (combo[0][j], combo[1][j], combo[2][j])
            if sum(g) == 0:          # empty bin -> not a partition into p non-empty parts
                ok = False
                break
            groups.append(g)
        if not ok:
            continue
        canon = tuple(sorted(groups))
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
    out.sort()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-group optimization (the heavy work) — runs in worker processes
# ─────────────────────────────────────────────────────────────────────────────
_WORKER = {}


def _init_worker(model_name):
    logging.disable(logging.CRITICAL)  # silence the per-layer DP progress logs
    _WORKER["config"] = build_config(model_name)


def _summarize(group, mode, results, opt_time, wall):
    """Extract a JSON-serializable record for the rank-1 pipeline of a group."""
    feasible = bool(results) and results[0].throughput > 0
    candidates = []
    for pl in (results or []):
        candidates.append({
            "throughput": pl.throughput,
            "tp": [TP_OF[s] for s in pl.stages],
        })
    rec = {
        "group": list(group),
        "mode": mode,
        "feasible": feasible,
        "opt_time": opt_time,
        "wall_time": wall,
        "n_results": len(results or []),
        "candidates": candidates,
        # best-throughput pipeline among the returned candidates (cheap ceiling
        # proxy when mode == soft_slo)
        "best_thr_in_candidates": max((c["throughput"] for c in candidates), default=0.0),
    }
    if feasible:
        best = results[0]
        used = Counter(best.stages)
        used_t = tuple(used.get(INSTANCE_ORDER[i], 0) for i in range(3))
        idle = tuple(group[i] - used_t[i] for i in range(3))
        rec.update({
            "throughput": best.throughput,
            "cost": best.cost,
            "efficiency": (best.throughput / best.cost) if best.cost else 0.0,
            "tp": [TP_OF[s] for s in best.stages],
            "stages": list(best.stages),
            "layer_per_stage": [int(x) for x in best.layer_per_stage],
            "num_stages": len(best.stages),
            "nodes_used": list(used_t),
            "nodes_idle": list(idle),
            "num_blocks": best.num_blocks,
            "global_batch_size": best.global_batch_size,
            "latency_per_global_batch": best.latency_per_global_batch,
        })
    else:
        rec.update({
            "throughput": 0.0, "cost": 0.0, "efficiency": 0.0, "tp": [],
            "stages": [], "layer_per_stage": [], "num_stages": 0,
            "nodes_used": [0, 0, 0], "nodes_idle": list(group),
            "num_blocks": 0, "global_batch_size": 0, "latency_per_global_batch": 0.0,
        })
    return rec


def _optimize_group_task(task):
    group, mode = task
    cfg = _WORKER["config"]
    cp = ClusterPool(available_spot_nodes=group_to_nodes(group), spot_prices=PRICES)
    t0 = time.time()
    results, _, opt_time = run_test_case(
        cfg, budget=BUDGET, latency_slo=LATENCY_SLO, cluster_pool=cp,
        max_stages=MAX_STAGES, top_k=TOP_K, optimization_mode=mode,
    )
    wall = time.time() - t0
    # treat throughput<=0 rank-1 as infeasible, exactly like the greedy scripts
    if results and results[0].throughput <= 0:
        results = []
    return (tuple(group), mode, _summarize(group, mode, results, opt_time, wall))


# ─────────────────────────────────────────────────────────────────────────────
# Memo (parallel) with on-disk cache
# ─────────────────────────────────────────────────────────────────────────────
def _memo_key(group, mode):
    return f"{group[0]}_{group[1]}_{group[2]}|{mode}"


def compute_memo(model_short, processes=None, cache_path=None, refresh=False, log=print,
                 modes=MODES):
    """
    Optimize the rank-1 pipeline for every unique non-empty sub-cluster, in the
    requested modes, in parallel.  Returns memo where memo[(group, mode)] = record.
    Cached to `cache_path` (JSON) keyed by model; pass cache_path=None for a fresh
    in-memory run (used for clean timing).
    """
    model_name = MODELS[model_short]
    groups = all_unique_groups()
    tasks = [(g, m) for g in groups for m in modes]

    # ── load cache if valid ──
    cached = {}
    if cache_path and os.path.exists(cache_path) and not refresh:
        try:
            with open(cache_path) as f:
                blob = json.load(f)
            if blob.get("model") == model_name and blob.get("config_sig") == _config_sig(model_name):
                cached = blob.get("records", {})
                log(f"[memo] loaded {len(cached)} cached group/mode records from {cache_path}")
        except Exception as e:  # noqa: BLE001
            log(f"[memo] cache ignored ({e})")

    todo = [t for t in tasks if _memo_key(t[0], t[1]) not in cached]
    log(f"[memo] {len(tasks)} group/mode tasks total; {len(todo)} to compute, "
        f"{len(tasks) - len(todo)} from cache")

    records = dict(cached)
    if todo:
        from multiprocessing import Pool
        nproc = processes or max(1, (os.cpu_count() or 2) - 1)
        log(f"[memo] computing {len(todo)} tasks on {nproc} processes ...")
        wall0 = time.time()
        done = 0
        with Pool(processes=nproc, initializer=_init_worker, initargs=(model_name,)) as pool:
            for group, mode, rec in pool.imap_unordered(_optimize_group_task, todo, chunksize=1):
                records[_memo_key(group, mode)] = rec
                done += 1
                if done % 10 == 0 or done == len(todo):
                    log(f"[memo]   {done}/{len(todo)} done "
                        f"({time.time() - wall0:.1f}s elapsed)")
        wall = time.time() - wall0
        log(f"[memo] parallel compute wall-clock: {wall:.1f}s")
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"model": model_name, "config_sig": _config_sig(model_name),
                           "records": records}, f)
            log(f"[memo] wrote cache -> {cache_path}")

    memo = {}
    for g in groups:
        for m in modes:
            memo[(g, m)] = records[_memo_key(g, m)]
    return memo


_CONFIG_SIG_CACHE = {}


def _config_sig(model_name):
    if model_name not in _CONFIG_SIG_CACHE:
        c = build_config(model_name)
        _CONFIG_SIG_CACHE[model_name] = (
            f"L{c['num_layers']}_H{c['hidden_size']}_in{c['expected_input_len']}"
            f"_out{c['expected_output_len']}_mem{c['gpu_mem_utilization']}"
            f"_topk{TOP_K}_stages{MAX_STAGES}"
        )
    return _CONFIG_SIG_CACHE[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# Partition evaluation & ranking
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_partition(partition, memo, mode):
    parts = []
    total = 0.0
    n_feasible = 0
    for g in partition:
        rec = memo[(g, mode)]
        total += rec["throughput"]
        n_feasible += int(rec["feasible"])
        parts.append({"group": list(g), "throughput": rec["throughput"],
                      "feasible": rec["feasible"], "tp": rec["tp"],
                      "nodes_idle": rec["nodes_idle"]})
    return {
        "partition": [list(g) for g in partition],
        "total_throughput": total,
        "n_feasible_parts": n_feasible,
        "all_feasible": n_feasible == len(partition),
        "parts": parts,
    }


def rank_partitions(partitions, memo, mode):
    evals = [evaluate_partition(p, memo, mode) for p in partitions]
    evals.sort(key=lambda e: (e["total_throughput"], e["n_feasible_parts"]), reverse=True)
    for i, e in enumerate(evals, 1):
        e["rank"] = i
    return evals


def find_partition_rank(ranked, target_partition):
    """Locate a specific partition (canonicalized) inside a ranked list."""
    canon = tuple(sorted(tuple(g) for g in target_partition))
    for e in ranked:
        if tuple(sorted(tuple(g) for g in e["partition"])) == canon:
            return e
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm-time accounting
# ─────────────────────────────────────────────────────────────────────────────
def algorithm_time_report(partitions, memo, mode):
    """
    Characterize the cost of the JOINT brute force for this p:
      - unique groups that appear in any partition (these are what you must
        optimize, once each, with memoization)
      - memoized optimizer time  = sum of opt_time over those unique groups
      - naive optimizer time     = sum over partitions of sum of part opt_times
                                    (i.e. re-optimizing every group occurrence)
    """
    unique_groups = set()
    for part in partitions:
        unique_groups.update(part)
    memoized = sum(memo[(g, mode)]["opt_time"] for g in unique_groups)
    naive = 0.0
    for part in partitions:
        for g in part:
            naive += memo[(g, mode)]["opt_time"]
    per_group = {group_str(g): memo[(g, mode)]["opt_time"] for g in unique_groups}
    return {
        "num_partitions": len(partitions),
        "num_unique_groups_used": len(unique_groups),
        "memoized_optimizer_time_s": memoized,
        "naive_optimizer_time_s": naive,
        "max_single_group_time_s": max(per_group.values()) if per_group else 0.0,
        "num_optimizer_invocations_memoized": len(unique_groups),
        "num_optimizer_invocations_naive": sum(len(p) for p in partitions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Greedy reference helpers
# ─────────────────────────────────────────────────────────────────────────────
def greedy_solution_total(model_short, memo, mode=PRIMARY_MODE):
    """Re-derive the greedy total from the memo and cross-check vs the paper."""
    ref = GREEDY_REF[model_short]
    total = 0.0
    parts = []
    for pl in ref["pipelines"]:
        g = tuple(pl["group"])
        rec = memo[(g, mode)]
        total += rec["throughput"]
        parts.append({"group": list(g), "memo_thr": rec["throughput"],
                      "ref_thr": pl["thr"], "tp": rec["tp"]})
    return {"K": ref["K"], "paper_total": ref["total"], "memo_total": total, "parts": parts}


# ─────────────────────────────────────────────────────────────────────────────
# Subset-DP: GLOBAL joint optimum over ALL partitions (any #pipelines), WITHOUT
# enumerating partitions.  best[S] = max over non-empty sub-group T<=S of
# (thr[T] + best[S-T]).  States = count-vectors (a,b,c); O(prod (n_i+1)(n_i+2)/2).
# ─────────────────────────────────────────────────────────────────────────────
def joint_optimum_dp(memo, mode=PRIMARY_MODE, cluster=CLUSTER):
    t0 = time.time()
    thr = {}
    for g in all_unique_groups(cluster):
        rec = memo[(g, mode)]
        thr[g] = rec["throughput"] if rec["feasible"] else 0.0

    def subgroups(S):
        for a in range(S[0] + 1):
            for b in range(S[1] + 1):
                for c in range(S[2] + 1):
                    if a or b or c:
                        yield (a, b, c)

    best = {(0, 0, 0): (0.0, [])}
    states = [(a, b, c) for a in range(cluster[0] + 1)
              for b in range(cluster[1] + 1) for c in range(cluster[2] + 1)]
    states.sort(key=sum)  # increasing total size → subproblems ready
    for S in states:
        if S == (0, 0, 0):
            continue
        bv, bp = float("-inf"), None
        for T in subgroups(S):
            rem = (S[0] - T[0], S[1] - T[1], S[2] - T[2])
            v = thr[T] + best[rem][0]
            if v > bv:
                bv, bp = v, [T] + best[rem][1]
        best[S] = (bv, bp)

    total, partition = best[cluster]
    n_feasible = sum(1 for g in partition if thr[g] > 0)
    return {
        "total": total,
        "partition": partition,
        "num_groups": len(partition),
        "num_feasible_groups": n_feasible,
        "dp_time": time.time() - t0,
    }


def run_greedy_timed(model_short, log=lambda *a: None, mode="soft_slo",
                     residual_merge=True, top_k=TOP_K):
    """
    Greedy extraction loop, instrumented for timing.  Uses the SAME unmodified
    run_test_case.  Defaults reproduce the deployed greedy (soft_slo, top_k=3,
    residual-merge).  Set mode="only_throughput", residual_merge=False for a
    throughput-greedy that simply STOPS when no feasible pipeline remains.
    """
    cfg = build_config(MODELS[model_short])
    remaining = {INSTANCE_ORDER[i]: CLUSTER[i] for i in range(3)}
    pipelines, opt_time, n_calls, calls = [], 0.0, 0, []
    wall0 = time.time()
    while any(v > 0 for v in remaining.values()):
        cp = ClusterPool(available_spot_nodes={k: v for k, v in remaining.items() if v > 0},
                         spot_prices=PRICES)
        results, _, ot = run_test_case(cfg, budget=BUDGET, latency_slo=LATENCY_SLO,
                                       cluster_pool=cp, max_stages=MAX_STAGES, top_k=top_k,
                                       optimization_mode=mode)
        opt_time += ot
        n_calls += 1
        if results and results[0].throughput <= 0:
            results = []
        if not results:
            calls.append({"iter": n_calls, "phase": "extract(no feasible)",
                          "opt_time_s": ot, "throughput": 0.0, "nodes": {}})
            # residual merge into the last pipeline (re-optimize for throughput)
            if residual_merge and pipelines and any(v > 0 for v in remaining.values()):
                last_used = Counter(pipelines[-1].stages)
                merged = {k: v for k, v in remaining.items() if v > 0}
                for inst, cnt in last_used.items():
                    merged[inst] = merged.get(inst, 0) + cnt
                cp2 = ClusterPool(available_spot_nodes=merged, spot_prices=PRICES)
                mres, _, mt = run_test_case(cfg, budget=BUDGET, latency_slo=LATENCY_SLO,
                                            cluster_pool=cp2, max_stages=MAX_STAGES, top_k=TOP_K,
                                            optimization_mode="only_throughput")
                opt_time += mt
                n_calls += 1
                calls.append({"iter": n_calls, "phase": "residual_merge", "opt_time_s": mt,
                              "throughput": mres[0].throughput if mres else 0.0,
                              "nodes": dict(Counter(mres[0].stages)) if mres else {}})
                if mres:
                    pipelines[-1] = mres[0]
                for k in remaining:
                    remaining[k] = 0
            break
        best = results[0]
        pipelines.append(best)
        calls.append({"iter": n_calls, "phase": "extract", "opt_time_s": ot,
                      "throughput": best.throughput,
                      "nodes": {SHORT.get(k, k): v for k, v in Counter(best.stages).items()},
                      "tp": [TP_OF[s] for s in best.stages]})
        for inst, cnt in Counter(best.stages).items():
            if inst in remaining:
                remaining[inst] -= cnt
    return {
        "K": len(pipelines),
        "total": sum(p.throughput for p in pipelines),
        "opt_time": opt_time,
        "wall": time.time() - wall0,
        "n_calls": n_calls,
        "calls": calls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment driver (shared by the three joint-p/*.py scripts)
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_partition(part_eval):
    bits = []
    for pr in part_eval["parts"]:
        tag = "" if pr["feasible"] else "✗"
        bits.append(f"{group_str(tuple(pr['group']))}{tag}->{pr['throughput']:.3f}")
    return "  +  ".join(bits)


def run_joint_experiment(model_short, p, out_dir=None, processes=None,
                         top_n_print=15, refresh_memo=False):
    model_name = MODELS[model_short]
    here = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   model_short, "joint-p", "results")
    os.makedirs(here, exist_ok=True)
    cache_path = os.path.join(here, f"memo_{model_short}.json")

    bar = "=" * 84
    print(bar)
    print(f"JOINT vs GREEDY pipeline extraction  —  model={model_name}  |  p={p}")
    print(f"Cluster (g5,g6,g6e) = {CLUSTER}  ({group_size(CLUSTER)} nodes, "
          f"{sum(CLUSTER[i]*TP_OF[INSTANCE_ORDER[i]] for i in range(3))} GPUs)")
    print(f"Per-pipeline optimizer: beam-DP, top_k={TOP_K}, max_stages={MAX_STAGES}, "
          f"modes={MODES}")
    print(bar)

    # ── 1. Memoized per-group optimization (parallel) ──
    t_wall0 = time.time()
    memo = compute_memo(model_short, processes=processes, cache_path=cache_path,
                        refresh=refresh_memo)
    memo_wall = time.time() - t_wall0

    # ── 2. Enumerate all p-group partitions ──
    partitions = enumerate_partitions(CLUSTER, p)
    print(f"\n[enum] partitions of {CLUSTER} into exactly {p} non-empty groups: "
          f"{len(partitions)}")

    out = {"model": model_name, "model_short": model_short, "p": p,
           "cluster": list(CLUSTER), "prices": PRICES,
           "optimizer": {"top_k": TOP_K, "max_stages": MAX_STAGES},
           "num_partitions": len(partitions), "memo_wall_clock_s": memo_wall,
           "rankings": {}, "algorithm_time": {}, "greedy": {}}

    # ── 3. Rank under each selection criterion ──
    for mode in MODES:
        ranked = rank_partitions(partitions, memo, mode)
        algo = algorithm_time_report(partitions, memo, mode)
        out["rankings"][mode] = ranked
        out["algorithm_time"][mode] = algo

        label = ("PRIMARY (soft_slo — same per-pipeline criterion as the paper)"
                 if mode == PRIMARY_MODE else
                 "SECONDARY (only_throughput — per-pipeline throughput ceiling)")
        print("\n" + "-" * 84)
        print(f"RANKING under mode = {mode}   [{label}]")
        print("-" * 84)
        best = ranked[0]
        print(f"  #1 (JOINT-OPTIMAL): total = {best['total_throughput']:.4f} req/s   "
              f"all_feasible={best['all_feasible']}")
        print(f"      {_fmt_partition(best)}")
        print(f"\n  Top {min(top_n_print, len(ranked))} of {len(ranked)} partitions:")
        for e in ranked[:top_n_print]:
            print(f"   #{e['rank']:>3}  total={e['total_throughput']:7.4f}  "
                  f"feas={e['n_feasible_parts']}/{p}   {_fmt_partition(e)}")

        print(f"\n  [algorithm time | mode={mode}]")
        print(f"    partitions evaluated              : {algo['num_partitions']}")
        print(f"    unique groups optimized (memoized): {algo['num_unique_groups_used']}")
        print(f"    optimizer time, MEMOIZED          : {algo['memoized_optimizer_time_s']:.2f}s "
              f"({algo['num_optimizer_invocations_memoized']} invocations)")
        print(f"    optimizer time, NAIVE (no memo)   : {algo['naive_optimizer_time_s']:.2f}s "
              f"({algo['num_optimizer_invocations_naive']} invocations)")
        print(f"    slowest single group optimize     : {algo['max_single_group_time_s']:.2f}s")
        print(f"    measured parallel memo wall-clock : {memo_wall:.2f}s "
              f"(covers ALL groups & both modes, {os.cpu_count()} CPUs)")

    # ── 4. Greedy placement & comparison (primary mode) ──
    gd = greedy_solution_total(model_short, memo, mode=PRIMARY_MODE)
    out["greedy"] = gd
    ranked_primary = out["rankings"][PRIMARY_MODE]
    best_primary = ranked_primary[0]
    print("\n" + "=" * 84)
    print("GREEDY vs JOINT  (primary mode = soft_slo)")
    print("=" * 84)
    print(f"  greedy K = {gd['K']} pipelines")
    print(f"  greedy total throughput : paper={gd['paper_total']:.4f}  "
          f"memo-recomputed={gd['memo_total']:.4f}  "
          f"(match={abs(gd['paper_total']-gd['memo_total'])<1e-6})")
    for pr in gd["parts"]:
        print(f"      {group_str(tuple(pr['group']))} -> memo {pr['memo_thr']:.4f} "
              f"(paper {pr['ref_thr']:.4f})  TP={pr['tp']}")
    print(f"  joint-optimal (p={p}) total : {best_primary['total_throughput']:.4f}")
    print(f"      {_fmt_partition(best_primary)}")

    if gd["K"] == p:
        g_part = [pl["group"] for pl in GREEDY_REF[model_short]["pipelines"]]
        g_in_rank = find_partition_rank(ranked_primary, g_part)
        if g_in_rank is not None:
            out["greedy"]["rank_in_joint"] = g_in_rank["rank"]
            out["greedy"]["rank_total"] = g_in_rank["total_throughput"]
            gap = best_primary["total_throughput"] - g_in_rank["total_throughput"]
            rel = 100.0 * gap / g_in_rank["total_throughput"] if g_in_rank["total_throughput"] else 0.0
            print(f"\n  >>> greedy's partition ranks #{g_in_rank['rank']} of {len(ranked_primary)} "
                  f"(K=p={p}, DIRECTLY comparable)")
            print(f"  >>> joint-optimal beats greedy by {gap:+.4f} req/s ({rel:+.2f}%)")
            if g_in_rank["rank"] == 1:
                print("  >>> CONCLUSION: greedy extraction IS the joint optimum here.")
        else:
            print("\n  [warn] greedy partition not found in enumeration (unexpected).")
    else:
        gap = best_primary["total_throughput"] - gd["memo_total"]
        print(f"\n  >>> greedy uses K={gd['K']} != p={p}: comparing TOTALS only.")
        print(f"  >>> joint(p={p}) best total {best_primary['total_throughput']:.4f} "
              f"vs greedy(K={gd['K']}) total {gd['memo_total']:.4f}  -> {gap:+.4f} req/s")

    # ── 5. Save ──
    out_path = os.path.join(here, f"joint_p={p}_{model_short}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 84)
    print(f"Saved full results -> {out_path}")
    print("=" * 84)
    return out
