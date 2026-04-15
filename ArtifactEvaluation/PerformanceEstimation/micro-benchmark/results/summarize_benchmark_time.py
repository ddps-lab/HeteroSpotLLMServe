#!/usr/bin/env python3
"""Summarize wall-clock time required by the hardware micro-benchmarks.

Reads every ``<instance>.json`` in this directory (produced by
``bench_hw.py --output ...``) and reports, per instance:

  * which GPU / config was measured
  * which operations were benchmarked and with what shape
  * the sum of ``elapsed_ms`` per operation
  * an estimate of the total wall-clock time spent running the benchmark

The estimate is a kernel-time lower bound:

    per_(op, batch) time   ~= elapsed_ms * (warmup + repeat)    [ms]
    per_instance_total     ~= sum over all (op, batch) of the above

because ``gpu_timer`` in ``bench_hw.py`` invokes ``fn()`` exactly
``warmup + repeat`` times per (op, batch). CUDA sync, tensor allocation,
NCCL init, python overhead, etc. are NOT counted, so the true wall-clock
was somewhat larger than what this script reports.

Usage:
    python summarize_benchmark_time.py
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def instance_name(path: Path) -> str:
    # g5_48xlarge.json -> g5.48xlarge
    return path.stem.replace("_", ".")


def op_config_summary(op: str, entries: list, results: dict) -> str:
    """One-line human description of the op's shape, derived from the data."""
    if not entries:
        return "(no entries)"
    first = entries[0]
    H = results["hidden_dim"]
    I = results["intermediate_dim"]

    if op == "gemv":
        return f"M=bs, K={H}, N={I}  (decode-like, memory-bound)"
    if op == "gemm":
        # M = bs * seq_len; infer seq_len from M/batch of first entry
        seq = first["M"] // first["batch"] if first.get("batch") else "?"
        return f"M=bs*{seq}, K={H}, N={I}  (prefill-like, compute-bound)"
    if op == "flash_attn_prefill":
        return f"seq_len={first['seq_len']}  (bs Q/K/V tokens, causal)"
    if op == "flash_attn_decode":
        method = first.get("method", "?")
        return f"kv_len={first['kv_len']}  (Q_len=1, method={method})"
    if op == "allreduce_decode":
        return f"world_size={first['world_size']}, data=bs*{H}*elem_bytes"
    if op == "allreduce_prefill":
        # derive seq from data_bytes = bs * seq * H * elem
        db = first["data_bytes"]
        # first entry is bs=1, so data_bytes = seq * H * elem  =>  seq = db / (H * elem)
        # elem = 2 for bfloat16/float16
        elem = 2 if results["dtype"] in ("bfloat16", "float16") else 4
        seq = db // (H * elem) if first.get("batch") == 1 else "?"
        return f"world_size={first['world_size']}, data=bs*{seq}*{H}*elem_bytes"
    return "(unknown op)"


def summarize_instance(path: Path) -> dict:
    results = json.loads(path.read_text())
    warmup = results["warmup"]
    repeat = results["repeat"]
    factor = warmup + repeat

    per_op = []
    grand_ms = 0.0
    for op, entries in results["benchmarks"].items():
        valid = [r for r in entries if "error" not in r]
        sum_ms = sum(r["elapsed_ms"] for r in valid)
        est_s = sum_ms * factor / 1000.0
        grand_ms += sum_ms
        batches = [r.get("batch", r.get("M")) for r in valid]
        per_op.append({
            "op": op,
            "config": op_config_summary(op, valid, results),
            "n_entries": len(valid),
            "batches": batches,
            "sum_elapsed_ms": sum_ms,
            "est_time_s": est_s,
        })

    return {
        "instance": instance_name(path),
        "gpu_name": results["gpu_name"],
        "gpu_memory_gb": results["gpu_memory_gb"],
        "dtype": results["dtype"],
        "hidden_dim": results["hidden_dim"],
        "intermediate_dim": results["intermediate_dim"],
        "batch_sizes": results["batch_sizes"],
        "max_batch": results["max_batch"],
        "warmup": warmup,
        "repeat": repeat,
        "factor": factor,
        "per_op": per_op,
        "instance_total_ms": grand_ms,
        "instance_total_s": grand_ms * factor / 1000.0,
    }


def main() -> None:
    files = sorted(p for p in HERE.glob("*.json"))
    if not files:
        print(f"No *.json files found in {HERE}")
        return

    summaries = [summarize_instance(p) for p in files]

    bar = "=" * 78
    sub = "-" * 78
    print(bar)
    print("Micro-Benchmark Wall-Clock Time Summary (kernel-time lower bound)")
    print(bar)
    print(f"Source directory : {HERE}")
    print(f"Instances measured : {len(summaries)}")
    for s in summaries:
        print(f"  - {s['instance']:<14s} "
              f"({s['gpu_name']}, {s['dtype']}, max_batch={s['max_batch']})")
    print()

    for s in summaries:
        print(sub)
        print(f"[{s['instance']}]")
        print(f"  GPU           : {s['gpu_name']} ({s['gpu_memory_gb']:.2f} GB)")
        print(f"  dtype         : {s['dtype']}")
        print(f"  hidden_dim    : {s['hidden_dim']}    intermediate_dim: {s['intermediate_dim']}")
        print(f"  batch_sizes   : {s['batch_sizes']}")
        print(f"  warmup/repeat : {s['warmup']} / {s['repeat']}  "
              f"(each fn() called {s['factor']} times per batch)")
        print()
        print(f"  {'op':<20s}  {'config':<58s}  {'n':>3s}  "
              f"{'Σelapsed(ms)':>13s}  {'est_time(s)':>11s}")
        for op in s["per_op"]:
            print(f"  {op['op']:<20s}  {op['config']:<58s}  {op['n_entries']:>3d}  "
                  f"{op['sum_elapsed_ms']:>13.3f}  {op['est_time_s']:>11.3f}")
        print(f"  {sub[2:]}")
        total_s = s["instance_total_s"]
        print(f"  instance total estimate : {total_s:>8.2f} s   "
              f"(~{total_s/60:.2f} min)")
        print()

    # Cross-instance comparison
    print(bar)
    print("Overall (per-instance end-to-end estimate)")
    print(bar)
    print(f"  {'instance':<14s}  {'GPU':<22s}  {'max_bs':>7s}  "
          f"{'time(s)':>9s}  {'time(min)':>10s}")
    grand_s = 0.0
    for s in summaries:
        grand_s += s["instance_total_s"]
        print(f"  {s['instance']:<14s}  {s['gpu_name']:<22s}  {s['max_batch']:>7d}  "
              f"{s['instance_total_s']:>9.2f}  {s['instance_total_s']/60:>10.2f}")
    print(f"  {sub[2:]}")
    print(f"  {'total':<14s}  {'':<22s}  {'':>7s}  "
          f"{grand_s:>9.2f}  {grand_s/60:>10.2f}")
    print()
    print("Note: kernel-time lower bound; excludes tensor allocation, CUDA/NCCL")
    print("      initialization, python overhead, and torchrun spin-up.")


if __name__ == "__main__":
    main()
