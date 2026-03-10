#!/usr/bin/env python3
"""Hardware micro-benchmark: measure effective BW, FLOPS, and comm bandwidth.

Runs three saturated workloads on the current GPU(s) and reports:
  1. GEMV  (memory-bound) → effective HBM bandwidth (GB/s)
  2. GEMM  (compute-bound) → effective TFLOPS
  3. AllReduce (comm)      → effective inter-GPU bandwidth (GB/s)

Usage:
  # Single-GPU (GEMV + GEMM only)
  python bench_hw.py

  # Multi-GPU (includes AllReduce)
  torchrun --nproc_per_node=8 bench_hw.py

  # Custom parameters
  python bench_hw.py --warmup 20 --repeat 100 --dtype bfloat16

  # Save results
  python bench_hw.py --output results.json
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


# ── Helpers ──────────────────────────────────────────────────────────

def gpu_timer(fn, warmup: int = 10, repeat: int = 50) -> float:
    """Time a GPU function using cuda events. Returns median time in seconds."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # ms → sec

    times.sort()
    # Use median
    return times[len(times) // 2]


def get_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


# ── Benchmarks ───────────────────────────────────────────────────────

def bench_gemv(M: int, K: int, N: int, dtype, warmup: int, repeat: int) -> dict:
    """GEMV-like: [M, K] × [K, N] where M is small (memory-bound).

    Measures effective HBM bandwidth.
    M=1 is pure GEMV; M=2-4 also memory-bound on most GPUs.
    """
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    def fn():
        torch.matmul(A, B)

    elapsed = gpu_timer(fn, warmup, repeat)

    # Memory traffic: read A (M×K) + read B (K×N) + write C (M×N)
    elem_bytes = A.element_size()
    bytes_moved = (M * K + K * N + M * N) * elem_bytes
    bw_GBs = bytes_moved / elapsed / 1e9

    # FLOPs (for reference)
    flops = 2 * M * K * N
    tflops = flops / elapsed / 1e12

    return {
        "workload": f"GEMV [{M}×{K}] × [{K}×{N}]",
        "dtype": str(dtype).replace("torch.", ""),
        "elapsed_ms": round(elapsed * 1000, 4),
        "bytes_moved": bytes_moved,
        "effective_bw_GBs": round(bw_GBs, 2),
        "flops": flops,
        "effective_tflops": round(tflops, 4),
    }


def bench_gemm(M: int, K: int, N: int, dtype, warmup: int, repeat: int) -> dict:
    """Large GEMM: [M, K] × [K, N] where M is large (compute-bound).

    Measures effective TFLOPS.
    """
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    def fn():
        torch.matmul(A, B)

    elapsed = gpu_timer(fn, warmup, repeat)

    flops = 2 * M * K * N
    tflops = flops / elapsed / 1e12

    # BW for reference
    elem_bytes = A.element_size()
    bytes_moved = (M * K + K * N + M * N) * elem_bytes
    bw_GBs = bytes_moved / elapsed / 1e9

    return {
        "workload": f"GEMM [{M}×{K}] × [{K}×{N}]",
        "dtype": str(dtype).replace("torch.", ""),
        "elapsed_ms": round(elapsed * 1000, 4),
        "bytes_moved": bytes_moved,
        "effective_bw_GBs": round(bw_GBs, 2),
        "flops": flops,
        "effective_tflops": round(tflops, 4),
    }


def bench_allreduce(size_bytes: int, dtype, warmup: int, repeat: int) -> dict:
    """AllReduce benchmark across all GPUs in the process group.

    Measures effective inter-GPU bandwidth.
    """
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    num_elements = size_bytes // elem_bytes
    tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

    def fn():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    elapsed = gpu_timer(fn, warmup, repeat)

    world_size = dist.get_world_size()
    # Ring all-reduce: each GPU sends/receives 2 × (n-1)/n × data
    algo_bytes = 2 * (world_size - 1) / world_size * size_bytes
    bw_GBs = algo_bytes / elapsed / 1e9

    return {
        "workload": f"AllReduce {size_bytes / 1e6:.1f}MB × {world_size} GPUs",
        "dtype": str(dtype).replace("torch.", ""),
        "elapsed_ms": round(elapsed * 1000, 4),
        "data_bytes": size_bytes,
        "algo_bytes": round(algo_bytes),
        "world_size": world_size,
        "effective_bw_GBs": round(bw_GBs, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hardware micro-benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")

    # GEMV dimensions (memory-bound, decode-like)
    parser.add_argument("--gemv-m", type=int, default=1, help="GEMV M dimension (batch, keep small)")
    parser.add_argument("--gemv-k", type=int, default=8192, help="GEMV K dimension (hidden_dim)")
    parser.add_argument("--gemv-n", type=int, default=28672, help="GEMV N dimension (intermediate_dim)")

    # GEMM dimensions (compute-bound, prefill-like)
    parser.add_argument("--gemm-m", type=int, default=2048, help="GEMM M dimension (large batch)")
    parser.add_argument("--gemm-k", type=int, default=8192, help="GEMM K dimension (hidden_dim)")
    parser.add_argument("--gemm-n", type=int, default=28672, help="GEMM N dimension (intermediate_dim)")

    # AllReduce size
    parser.add_argument("--ar-size-mb", type=float, default=64, help="AllReduce data size in MB")

    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    is_distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    if is_distributed:
        dist.init_process_group(backend="nccl")

    gpu_name = torch.cuda.get_device_name(local_rank)
    gpu_mem_gb = torch.cuda.get_device_properties(local_rank).total_mem / 1e9

    results = {
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_mem_gb, 2),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "benchmarks": {},
    }

    if rank == 0:
        print(f"{'='*60}")
        print(f"Hardware Micro-Benchmark")
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
        print(f"dtype: {args.dtype}, warmup: {args.warmup}, repeat: {args.repeat}")
        print(f"{'='*60}")

    # 1. GEMV (memory-bound)
    if rank == 0:
        print(f"\n[1/3] GEMV (memory-bound) [{args.gemv_m}×{args.gemv_k}] × [{args.gemv_k}×{args.gemv_n}]")
    gemv = bench_gemv(args.gemv_m, args.gemv_k, args.gemv_n, dtype, args.warmup, args.repeat)
    results["benchmarks"]["gemv"] = gemv
    if rank == 0:
        print(f"  Time: {gemv['elapsed_ms']:.4f} ms")
        print(f"  Effective BW:    {gemv['effective_bw_GBs']:.2f} GB/s")
        print(f"  Effective FLOPS: {gemv['effective_tflops']:.4f} TFLOPS")

    # Run additional GEMV sizes for stability check
    gemv_sizes = [
        (1, args.gemv_k, args.gemv_n),
        (1, args.gemv_k, args.gemv_k),  # square-ish weight
        (4, args.gemv_k, args.gemv_n),  # small batch
    ]
    gemv_bws = []
    for m, k, n in gemv_sizes:
        r = bench_gemv(m, k, n, dtype, args.warmup, args.repeat)
        gemv_bws.append(r["effective_bw_GBs"])
        if rank == 0:
            print(f"  [{m}×{k}]×[{k}×{n}]: {r['effective_bw_GBs']:.2f} GB/s")
    results["benchmarks"]["gemv_sweep"] = {
        "sizes": [f"{m}×{k}×{n}" for m, k, n in gemv_sizes],
        "bw_GBs": gemv_bws,
        "median_bw_GBs": round(sorted(gemv_bws)[len(gemv_bws) // 2], 2),
    }
    if rank == 0:
        print(f"  Median BW across sizes: {results['benchmarks']['gemv_sweep']['median_bw_GBs']:.2f} GB/s")

    # 2. GEMM (compute-bound)
    if rank == 0:
        print(f"\n[2/3] GEMM (compute-bound) [{args.gemm_m}×{args.gemm_k}] × [{args.gemm_k}×{args.gemm_n}]")
    gemm = bench_gemm(args.gemm_m, args.gemm_k, args.gemm_n, dtype, args.warmup, args.repeat)
    results["benchmarks"]["gemm"] = gemm
    if rank == 0:
        print(f"  Time: {gemm['elapsed_ms']:.4f} ms")
        print(f"  Effective FLOPS: {gemm['effective_tflops']:.4f} TFLOPS")
        print(f"  Effective BW:    {gemm['effective_bw_GBs']:.2f} GB/s")

    # Additional GEMM sizes
    gemm_sizes = [
        (2048, args.gemm_k, args.gemm_n),
        (4096, args.gemm_k, args.gemm_n),
        (2048, args.gemm_k, args.gemm_k),
    ]
    gemm_tflops = []
    for m, k, n in gemm_sizes:
        r = bench_gemm(m, k, n, dtype, args.warmup, args.repeat)
        gemm_tflops.append(r["effective_tflops"])
        if rank == 0:
            print(f"  [{m}×{k}]×[{k}×{n}]: {r['effective_tflops']:.4f} TFLOPS")
    results["benchmarks"]["gemm_sweep"] = {
        "sizes": [f"{m}×{k}×{n}" for m, k, n in gemm_sizes],
        "tflops": gemm_tflops,
        "median_tflops": round(sorted(gemm_tflops)[len(gemm_tflops) // 2], 4),
    }
    if rank == 0:
        print(f"  Median FLOPS across sizes: {results['benchmarks']['gemm_sweep']['median_tflops']:.4f} TFLOPS")

    # 3. AllReduce (communication)
    if is_distributed and dist.get_world_size() > 1:
        ar_bytes = int(args.ar_size_mb * 1e6)
        if rank == 0:
            print(f"\n[3/3] AllReduce ({args.ar_size_mb:.0f} MB, {dist.get_world_size()} GPUs)")
        ar = bench_allreduce(ar_bytes, dtype, args.warmup, args.repeat)
        results["benchmarks"]["allreduce"] = ar
        if rank == 0:
            print(f"  Time: {ar['elapsed_ms']:.4f} ms")
            print(f"  Effective BW: {ar['effective_bw_GBs']:.2f} GB/s")

        # Sweep sizes
        ar_sizes_mb = [1, 8, 32, 64, 128]
        ar_bws = []
        for sz in ar_sizes_mb:
            r = bench_allreduce(int(sz * 1e6), dtype, args.warmup, args.repeat)
            ar_bws.append(r["effective_bw_GBs"])
            if rank == 0:
                print(f"  {sz}MB: {r['effective_bw_GBs']:.2f} GB/s")
        results["benchmarks"]["allreduce_sweep"] = {
            "sizes_mb": ar_sizes_mb,
            "bw_GBs": ar_bws,
            "max_bw_GBs": round(max(ar_bws), 2),
        }
        if rank == 0:
            print(f"  Max BW (saturated): {max(ar_bws):.2f} GB/s")
    else:
        if rank == 0:
            print(f"\n[3/3] AllReduce — skipped (single GPU or not distributed)")

    # Summary
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"SUMMARY — {gpu_name}")
        print(f"{'='*60}")
        print(f"  Effective HBM BW:   {results['benchmarks']['gemv_sweep']['median_bw_GBs']:.2f} GB/s")
        print(f"  Effective TFLOPS:   {results['benchmarks']['gemm_sweep']['median_tflops']:.4f} TFLOPS")
        if "allreduce_sweep" in results["benchmarks"]:
            print(f"  Effective Comm BW:  {results['benchmarks']['allreduce_sweep']['max_bw_GBs']:.2f} GB/s")
        print(f"{'='*60}")

        results["summary"] = {
            "effective_bw_GBs": results["benchmarks"]["gemv_sweep"]["median_bw_GBs"],
            "effective_tflops": results["benchmarks"]["gemm_sweep"]["median_tflops"],
        }
        if "allreduce_sweep" in results["benchmarks"]:
            results["summary"]["effective_comm_bw_GBs"] = results["benchmarks"]["allreduce_sweep"]["max_bw_GBs"]

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
