#!/usr/bin/env python3
"""Hardware micro-benchmark: measure effective BW, FLOPS, attention, and comm bandwidth.

Runs workloads on the current GPU(s) and reports:
  1. GEMV  (memory-bound)  -> effective HBM bandwidth (GB/s)
  2. GEMM  (compute-bound) -> effective TFLOPS
  3. FlashAttention prefill -> effective attention throughput (ms, TFLOPS)
  4. FlashAttention decode  -> effective decode throughput with KV cache (ms, TFLOPS)
  5. AllReduce (comm)       -> effective inter-GPU bandwidth (GB/s)

All benchmarks sweep batch sizes from 1 to --max-batch (powers of 2).

Usage:
  # Single-GPU (GEMV + GEMM + FlashAttention only)
  python bench_hw.py

  # Multi-GPU (includes AllReduce)
  torchrun --nproc_per_node=4 bench_hw.py

  # Custom model dimensions (Qwen3-32B example)
  python bench_hw.py --hidden-dim 5120 --intermediate-dim 25600 \\
      --attn-heads 64 --attn-kv-heads 8 --attn-head-dim 128 --max-batch 256

  # Save results
  python bench_hw.py --output results.json
"""

import argparse
import json
import os

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
    return times[len(times) // 2]


def get_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def batch_sizes_list(max_batch: int) -> list:
    """Generate batch sizes: 1, 2, 4, 8, ..., max_batch."""
    sizes = []
    bs = 1
    while bs <= max_batch:
        sizes.append(bs)
        bs *= 2
    # Include max_batch if not a power of 2
    if sizes[-1] != max_batch:
        sizes.append(max_batch)
    return sizes


# ── Benchmarks ───────────────────────────────────────────────────────

def bench_gemv(M: int, K: int, N: int, dtype, warmup: int, repeat: int) -> dict:
    """GEMV-like: [M, K] x [K, N] where M is small (memory-bound)."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    def fn():
        torch.matmul(A, B)

    elapsed = gpu_timer(fn, warmup, repeat)

    elem_bytes = A.element_size()
    bytes_moved = (M * K + K * N + M * N) * elem_bytes
    bw_GBs = bytes_moved / elapsed / 1e9
    flops = 2 * M * K * N
    tflops = flops / elapsed / 1e12

    return {
        "M": M, "K": K, "N": N,
        "elapsed_ms": round(elapsed * 1000, 4),
        "effective_bw_GBs": round(bw_GBs, 2),
        "effective_tflops": round(tflops, 4),
    }


def bench_gemm(M: int, K: int, N: int, dtype, warmup: int, repeat: int) -> dict:
    """Large GEMM: [M, K] x [K, N] (compute-bound)."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    def fn():
        torch.matmul(A, B)

    elapsed = gpu_timer(fn, warmup, repeat)

    flops = 2 * M * K * N
    tflops = flops / elapsed / 1e12

    return {
        "M": M, "K": K, "N": N,
        "elapsed_ms": round(elapsed * 1000, 4),
        "effective_tflops": round(tflops, 4),
    }


def bench_flash_attention_prefill(batch: int, seq_len: int, num_heads: int,
                                  head_dim: int, num_kv_heads: int,
                                  dtype, warmup: int, repeat: int) -> dict:
    """FlashAttention prefill benchmark."""
    try:
        from flash_attn import flash_attn_func
    except Exception as e:
        return {"error": f"flash_attn import failed: {e}"}

    Q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    K = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    V = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    def fn():
        flash_attn_func(Q, K, V, causal=True)

    elapsed = gpu_timer(fn, warmup, repeat)

    # Attention FLOPs: 2 * batch * heads * seq^2 * head_dim * 2 (QK + AV)
    attn_flops = 4 * batch * num_heads * seq_len * seq_len * head_dim
    tflops = attn_flops / elapsed / 1e12

    return {
        "batch": batch, "seq_len": seq_len,
        "elapsed_ms": round(elapsed * 1000, 4),
        "effective_tflops": round(tflops, 4),
    }


def bench_flash_attention_decode(batch: int, kv_len: int, num_heads: int,
                                 head_dim: int, num_kv_heads: int,
                                 dtype, warmup: int, repeat: int) -> dict:
    """FlashAttention decode benchmark (single query token with KV cache).

    Simulates autoregressive decode: Q has seq_len=1, K/V have seq_len=kv_len.
    """
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError:
        try:
            # Fallback: use flash_attn_func with seq_len=1 for Q
            from flash_attn import flash_attn_func

            Q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device="cuda")
            K = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
            V = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

            def fn():
                flash_attn_func(Q, K, V, causal=True)

            elapsed = gpu_timer(fn, warmup, repeat)

            # Memory traffic (memory-bound): read Q, K_cache, V_cache + write O
            # Q: batch * 1 * num_heads * head_dim
            # K_cache, V_cache: batch * kv_len * num_kv_heads * head_dim (each)
            # O: batch * 1 * num_heads * head_dim
            elem = Q.element_size()
            bytes_moved = (
                batch * 1 * num_heads * head_dim  # Q read
                + batch * kv_len * num_kv_heads * head_dim * 2  # K + V cache read
                + batch * 1 * num_heads * head_dim  # O write
            ) * elem
            bw_GBs = bytes_moved / elapsed / 1e9

            return {
                "batch": batch, "kv_len": kv_len,
                "elapsed_ms": round(elapsed * 1000, 4),
                "bytes_moved": bytes_moved,
                "effective_bw_GBs": round(bw_GBs, 2),
                "method": "flash_attn_func",
            }
        except Exception as e:
            return {"error": f"flash_attn import failed: {e}"}

    # Preferred: flash_attn_with_kvcache
    Q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device="cuda")
    K_cache = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    V_cache = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    def fn():
        flash_attn_with_kvcache(Q, K_cache, V_cache, causal=True)

    elapsed = gpu_timer(fn, warmup, repeat)

    elem = Q.element_size()
    bytes_moved = (
        batch * 1 * num_heads * head_dim
        + batch * kv_len * num_kv_heads * head_dim * 2
        + batch * 1 * num_heads * head_dim
    ) * elem
    bw_GBs = bytes_moved / elapsed / 1e9

    return {
        "batch": batch, "kv_len": kv_len,
        "elapsed_ms": round(elapsed * 1000, 4),
        "bytes_moved": bytes_moved,
        "effective_bw_GBs": round(bw_GBs, 2),
        "method": "flash_attn_with_kvcache",
    }


def bench_allreduce(size_bytes: int, dtype, warmup: int, repeat: int) -> dict:
    """AllReduce benchmark across all GPUs in the process group."""
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    num_elements = size_bytes // elem_bytes
    tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

    def fn():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    elapsed = gpu_timer(fn, warmup, repeat)

    world_size = dist.get_world_size()
    # Ring all-reduce: each GPU sends/receives 2 x (n-1)/n x data
    algo_bytes = 2 * (world_size - 1) / world_size * size_bytes
    bw_GBs = algo_bytes / elapsed / 1e9

    return {
        "data_bytes": size_bytes,
        "data_MB": round(size_bytes / 1e6, 2),
        "world_size": world_size,
        "elapsed_ms": round(elapsed * 1000, 4),
        "effective_bw_GBs": round(bw_GBs, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hardware micro-benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    # Model dimensions
    parser.add_argument("--hidden-dim", type=int, default=8192,
                        help="Hidden dimension (default: 8192 for Llama-70B)")
    parser.add_argument("--intermediate-dim", type=int, default=28672,
                        help="MLP intermediate dimension (default: 28672 for Llama-70B)")

    # Batch size sweep
    parser.add_argument("--max-batch", type=int, default=512,
                        help="Maximum batch size for sweep (powers of 2 from 1 to max)")

    # FlashAttention dimensions
    parser.add_argument("--attn-seq", type=int, default=763,
                        help="Prefill sequence length (default: 763)")
    parser.add_argument("--attn-heads", type=int, default=64,
                        help="Number of query heads")
    parser.add_argument("--attn-kv-heads", type=int, default=8,
                        help="Number of KV heads (GQA)")
    parser.add_argument("--attn-head-dim", type=int, default=128,
                        help="Head dimension")
    parser.add_argument("--kv-len", type=int, default=763,
                        help="KV cache length for decode benchmark (default: 763)")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Skip FlashAttention benchmarks")

    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    is_distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    if is_distributed:
        dist.init_process_group(backend="nccl")

    gpu_name = torch.cuda.get_device_name(local_rank)
    gpu_mem_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1e9

    H = args.hidden_dim
    I = args.intermediate_dim
    bs_list = batch_sizes_list(args.max_batch)

    results = {
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_mem_gb, 2),
        "dtype": args.dtype,
        "hidden_dim": H,
        "intermediate_dim": I,
        "max_batch": args.max_batch,
        "batch_sizes": bs_list,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "benchmarks": {},
    }

    if rank == 0:
        print(f"{'='*70}")
        print(f"Hardware Micro-Benchmark")
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
        print(f"dtype: {args.dtype}, hidden_dim: {H}, intermediate_dim: {I}")
        print(f"Batch sizes: {bs_list}")
        print(f"{'='*70}")

    # ── Single-GPU benchmarks (rank 0 only) ────────────────────────────
    # Other ranks wait at barrier before AllReduce.

    if rank == 0:
        # ── 1. GEMV (memory-bound, decode-like) ─────────────────────
        print(f"\n[1/6] GEMV (memory-bound) — sweep batch sizes")
        print(f"      [{H}] x [{H}x{I}]  (hidden → intermediate)")
        print(f"      {'BS':>6}  {'Time (ms)':>10}  {'BW (GB/s)':>10}  {'TFLOPS':>8}")
        gemv_results = []
        for bs in bs_list:
            r = bench_gemv(bs, H, I, dtype, args.warmup, args.repeat)
            gemv_results.append(r)
            print(f"      {bs:>6}  {r['elapsed_ms']:>10.4f}  {r['effective_bw_GBs']:>10.2f}  {r['effective_tflops']:>8.4f}")
        results["benchmarks"]["gemv"] = gemv_results

        # ── 2. GEMM (compute-bound, prefill-like) ───────────────────
        print(f"\n[2/6] GEMM (compute-bound) — sweep batch sizes")
        print(f"      [BS*{args.attn_seq}, {H}] x [{H}, {I}]  (prefill-like)")
        print(f"      {'BS':>6}  {'M':>8}  {'Time (ms)':>10}  {'TFLOPS':>8}")
        gemm_results = []
        for bs in bs_list:
            M = bs * args.attn_seq
            r = bench_gemm(M, H, I, dtype, args.warmup, args.repeat)
            r["batch"] = bs
            gemm_results.append(r)
            print(f"      {bs:>6}  {M:>8}  {r['elapsed_ms']:>10.4f}  {r['effective_tflops']:>8.4f}")
        results["benchmarks"]["gemm"] = gemm_results

        # ── 3. FlashAttention prefill ────────────────────────────────
        if not args.no_flash_attn:
            print(f"\n[3/6] FlashAttention (prefill) — sweep batch sizes")
            print(f"      S={args.attn_seq} H={args.attn_heads} Hkv={args.attn_kv_heads} D={args.attn_head_dim}")
            print(f"      {'BS':>6}  {'Time (ms)':>10}  {'TFLOPS':>8}")
            fa_prefill_results = []
            for bs in bs_list:
                r = bench_flash_attention_prefill(
                    bs, args.attn_seq, args.attn_heads, args.attn_head_dim,
                    args.attn_kv_heads, dtype, args.warmup, args.repeat,
                )
                fa_prefill_results.append(r)
                if "error" in r:
                    print(f"      {bs:>6}  ERROR: {r['error']}")
                    break
                else:
                    print(f"      {bs:>6}  {r['elapsed_ms']:>10.4f}  {r['effective_tflops']:>8.4f}")
            results["benchmarks"]["flash_attn_prefill"] = fa_prefill_results

            # ── 4. FlashAttention decode ─────────────────────────────
            print(f"\n[4/6] FlashAttention (decode) — sweep batch sizes")
            print(f"      Q_len=1 KV_len={args.kv_len} H={args.attn_heads} Hkv={args.attn_kv_heads} D={args.attn_head_dim}")
            print(f"      {'BS':>6}  {'Time (ms)':>10}  {'BW (GB/s)':>10}")
            fa_decode_results = []
            for bs in bs_list:
                r = bench_flash_attention_decode(
                    bs, args.kv_len, args.attn_heads, args.attn_head_dim,
                    args.attn_kv_heads, dtype, args.warmup, args.repeat,
                )
                fa_decode_results.append(r)
                if "error" in r:
                    print(f"      {bs:>6}  ERROR: {r['error']}")
                    break
                else:
                    print(f"      {bs:>6}  {r['elapsed_ms']:>10.4f}  {r['effective_bw_GBs']:>10.2f}")
            results["benchmarks"]["flash_attn_decode"] = fa_decode_results
        else:
            print(f"\n[3/6] FlashAttention prefill — skipped")
            print(f"[4/6] FlashAttention decode — skipped")

    # Synchronize all ranks before AllReduce
    if is_distributed:
        dist.barrier()

    # ── 5. AllReduce decode (communication, all ranks participate) ────
    if is_distributed and dist.get_world_size() > 1:
        if rank == 0:
            print(f"\n[5/6] AllReduce decode ({dist.get_world_size()} GPUs) — sweep batch sizes")
            print(f"      data_size = BS × {H} × {elem_bytes} bytes")
            print(f"      {'BS':>6}  {'Size (MB)':>10}  {'Time (ms)':>10}  {'BW (GB/s)':>10}")
        ar_decode_results = []
        for bs in bs_list:
            ar_bytes = bs * H * elem_bytes
            r = bench_allreduce(ar_bytes, dtype, args.warmup, args.repeat)
            r["batch"] = bs
            ar_decode_results.append(r)
            if rank == 0:
                print(f"      {bs:>6}  {ar_bytes / 1e6:>10.2f}  {r['elapsed_ms']:>10.4f}  {r['effective_bw_GBs']:>10.2f}")
        results["benchmarks"]["allreduce_decode"] = ar_decode_results

        # ── 6. AllReduce prefill (communication, all ranks participate) ──
        seq = args.attn_seq
        if rank == 0:
            print(f"\n[6/6] AllReduce prefill ({dist.get_world_size()} GPUs) — sweep batch sizes")
            print(f"      data_size = BS × {seq} × {H} × {elem_bytes} bytes")
            print(f"      {'BS':>6}  {'Size (MB)':>10}  {'Time (ms)':>10}  {'BW (GB/s)':>10}")
        ar_prefill_results = []
        for bs in bs_list:
            ar_bytes = bs * seq * H * elem_bytes
            r = bench_allreduce(ar_bytes, dtype, args.warmup, args.repeat)
            r["batch"] = bs
            ar_prefill_results.append(r)
            if rank == 0:
                print(f"      {bs:>6}  {ar_bytes / 1e6:>10.2f}  {r['elapsed_ms']:>10.4f}  {r['effective_bw_GBs']:>10.2f}")
        results["benchmarks"]["allreduce_prefill"] = ar_prefill_results
    else:
        if rank == 0:
            print(f"\n[5/6] AllReduce decode — skipped (single GPU or not distributed)")
            print(f"[6/6] AllReduce prefill — skipped (single GPU or not distributed)")

    # ── Summary ──────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"SUMMARY — {gpu_name}")
        print(f"{'='*70}")

        # Peak values from sweeps
        gemv_data = results["benchmarks"].get("gemv", [])
        if gemv_data:
            peak_bw = max(r["effective_bw_GBs"] for r in gemv_data)
            print(f"  Peak HBM BW (GEMV):     {peak_bw:.2f} GB/s")
        gemm_data = results["benchmarks"].get("gemm", [])
        if gemm_data:
            peak_tflops = max(r["effective_tflops"] for r in gemm_data)
            print(f"  Peak TFLOPS (GEMM):     {peak_tflops:.4f} TFLOPS")
        fa_prefill = results["benchmarks"].get("flash_attn_prefill", [])
        if fa_prefill and "error" not in fa_prefill[0]:
            peak_fa = max(r["effective_tflops"] for r in fa_prefill if "error" not in r)
            print(f"  Peak FlashAttn prefill: {peak_fa:.4f} TFLOPS")
        fa_decode = results["benchmarks"].get("flash_attn_decode", [])
        if fa_decode and "error" not in fa_decode[0]:
            peak_fd_bw = max(r["effective_bw_GBs"] for r in fa_decode if "error" not in r)
            print(f"  Peak FlashAttn decode:  {peak_fd_bw:.2f} GB/s")
        ar_dec = results["benchmarks"].get("allreduce_decode", [])
        if ar_dec:
            peak_ar_dec = max(r["effective_bw_GBs"] for r in ar_dec)
            print(f"  Peak AR BW (decode):    {peak_ar_dec:.2f} GB/s")
        ar_pre = results["benchmarks"].get("allreduce_prefill", [])
        if ar_pre:
            peak_ar_pre = max(r["effective_bw_GBs"] for r in ar_pre)
            print(f"  Peak AR BW (prefill):   {peak_ar_pre:.2f} GB/s")

        print(f"{'='*70}")

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
