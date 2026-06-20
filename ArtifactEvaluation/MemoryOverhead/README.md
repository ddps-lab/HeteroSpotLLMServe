# Memory-Overhead Measurement (Rebuttal R2#4)

Measures the GPU memory overhead of running a **second vLLM engine that shares one
tensor store** with the first — the exact situation during ShuntServe's *concurrent
initialization* overlap window. Because the shared tensor store holds the model
weights and KV cache **once**, the marginal footprint of the second engine is only
its **duplicate CUDA context + cuBLAS/cuDNN workspaces + NCCL communicators**. This is
what reviewer R2#4 asks about.

## What it does

On **2× g6.12xlarge (8× L4)** in a **TP=4 (intra-node) + PP=2 (inter-node)** topology
— mirroring SS-P1's `parallel_strategy=[4,4,...]` g6.12xlarge stages — the orchestrator
brings the system up step by step and snapshots GPU memory after each step:

```
resident observers (memory meters, contexts cancel out in the diff)
  → 1 shared tensor store (8 procs, weights + KV loaded once)   ⇒ S0
  → engine 1  (Ray cluster :6379, API :8001)                    ⇒ S1
  → engine 2  (Ray cluster :6380, API :8002, SAME GPUs/store)   ⇒ S2   (no 2nd store)
```

**Headline metric:** `ΔE2 = used(S2) − used(S1)` per GPU = the second engine's overhead.
Also reports `used(S1) − used(S0)` (first-engine cost) and `ΔE2 / engine1` (%).

Memory is read with **`torch.cuda.mem_get_info`** — the same driver-level call the
tensor store uses to size the KV cache ([../../TensorStore/raw_s3_tensor_store_server.py:505](../../TensorStore/raw_s3_tensor_store_server.py#L505)).
Unlike `torch.cuda.memory_reserved/allocated`, it captures the CUDA context, cuBLAS/cuDNN
workspaces and NCCL buffers, which live **outside** the torch caching allocator.

## Files

- `measure_engine_overhead.py` — orchestrator (run on the head node N0). Reuses
  ShuntServe's `GlobalServer/command.py` builders and `protocols.py`; modifies nothing.
- `mem_observer.py` — one per node, pinned to all local GPUs; serves `mem_get_info`
  over a socket. Launched (and torn down) automatically by the orchestrator.

## Prerequisites

- Two g6.12xlarge nodes with ShuntServe at `/home/ubuntu/ShuntServe` (paths come from
  `GlobalServer/command.py`), passwordless SSH from the driver (N0) to both node IPs
  (and to N0's own IP), and the model's S3 weights present.
- Run the script **on N0** (it also acts as the Ray head).

## Run

```bash
cd /home/ubuntu/ShuntServe/ArtifactEvaluation/MemoryOverhead

# 8B
python measure_engine_overhead.py --nodes <N0_IP> <N1_IP> --model 8b  --out overhead_8b.json
# 70B  (also confirms the overhead is model-independent)
python measure_engine_overhead.py --nodes <N0_IP> <N1_IP> --model 70b --out overhead_70b.json
```

The script starts/stops the two Ray heads (`:6379`, `:6380`) with the exact commands
from the top-level `README.md` (`ray start --head --port=...`, no temp-dir — both heads
coexist on the default temp dir because clusters are always addressed by explicit
host:port). If you already started both heads + ran `SpotTolerance/init_ray_workers.sh`,
pass `--skip-ray-setup`.

Useful flags: `--ssh-user`, `--head-ip`, `--keep-up` (leave everything running to inspect),
`--out`. Per-run logs land in `logs/<model>_<ts>/`.

## Reading the result

- The reviewer's three components are reported as **one aggregate** `ΔE2` (their sum).
  Expect a small per-GPU value (CUDA context + cuBLAS/cuDNN + NCCL), a few % of the
  24 GB L4 — well within vLLM's `gpu_memory_utilization` margin, so the overlap does
  **not** reduce the serving engine's KV-cache batch capacity.
- **Alignment is correct by construction:** both engines use the *same*
  `node_rank_mapping`, and vLLM places rank *i* deterministically on `cuda:i`, which
  connects to tensor-store port `50001+i` — so engine 2 attaches to engine 1's GPUs
  and store. A misplacement (engine 2 loading its own weights) would show up
  immediately as `ΔE2` jumping to ~a full engine (weights + KV included); a small
  `ΔE2` is itself the confirmation.
- Compare 8B vs 70B `ΔE2`: they should differ by < ~10% (per-engine overhead is
  governed by per-process context + topology-dependent NCCL buffers, not model size).

## Paper sentence (fill in measured X, Y)

> A concurrently initialized second engine that shares the tensor store adds only
> **X.X GB/GPU** (duplicate CUDA context + cuBLAS/cuDNN + NCCL communicators), **< Y %**
> of L4 capacity and within vLLM's `gpu_memory_utilization` margin, so it does not
> reduce the serving engine's KV-cache batch capacity during the overlap window.
