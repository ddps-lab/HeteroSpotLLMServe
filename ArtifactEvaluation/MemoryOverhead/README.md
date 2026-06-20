# Memory-Overhead Measurement (Rebuttal R2#4)

Measures the GPU memory overhead of running a **second vLLM engine that shares one
tensor store** with the first — the situation during ShuntServe's *concurrent
initialization* overlap window. Because the shared tensor store holds the model weights
and KV cache **once**, the marginal footprint of the second engine is only its
**duplicate CUDA context + cuBLAS/cuDNN workspaces + NCCL communicators**. This is what
reviewer R2#4 asks about.

## This is the real production scenario (not a constructed worst case)

In `switch_node` ([../../GlobalServer/VNode.py:728](../../GlobalServer/VNode.py#L728)) only the
interrupted node is replaced (`self.vnodes[target_index] = new_vnode`), but the **new
Ray cluster (6380) is joined by *all* current vnodes — including the unchanged ones**
([VNode.py:779](../../GlobalServer/VNode.py#L779)), a fresh tensor store is started **only on
the new node** ([VNode.py:802](../../GlobalServer/VNode.py#L802)), and the new API server is
brought up and waited on **before** the old one is stopped
([VNode.py:830-847](../../GlobalServer/VNode.py#L830)). So during the overlap, every **unchanged
stage node** runs the old pipeline's workers *and* the new pipeline's workers on the
**same GPUs, sharing the same tensor store**. That is exactly `1 store + 2 engines` —
what this harness measures. (Only the single swapped node never doubles up.)

## What it does

On **2× g6.12xlarge (8× L4)**, **TP=4 (intra-node) + PP=2 (inter-node)** — mirroring
SS-P1's `parallel_strategy=[4,4,...]` g6.12xlarge stages — the orchestrator brings the
system up and snapshots GPU memory at five points:

```
resident observers (memory meters; their context cancels out in the diffs)
  → 1 shared tensor store (8 procs, weights + KV loaded once)        ⇒ S0
  → engine 1 (Ray :6379, API :8001)  @health                        ⇒ S1h   ctx + NCCL
                                      warmed (1 request)             ⇒ S1w   + cuBLAS/cuDNN
  → engine 2 (Ray :6380, API :8002, SAME GPUs/store, no 2nd store)
                                      @health                        ⇒ S2h
                                      warmed (1 request)             ⇒ S2w
```

**Why health *and* warmup?** vLLM allocates the NCCL communicators **eagerly** at
distributed init — `PyNcclCommunicator.__init__` calls `ncclCommInitRank` + an eager
warmup all-reduce ([../../submodules/vLLM/vllm/distributed/device_communicators/pynccl.py:99-105](../../submodules/vLLM/vllm/distributed/device_communicators/pynccl.py#L99))
— so the **CUDA context + NCCL buffers are already present at `/health`**. Only the
**cuBLAS/cuDNN GEMM workspace is lazy** (needs a real forward), which the warmup request
realizes. This splits the second engine's overhead into its two natural parts.

**Reported metrics (per GPU, mean over 8):**

| metric | formula | meaning |
|---|---|---|
| `ctx+NCCL` | `S2h − S1w` | second engine's **overlap-window** cost (it only reaches `/health` before switchover, so no cuBLAS yet) |
| `+cuBLAS/cuDNN` | `S2w − S2h` | extra workspace once it actually serves |
| `full` | `S2w − S1w` | full per-engine marginal |

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
python measure_engine_overhead.py --nodes <N0_IP> <N1_IP> --model 8b
# 70B  (also confirms the overhead is model-independent)
python measure_engine_overhead.py --nodes <N0_IP> <N1_IP> --model 70b
```

Results are written to `results/overhead_<model>.json`; per-run logs to `logs/<model>/`.

The script starts/stops the two Ray heads (`:6379`, `:6380`) with the exact commands
from the top-level `README.md` (`ray start --head --port=...`, no temp-dir — both heads
coexist on the default temp dir because clusters are always addressed by explicit
host:port). If you already started both heads + ran `SpotTolerance/init_ray_workers.sh`,
pass `--skip-ray-setup`.

Useful flags: `--ssh-user`, `--head-ip`, `--keep-up` (leave everything running to inspect),
`--out`. Per-run logs land in `logs/<model>_<ts>/`.

## Reading the result

- The **overlap-window** number is `ctx+NCCL` (`S2h − S1w`): what the second engine adds
  while it initializes, before it takes over. `full` (`S2w − S1w`) bounds it including
  the cuBLAS/cuDNN workspace it will use once serving. Both should be a small per-GPU
  value — a few % of the 24 GB L4, within vLLM's `gpu_memory_utilization` margin — so the
  overlap does **not** reduce the serving engine's KV-cache batch capacity.
- **Alignment is correct by construction:** both engines use the *same*
  `node_rank_mapping`, and vLLM places rank *i* deterministically on `cuda:i`, which
  connects to tensor-store port `50001+i` — so engine 2 attaches to engine 1's GPUs and
  store. A misplacement (engine 2 loading its own weights) would show up immediately as
  `full` jumping to ~a whole engine (weights + KV included); a small `full` is itself the
  confirmation.
- Compare 8B vs 70B: the per-engine overhead should differ by < ~10% (governed by
  per-process context + topology-dependent NCCL buffers, not model size).

## Paper sentence (fill in measured X, Y)

> During concurrent initialization a second engine shares the tensor store on the
> unchanged stage GPUs and adds only **X.X GB/GPU** (duplicate CUDA context + NCCL
> communicators, with cuBLAS/cuDNN workspace a further negligible amount), **< Y %** of
> L4 capacity and within vLLM's `gpu_memory_utilization` margin, so it does not reduce
> the serving engine's KV-cache batch capacity during the overlap window.
