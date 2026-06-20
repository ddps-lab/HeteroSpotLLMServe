#!/usr/bin/env python3
"""Measure the GPU memory overhead of a 2nd vLLM engine that shares one tensor store.

Rebuttal R2#4: during ShuntServe's concurrent initialization two engine processes
coexist on the same GPUs while sharing ONE tensor store. Because the store holds the
model weights and KV cache exactly once, the *marginal* footprint of the second engine
is only its duplicate CUDA context + cuBLAS/cuDNN workspaces + NCCL communicators.

This script reproduces that controlled worst case on 2x g6.12xlarge (8x L4), TP=4
intra-node + PP=2 inter-node (mirroring SS-P1's parallel_strategy=[4,4,...]), and
reports the aggregate per-engine overhead:

    observers (resident meters)
      -> shared tensor store (8 procs)            ===> snapshot S0
      -> engine 1 (Ray cluster 6379, API :8001)   ===> snapshot S1
      -> engine 2 (Ray cluster 6380, API :8002)   ===> snapshot S2     (no 2nd store)

    Headline:  dE2 = used(S2) - used(S1)   per GPU, summed/averaged over the 8 GPUs.
    Context:   used(S1) - used(S0)         = first-engine cost.

Memory is read with torch.cuda.mem_get_info (the same call the store uses to size KV,
TensorStore/raw_s3_tensor_store_server.py:505) via a resident per-node observer whose
own context cancels out in the S2-S1 difference (see mem_observer.py).

It reuses ShuntServe's own launch-command builders so the engines/stores are configured
exactly as in production: GlobalServer/command.py and protocols.py. No ShuntServe or
vLLM source is modified.

Prerequisites on the remote nodes (g6.12xlarge, user 'ubuntu'):
  - ShuntServe checked out at /home/ubuntu/ShuntServe (see GlobalServer/command.py paths)
  - passwordless SSH from this driver host to both node IPs (and to N0's own IP)
  - S3 weights present for the chosen model
  - This script runs ON the head node N0 (the first --nodes IP), like the Ray head.

Usage:
  python measure_engine_overhead.py --nodes <N0_IP> <N1_IP> --model 8b  --out results_8b.json
  python measure_engine_overhead.py --nodes <N0_IP> <N1_IP> --model 70b --out results_70b.json
  # if you manage the two Ray heads yourself (README.md:185-186 + init_ray_workers.sh):
  python measure_engine_overhead.py --nodes <N0_IP> <N1_IP> --model 8b --skip-ray-setup
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

# ── Repo wiring: import ShuntServe's command builders and protocol enums ──────────
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    parent = os.path.dirname(_d)
    if parent == _d:
        raise RuntimeError("Could not locate repo root (.git)")
    _d = parent
_REPO_ROOT = _d
sys.path.insert(0, _REPO_ROOT)                       # protocols.py
sys.path.insert(0, os.path.join(_REPO_ROOT, "GlobalServer"))  # command.py

from command import (  # noqa: E402
    PYTHON, PROJECT_PATH, RAY,
    get_tensor_store_command, get_api_server_command,
)
from protocols import TensorStoreRequest, TensorStoreResponse  # noqa: E402

SSH_OPTS = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# Ports (bases; per-GPU offset = local_rank is added by the builders / by us)
TS_STATUS_BASE = 10001   # tensor store status/manager base (command.py DEFAULT_TENSOR_STORE_BASE_PORT)
API_PORT_1 = 8001
API_PORT_2 = 8002
RAY_PORT_1 = 6379
RAY_PORT_2 = 6380
OBSERVER_PORT = 9099

OBSERVER_REMOTE = f"{PROJECT_PATH}/ArtifactEvaluation/MemoryOverhead/mem_observer.py"

# ── Per-model configs (both run on 2x g6.12xlarge, TP=4 + PP=2) ───────────────────
_COMMON = {
    "model_source": "s3",
    # No --dtype: the tensor store has no such arg (weight dtype comes from the TRAW
    # upload) and vLLM auto-detects bfloat16 for Llama-3.1. Mirrors production configs,
    # which carry no "dtype" field. cache_dtype="auto" is a separate, accepted arg.
    "dtype": None,
    "cache_dtype": "auto",
    "block_size": 16,
    "swap_space": 4.0,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 256,
    "parallel_strategy": [4, 4],   # TP=4 per stage, 2 PP stages -> one stage per node
}
CONFIGS = {
    "8b": {
        **_COMMON,
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "s3_path": "s3://hetero-spot-llm-serve-models/meta-llama/Llama-3.1-8B-Instruct",
        "total_num_layers": 32,
        "pp_layer_partition": "16,16",
        "num_gpu_blocks": 4096,
    },
    "70b": {
        **_COMMON,
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "s3_path": "s3://hetero-spot-llm-serve-models/meta-llama/Llama-3.1-70B-Instruct",
        "total_num_layers": 80,
        "pp_layer_partition": "40,40",
        # ~17.5 GB/GPU of weights on L4 (24 GB); keep KV small so weights + both
        # engines' context/NCCL + KV all fit. Tune up if there is headroom.
        "num_gpu_blocks": 512,
    },
}

GiB = 1024 ** 3


# ── SSH / process helpers (mirror GlobalServer/VNode.py launch patterns) ──────────

def _target(node_ip, user):
    return f"{user}@{node_ip}" if user else node_ip


def ssh_bg(node_ip, user, command, local_log_path):
    """Launch a long-running remote server; stream its log to a local file. Returns Popen."""
    os.makedirs(os.path.dirname(local_log_path), exist_ok=True)
    full = f"ssh {SSH_OPTS} {_target(node_ip, user)} '{command}' > {local_log_path} 2>&1 &"
    print(f"  [ssh-bg {node_ip}] {command[:110]}{'...' if len(command) > 110 else ''}")
    return subprocess.Popen(full, shell=True)


def ssh_run(node_ip, user, command, timeout=180):
    full = f"ssh {SSH_OPTS} {_target(node_ip, user)} '{command}'"
    return subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)


# ── Observers ─────────────────────────────────────────────────────────────────────

def query_observer(node_ip, port=OBSERVER_PORT, timeout=10.0):
    with socket.create_connection((node_ip, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(b"QUERY\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("ascii").strip())


def shutdown_observer(node_ip, port=OBSERVER_PORT):
    try:
        with socket.create_connection((node_ip, port), timeout=5.0) as s:
            s.sendall(b"SHUTDOWN\n")
            s.recv(16)
    except OSError:
        pass


def snapshot(node_ips):
    """Return {(node_ip, gpu_idx): used_bytes} across all GPUs of all nodes."""
    out = {}
    for ip in node_ips:
        gpus = query_observer(ip)["gpus"]
        for gidx, m in gpus.items():
            out[(ip, int(gidx))] = m["used"]
    return out


# ── Tensor store status (protocols.py) ────────────────────────────────────────────

def store_ready(node_ip, local_rank, timeout=2.0):
    try:
        with socket.create_connection((node_ip, TS_STATUS_BASE + local_rank), timeout=timeout) as s:
            s.settimeout(timeout)
            s.send(TensorStoreRequest.STATUS_CHECK.value)
            return s.recv(1) == TensorStoreResponse.READY.value
    except OSError:
        return False


def store_shutdown(node_ip, local_rank, timeout=5.0):
    try:
        with socket.create_connection((node_ip, TS_STATUS_BASE + local_rank), timeout=timeout) as s:
            s.settimeout(timeout)
            s.send(TensorStoreRequest.SHUTDOWN.value)
            s.recv(1)
    except OSError:
        pass


# ── API server (HTTP) ─────────────────────────────────────────────────────────────

def http_ok(url, timeout=3.0, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as e:  # noqa: BLE001
        return None, str(e).encode()


def wait_until(predicate, what, timeout, interval=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    raise TimeoutError(f"Timed out after {timeout}s waiting for: {what}")


# ── Node layout from the model config ─────────────────────────────────────────────

def node_layout(cfg, node_ips):
    """[(node_ip, pp_rank, tp_size, start_layer, end_layer)] + node_rank_mapping."""
    parts = [int(x) for x in cfg["pp_layer_partition"].split(",")]
    tp = cfg["parallel_strategy"]
    assert len(parts) == len(tp) == len(node_ips), (
        f"pp_layer_partition({len(parts)}) / parallel_strategy({len(tp)}) / nodes({len(node_ips)}) mismatch")
    layout, nrm, layer, rank = [], {}, 0, 0
    for ip, n_layers, tp_size in zip(node_ips, parts, tp):
        layout.append((ip, len(layout), tp_size, layer, layer + n_layers))
        nrm[ip] = list(range(rank, rank + tp_size))
        layer += n_layers
        rank += tp_size
    return layout, nrm


# ── Orchestration steps ───────────────────────────────────────────────────────────

def launch_observers(node_ips, user, logdir):
    procs = []
    for ip in node_ips:
        cmd = f"{PYTHON} {OBSERVER_REMOTE} --num-gpus 4 --port {OBSERVER_PORT}"
        procs.append(ssh_bg(ip, user, cmd, os.path.join(logdir, f"observer_{ip}.log")))
    wait_until(lambda: all(_observer_up(ip) for ip in node_ips),
               "observers ready", timeout=120)
    print("  observers ready")
    return procs


def _observer_up(ip):
    try:
        query_observer(ip, timeout=3.0)
        return True
    except OSError:
        return False


def launch_store(cfg, layout, user, logdir):
    procs = []
    for ip, pp_rank, tp_size, start, end in layout:
        for lr in range(tp_size):
            cmd = get_tensor_store_command(
                model_name=cfg["model_name"],
                tensor_parallel_size=tp_size,
                tensor_parallel_rank=lr,
                local_rank=lr,
                pipeline_parallel_size=len(layout),
                pipeline_parallel_rank=pp_rank,
                start_layer_id=start,
                end_layer_id=end,
                block_size=cfg["block_size"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                swap_space=cfg["swap_space"],
                cache_dtype=cfg["cache_dtype"],
                max_model_len=cfg["max_model_len"],
                status_port=TS_STATUS_BASE,
                dtype=cfg["dtype"],
                s3_path=cfg["s3_path"],
                gpu_num_blocks=cfg["num_gpu_blocks"],
            )
            log = os.path.join(logdir, f"store_{ip}_lr{lr}.log")
            procs.append(ssh_bg(ip, user, cmd, log))

    def all_ready():
        return all(store_ready(ip, lr) for ip, _, tp, _, _ in layout for lr in range(tp))

    wait_until(all_ready, "tensor store ready (all ranks)", timeout=1800)
    print("  tensor store ready (weights + KV loaded)")
    return procs


def launch_engine(cfg, layout, nrm, node_ips, head_ip, api_port, ray_port, user, logdir, tag):
    n0_ip = layout[0][0]  # PP rank 0 hosts the API server
    cmd = get_api_server_command(
        model_name=cfg["model_name"],
        pp_layer_partition=cfg["pp_layer_partition"],
        parallel_strategy=cfg["parallel_strategy"],
        host="0.0.0.0",
        node_rank_mapping=json.dumps(nrm),
        ray_address=f"{head_ip}:{ray_port}",
        num_gpu_blocks_override=cfg["num_gpu_blocks"],
        port=api_port,
        dtype=cfg["dtype"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        max_num_seqs=cfg["max_num_seqs"],
        enforce_eager=True,
    )
    log = os.path.join(logdir, f"engine_{tag}_{n0_ip}.log")
    proc = ssh_bg(n0_ip, user, cmd, log)

    health = f"http://{n0_ip}:{api_port}/health"
    wait_until(lambda: http_ok(health)[0] == 200, f"engine {tag} /health", timeout=1800)
    time.sleep(5)  # settle
    print(f"  engine {tag} healthy on {n0_ip}:{api_port}")
    # NOTE: at /health the CUDA context AND the NCCL communicators are already
    # allocated — vLLM's PyNcclCommunicator.__init__ calls ncclCommInitRank + an eager
    # warmup all_reduce at distributed setup (submodules/vLLM/.../pynccl.py:99-105).
    # Only the cuBLAS/cuDNN GEMM workspace is still lazy (needs a real forward), which
    # is what warmup_engine() below realizes. So the health snapshot already captures
    # context + NCCL — the dominant, overlap-relevant part of the overhead.
    return proc, n0_ip, api_port


def warmup_engine(n0_ip, api_port, model_name):
    """One real forward → allocates the lazy cuBLAS/cuDNN GEMM workspace (and the
    served-path scratch). The eager all_reduce at init is not a GEMM, so this is the
    only step that realizes the cuBLAS/cuDNN component of the overhead."""
    status, _ = http_ok(
        f"http://{n0_ip}:{api_port}/v1/completions", timeout=120, method="POST",
        payload={"model": model_name, "prompt": "Hello", "max_tokens": 4, "temperature": 0.0},
    )
    print(f"  warmup {n0_ip}:{api_port}: status={status}")
    time.sleep(5)  # settle


def cleanup_nodes(node_ips, user, stop_ray):
    """Clean slate before a run so re-runs are idempotent (frees the observer port,
    kills any store/engine left from a failed run). Safe at startup; NEVER call this
    mid-measurement (it would kill the shared store)."""
    for ip in node_ips:
        ssh_run(ip, user,
                "pkill -f mem_observer.py; pkill -f raw_s3_tensor_store_server.py; "
                "pkill -f InferenceServer/api_server.py; true", timeout=30)
        if stop_ray:
            ssh_run(ip, user, f"{RAY} stop --force", timeout=60)


def ray_setup(node_ips, head_ip, ray_port, user):
    """Start a Ray head + join workers, exactly as README.md:185-186 (no temp-dir).

    Two heads (6379/6380) coexist on the same node under Ray's default temp dir:
    ShuntServe always addresses clusters by explicit host:port (never "auto"), so the
    shared /tmp/ray ray_current_cluster pointer is irrelevant. This mirrors the proven
    `ray start --head` + SpotTolerance/init_ray_workers.sh flow.
    """
    head_cmd = f"{RAY} start --head --port={ray_port} --disable-usage-stats"
    r = ssh_run(head_ip, user, head_cmd)
    print(f"  ray head :{ray_port} on {head_ip} rc={r.returncode}")
    for ip in node_ips:
        if ip == head_ip:
            continue
        join = f"{RAY} start --address={head_ip}:{ray_port} --disable-usage-stats"  # == get_ray_start_worker_command
        r = ssh_run(ip, user, join)
        print(f"  ray join {ip} -> :{ray_port} rc={r.returncode}")


def report(snaps, node_ips):
    """Decompose the 2nd-engine overhead using 5 snapshots.

      S0  = store only
      S1h = store + engine1 (just /health: CUDA context + NCCL, both eager)
      S1w = store + engine1 warmed (adds engine1's cuBLAS/cuDNN GEMM workspace)
      S2h = store + engine1(warmed) + engine2 (/health)
      S2w = store + engine1(warmed) + engine2 warmed

    Second-engine overhead components (per GPU):
      ctx+NCCL (eager)        = S2h - S1w   <- this is the OVERLAP-window cost: engine2
                                              only reaches /health before switchover, so
                                              its cuBLAS workspace isn't allocated yet
      + cuBLAS/cuDNN (lazy)   = S2w - S2h   <- realized once it starts serving
      full marginal           = S2w - S1w
    """
    s0, s1h, s1w = snaps["S0"], snaps["S1h"], snaps["S1w"]
    s2h, s2w = snaps["S2h"], snaps["S2w"]
    keys = sorted(s0.keys())
    rows = {m: 0 for m in ("e1_ctx_nccl", "e1_cublas", "e2_ctx_nccl", "e2_cublas", "e2_full")}
    per_gpu = []
    for k in keys:
        e1_ctx = s1h[k] - s0[k]
        e1_blas = s1w[k] - s1h[k]
        e2_ctx = s2h[k] - s1w[k]
        e2_blas = s2w[k] - s2h[k]
        e2_full = s2w[k] - s1w[k]
        rows["e1_ctx_nccl"] += e1_ctx
        rows["e1_cublas"] += e1_blas
        rows["e2_ctx_nccl"] += e2_ctx
        rows["e2_cublas"] += e2_blas
        rows["e2_full"] += e2_full
        per_gpu.append({
            "node": k[0], "gpu": k[1],
            "S0_GiB": round(s0[k] / GiB, 3), "S1h_GiB": round(s1h[k] / GiB, 3),
            "S1w_GiB": round(s1w[k] / GiB, 3), "S2h_GiB": round(s2h[k] / GiB, 3),
            "S2w_GiB": round(s2w[k] / GiB, 3),
            "e2_ctx_nccl_GiB": round(e2_ctx / GiB, 3),
            "e2_cublas_GiB": round(e2_blas / GiB, 3),
            "e2_full_GiB": round(e2_full / GiB, 3),
        })
    n = len(keys)

    def g(b):
        return b / GiB

    print("\n" + "=" * 86)
    print("Per-GPU 2nd-engine overhead (GiB)   [ctx+NCCL eager @health | +cuBLAS/cuDNN @warmup]")
    print("=" * 86)
    print(f"{'node':>15} {'gpu':>3} {'S1w':>8} {'S2h':>8} {'S2w':>8} "
          f"{'ctx+NCCL':>9} {'+cuBLAS':>8} {'full':>7}")
    for r in per_gpu:
        print(f"{r['node']:>15} {r['gpu']:>3} {r['S1w_GiB']:>8.3f} {r['S2h_GiB']:>8.3f} "
              f"{r['S2w_GiB']:>8.3f} {r['e2_ctx_nccl_GiB']:>9.3f} {r['e2_cublas_GiB']:>8.3f} "
              f"{r['e2_full_GiB']:>7.3f}")
    print("-" * 86)
    print(f"  2nd-engine, ctx+NCCL (overlap-window cost): "
          f"mean {g(rows['e2_ctx_nccl']) / n:.3f} GiB/GPU  (total {g(rows['e2_ctx_nccl']):.3f})")
    print(f"  2nd-engine, + cuBLAS/cuDNN (when serving) : "
          f"mean {g(rows['e2_cublas']) / n:.3f} GiB/GPU  (total {g(rows['e2_cublas']):.3f})")
    print(f"  2nd-engine, FULL marginal                 : "
          f"mean {g(rows['e2_full']) / n:.3f} GiB/GPU  (total {g(rows['e2_full']):.3f})")
    e1_full = rows["e1_ctx_nccl"] + rows["e1_cublas"]
    print(f"  1st-engine, full (context):                 "
          f"mean {g(e1_full) / n:.3f} GiB/GPU  (total {g(e1_full):.3f})")
    if e1_full:
        print(f"  2nd full / 1st full                       : {100.0 * rows['e2_full'] / e1_full:.1f}%"
              f"  (lower = second engine is nearly free)")
    print("=" * 86)
    return {
        "per_gpu": per_gpu, "n_gpus": n,
        "e2_ctx_nccl_mean_GiB_per_gpu": round(g(rows["e2_ctx_nccl"]) / n, 4),
        "e2_cublas_mean_GiB_per_gpu": round(g(rows["e2_cublas"]) / n, 4),
        "e2_full_mean_GiB_per_gpu": round(g(rows["e2_full"]) / n, 4),
        "e2_full_total_GiB": round(g(rows["e2_full"]), 4),
        "e1_full_mean_GiB_per_gpu": round(g(e1_full) / n, 4),
    }


def teardown(node_ips, layout, engines, user):
    """Graceful, store-safe teardown. Order: engines -> store -> ray -> observers."""
    print("\n[teardown]")
    for n0_ip, api_port in engines:
        http_ok(f"http://{n0_ip}:{api_port}/shutdown", timeout=10, method="POST", payload={})
    time.sleep(3)
    for ip, _, tp_size, _, _ in layout:
        for lr in range(tp_size):
            store_shutdown(ip, lr)          # store's own SHUTDOWN protocol — never pkill
    for ip in node_ips:
        ssh_run(ip, user, f"{RAY} stop --force", timeout=60)
    for ip in node_ips:
        shutdown_observer(ip)
    print("[teardown] done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", nargs=2, required=True, metavar=("N0_IP", "N1_IP"),
                    help="private IPs of the two g6.12xlarge nodes (N0 = head/API host)")
    ap.add_argument("--model", choices=list(CONFIGS), default="8b")
    ap.add_argument("--ssh-user", default=None, help="SSH user (default: ssh config default)")
    ap.add_argument("--head-ip", default=None, help="Ray head IP (default: N0)")
    ap.add_argument("--skip-ray-setup", action="store_true",
                    help="assume two Ray heads (6379/6380) + joined workers already exist")
    ap.add_argument("--keep-up", action="store_true", help="do not tear down after measuring")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    cfg = CONFIGS[args.model]
    node_ips = list(args.nodes)
    head_ip = args.head_ip or node_ips[0]
    user = args.ssh_user
    here = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(here, "logs", args.model)
    resultsdir = os.path.join(here, "results")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(resultsdir, exist_ok=True)
    layout, nrm = node_layout(cfg, node_ips)

    print(f"Model: {cfg['model_name']}  | nodes: {node_ips}  | TP/PP: {cfg['parallel_strategy']} / "
          f"{cfg['pp_layer_partition']}")
    print(f"node_rank_mapping: {nrm}  (identical for both engines -> rank i -> cuda:i -> store :{50001}+i)")
    print(f"logs: {logdir}\n")

    snaps, engines = {}, []
    try:
        # Clean slate: free the observer port + kill any store/engine from a failed run
        # (and stop Ray unless the operator manages it). Makes re-runs idempotent.
        print("[0/5] cleanup stale procs")
        cleanup_nodes(node_ips, user, stop_ray=not args.skip_ray_setup)
        print("[1/5] observers")
        launch_observers(node_ips, user, logdir)

        print("[2/5] shared tensor store")
        launch_store(cfg, layout, user, logdir)
        snaps["S0"] = snapshot(node_ips)
        print("  -> S0 captured (store only)")

        print("[3/5] engine 1 (Ray 6379, API 8001)")
        if not args.skip_ray_setup:
            ray_setup(node_ips, head_ip, RAY_PORT_1, user)
        _, n0, p1 = launch_engine(cfg, layout, nrm, node_ips, head_ip, API_PORT_1, RAY_PORT_1,
                                  user, logdir, "1")
        engines.append((n0, p1))
        snaps["S1h"] = snapshot(node_ips)
        print("  -> S1h (store + engine1 @health: ctx+NCCL)")
        warmup_engine(n0, p1, cfg["model_name"])
        snaps["S1w"] = snapshot(node_ips)
        print("  -> S1w (engine1 warmed: +cuBLAS/cuDNN)")

        print("[4/5] engine 2 (Ray 6380, API 8002) — SAME GPUs, SAME store, NO new store")
        if not args.skip_ray_setup:
            ray_setup(node_ips, head_ip, RAY_PORT_2, user)
        _, n0b, p2 = launch_engine(cfg, layout, nrm, node_ips, head_ip, API_PORT_2, RAY_PORT_2,
                                   user, logdir, "2")
        engines.append((n0b, p2))
        snaps["S2h"] = snapshot(node_ips)
        print("  -> S2h (+ engine2 @health: overlap-window state)")
        warmup_engine(n0b, p2, cfg["model_name"])
        snaps["S2w"] = snapshot(node_ips)
        print("  -> S2w (engine2 warmed: +cuBLAS/cuDNN)")

        print("[5/5] report")
        summary = report(snaps, node_ips)

        result = {
            "model": cfg["model_name"], "model_key": args.model,
            "nodes": node_ips, "parallel_strategy": cfg["parallel_strategy"],
            "pp_layer_partition": cfg["pp_layer_partition"],
            "num_gpu_blocks": cfg["num_gpu_blocks"],
            "gpu_memory_utilization": cfg["gpu_memory_utilization"],
            "node_rank_mapping": nrm,
            "snapshots_bytes": {k: {f"{ip}:{g}": v for (ip, g), v in s.items()} for k, s in snaps.items()},
            "summary": summary,
        }
        out = args.out or os.path.join(resultsdir, f"overhead_{args.model}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out}")
    finally:
        if not args.keep_up:
            try:
                teardown(node_ips, layout, engines, user)
            except Exception as e:  # noqa: BLE001
                print(f"[teardown] error: {e}")
        else:
            print("\n--keep-up: leaving store/engines/observers running. "
                  "Tear down manually (ray stop --force; store SHUTDOWN; observer SHUTDOWN).")


if __name__ == "__main__":
    main()
