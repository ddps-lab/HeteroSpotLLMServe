#!/usr/bin/env python3
"""GPU-local memory observer for the concurrent-initialization overhead experiment.

Runs ONE process per node, pinned to all local GPUs. It forces a CUDA context on
each GPU (a 1-element tensor that is kept alive), then serves the device-level
memory usage on request over a small TCP socket.

Why a resident observer instead of an ad-hoc ``python -c`` query:
  ``torch.cuda.mem_get_info(i)`` returns ``(free, total)`` straight from the CUDA
  driver (``cudaMemGetInfo``), so it captures *everything* on the device — CUDA
  contexts, cuBLAS/cuDNN workspaces and NCCL communicator buffers — which all live
  OUTSIDE the torch caching allocator (``memory_reserved``/``memory_allocated``
  would miss them). But any process that queries a GPU must itself hold a context
  (~0.4 GB) on it. By launching this observer ONCE, *before* the tensor store and
  both engines, and keeping it alive across every snapshot, its own context (and the
  store, and engine 1) appear identically in every snapshot and therefore cancel out
  in the S2 - S1 difference that isolates the second engine's overhead.

This is the same ``torch.cuda.mem_get_info`` call the tensor store already uses to
size the KV cache (TensorStore/raw_s3_tensor_store_server.py:505).

Protocol (newline-terminated ASCII over TCP, one command per connection):
  "QUERY\n"    -> JSON line: {"node": <hostname>, "gpus": {"<i>": {"used": B, "free": B, "total": B}}}
  "SHUTDOWN\n" -> "OK\n", then the process exits.

Usage (typically launched via SSH by measure_engine_overhead.py):
  python mem_observer.py --num-gpus 4 --port 9099
"""

import argparse
import json
import socket
import sys

import torch


def snapshot(num_gpus: int) -> dict:
    """Read device-level memory for every local GPU via the CUDA driver."""
    gpus = {}
    for i in range(num_gpus):
        free, total = torch.cuda.mem_get_info(i)
        gpus[str(i)] = {"used": int(total - free), "free": int(free), "total": int(total)}
    return {"node": socket.gethostname(), "gpus": gpus}


def main():
    parser = argparse.ArgumentParser(description="GPU-local memory observer")
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9099)
    args = parser.parse_args()

    # Force a CUDA context on each GPU and keep the holding tensors alive so the
    # contexts persist for the whole experiment.
    holders = []
    for i in range(args.num_gpus):
        torch.cuda.set_device(i)
        holders.append(torch.zeros(1, device=f"cuda:{i}"))
    torch.cuda.synchronize()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    # Readiness marker the orchestrator can grep for in the log.
    print(f"[mem_observer] READY host={socket.gethostname()} gpus={args.num_gpus} port={args.port}", flush=True)

    running = True
    while running:
        conn, _ = srv.accept()
        try:
            conn.settimeout(10.0)
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(64)
                if not chunk:
                    break
                buf += chunk
            cmd = buf.decode("ascii", "ignore").strip().upper()

            if cmd == "QUERY":
                conn.sendall((json.dumps(snapshot(args.num_gpus)) + "\n").encode("ascii"))
            elif cmd == "SHUTDOWN":
                conn.sendall(b"OK\n")
                running = False
            else:
                conn.sendall(b'{"error": "unknown command"}\n')
        except Exception as e:  # noqa: BLE001 - keep the observer alive on bad requests
            try:
                conn.sendall((json.dumps({"error": str(e)}) + "\n").encode("ascii"))
            except OSError:
                pass
        finally:
            conn.close()

    srv.close()
    print("[mem_observer] shutdown", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
