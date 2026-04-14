#!/bin/bash
# ─── Performance Estimation & Benchmark Pipeline ─────────────────────
#
# Usage:
#   ./run.sh estimate                  # Step 1: Run estimation (all models)
#   ./run.sh estimate --model llama3-70b --force   # Re-estimate one model
#   ./run.sh generate                  # Step 2: Generate benchmark p files
#   ./run.sh generate --model qwen3-32b            # Generate for one model
#   ./run.sh bench llama3-70b g6_48xlarge tp8_pp1  # Step 3: Run benchmark
#   ./run.sh all                       # Steps 1+2 (estimate → generate)
#
# Before running benchmarks:
#   1. Edit nodes.py — set IP addresses for each instance type
#   2. Ensure worker nodes have latest code (git pull)
#   3. Ray head must be running on head node (ray start --head)
#
# Directory structure (after estimate + generate):
#   {model}/in{N}-out{N}/
#     ├── {instance_dir}/{strategy}.py    ← benchmark scripts
#     └── results/data/
#         ├── estimated/est_*.json        ← estimation results
#         └── measured/bench_*.json       ← benchmark results
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

cmd="${1:-help}"
shift 2>/dev/null || true

case "$cmd" in
  estimate)
    echo "=== Step 1: Running estimation ==="
    python3 estimate.py "$@"
    ;;
  generate)
    echo "=== Step 2: Generating benchmark p files ==="
    python3 generate_p_files.py "$@"
    ;;
  bench)
    MODEL="${1:?Usage: ./run.sh bench <model> <instance_dir> <strategy>}"
    INSTANCE="${2:?Usage: ./run.sh bench <model> <instance_dir> <strategy>}"
    STRATEGY="${3:?Usage: ./run.sh bench <model> <instance_dir> <strategy>}"
    # Find the workload directory (use first available if not specified)
    WORKLOAD="${4:-$(ls -d "$MODEL"/in*-out*/ 2>/dev/null | head -1 | xargs basename)}"
    P_FILE="$MODEL/$WORKLOAD/$INSTANCE/$STRATEGY.py"
    if [ ! -f "$P_FILE" ]; then
      echo "Error: $P_FILE not found. Run './run.sh generate' first."
      exit 1
    fi
    echo "=== Step 3: Running benchmark ==="
    echo "  File: $P_FILE"
    python3 "$P_FILE"
    ;;
  all)
    echo "=== Running estimation + generation ==="
    python3 estimate.py "$@"
    python3 generate_p_files.py "$@"
    echo "=== Done. Edit nodes.py and run benchmarks with: ==="
    echo "  ./run.sh bench <model> <instance_dir> <strategy>"
    ;;
  help|*)
    cat <<EOF
Performance Estimation Pipeline

Steps:
  1. ./run.sh estimate [--model MODEL] [--force]
     Run analytical estimation for all (model, instance, strategy) combinations.
     Results cached to {model}/in{N}-out{N}/results/data/estimated/

  2. ./run.sh generate [--model MODEL]
     Generate benchmark Python scripts from estimation results.
     Output: {model}/in{N}-out{N}/{instance}/{strategy}.py

  3. ./run.sh bench <model> <instance_dir> <strategy> [workload]
     Run a specific benchmark (e.g., ./run.sh bench llama3-70b g6_48xlarge tp8_pp1)

  Shortcut:
     ./run.sh all [--model MODEL] [--force]   — runs steps 1+2

Prerequisites for benchmarks:
  - Edit nodes.py with worker IP addresses
  - Worker nodes: git pull (ensure latest code)
  - Ray head running on head node
  - S3 model weights accessible from workers

Models: llama3-70b, qwen3-32b
Instances: g5_48xlarge, g6_48xlarge, g6e_48xlarge, p4d_24xlarge, p5_48xlarge
Strategies: tp1_pp8, tp2_pp4, tp4_pp2, tp8_pp1
EOF
    ;;
esac
