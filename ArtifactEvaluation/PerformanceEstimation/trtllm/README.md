# TensorRT-LLM Pure GPU Benchmark

Measures pure GPU inference latency using TensorRT-LLM's `gptManagerBenchmark` with **static batching** — no Python scheduling overhead, no PagedAttention, contiguous KV cache.

**Purpose:** Validate the roofline-based performance estimator by isolating GPU compute + communication time from framework overhead (scheduling, detokenization, input preparation).

## Prerequisites

- **Docker image:** `nvcr.io/nvidia/tensorrt-llm/release:0.21.0` (CUDA 12.8, compatible with CUDA 12.9 drivers)
- **GPU driver:** ≥ 570 (for CUDA 12.8 containers)
- **Models:** Downloaded to ephemeral storage (e.g., `/local/models/`)

## Storage Layout

| Data | Location (in container) | Host path | Reason |
|---|---|---|---|
| HF models (original) | `/models` | `/local/models/` (ephemeral) | ~140GB per model, download once per instance |
| Converted checkpoints | `/trtllm/ckpt/` | `/local/trtllm/ckpt/` (ephemeral) | ~140GB per model, GPU-independent |
| Built engines | `/trtllm/engines/` | `/local/trtllm/engines/` (ephemeral) | ~30–70GB per strategy, GPU-specific |
| Datasets | `/workspace/{model}/...` | workspace (git repo) | Small (<10MB) |
| Results (logs/JSON) | `/workspace/{model}/...` | workspace (git repo) | Small, trackable |
| Estimator predictions | `/workspace/{model}/...` | workspace (git repo) | Regenerable via estimate.py |

**Ephemeral storage** (`/local/`) is instance-local NVMe — fast and large, but lost on termination.
Models, checkpoints, and engines are NOT stored in the workspace.

### Start Docker Container

```bash
docker run --rm -it --ipc host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /local/models:/models \
  -v /local/trtllm:/trtllm \
  -v /path/to/ShuntServe/ArtifactEvaluation/PerformanceEstimation/trtllm:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:0.21.0
```

### Download Models (on ephemeral storage)

```bash
# Inside Docker container (or on host — both write to the same mounted path)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
  --local-dir /models/Llama-3.1-70B-Instruct

huggingface-cli download Qwen/Qwen3-32B \
  --local-dir /models/Qwen3-32B
```

## Directory Structure

```
# Workspace (git repo)
trtllm/
├── README.md                           # This file (git tracked)
├── llama3-70b/
│   └── in763-out232/
│       ├── datasets/                   # Generated datasets (gitignored)
│       │   └── synthetic_763_232.json
│       ├── predicted/                  # Estimator predictions (regenerate via estimate.py, gitignored)
│       │   └── est_*.json
│       ├── measured/                   # Benchmark results — logs and parsed JSON
│       │   ├── trtllm_tp8_pp1_bs32.log
│       │   └── trtllm_tp8_pp1.json
│       └── results_viewer.ipynb        # Analysis notebook (git tracked)
└── qwen3-32b/
    └── in763-out232/
        └── (same structure)

# Ephemeral storage (instance-local, NOT in git)
/local/
├── models/                             # HF model weights
│   ├── Llama-3.1-70B-Instruct/
│   └── Qwen3-32B/
└── trtllm/
    ├── ckpt/                           # Converted checkpoints
    │   ├── llama3-70b/
    │   │   ├── tp8_pp1/
    │   │   ├── tp4_pp2/
    │   │   ├── tp2_pp4/
    │   │   └── tp1_pp8/
    │   └── qwen3-32b/
    │       └── ...
    └── engines/                        # Built TRT engines (GPU-specific)
        ├── llama3-70b/
        │   ├── tp8_pp1/
        │   └── ...
        └── qwen3-32b/
            └── ...
```

## Pipeline: Convert vs Build

The process has two stages, with different purposes:

| | Convert (Step 2) | Build (Step 3) |
|---|---|---|
| **Input** | HuggingFace model weights | Converted checkpoint |
| **Output** | TRT-LLM checkpoint | TensorRT engine |
| **What it does** | Reshapes weights for TP/PP partitioning | Compiles computation graph + optimizes GPU kernels |
| **GPU needed?** | ❌ CPU only | ✅ Must run on target GPU |
| **GPU-specific?** | No — same checkpoint works on any GPU | **Yes** — engine is architecture-specific (A10G ≠ L4) |
| **Size** | ~same as original model | ~same as original model |
| **Reusable?** | ✅ Across GPU types | ❌ Must rebuild per GPU type |
| **Time** | ~5–10 min (70B) | ~15–30 min (70B) |

**Convert** = "rearrange weights so rank 0 gets layers 0–9 with TP-sliced attention, rank 1 gets layers 10–19, etc."
**Build** = "compile those weights + computation graph into optimized CUDA kernels, fused ops, memory plans for this specific GPU."

## Step-by-Step Guide

### Step 1: Generate Fixed-Length Dataset

```bash
# Inside Docker container
# Generate enough requests for the largest batch sweep (max_bs=1024, ×10 = 10240)
python3 benchmarks/cpp/prepare_dataset.py \
  --tokenizer meta-llama/Llama-3.1-70B-Instruct \
  --output /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.json \
  token-norm-dist \
  --num-requests 10240 \
  --input-mean 763 --input-stdev 0 \
  --output-mean 232 --output-stdev 0
```

- `--input-stdev 0` / `--output-stdev 0`: all requests have exactly 763 input tokens and 232 output tokens
- Random token IDs → no early EOS
- `output_tokens` field in JSON controls generation length (not `--max_seq_len`)
- One dataset for all batch sizes — use `--max_num_samples` at runtime to control request count

### Step 2: Convert Checkpoint (All Strategies, Parallel)

Convert is CPU-only and GPU-independent. All 4 strategies can run in parallel as background jobs.
Each strategy's `--workers` parallelizes across ranks (tp×pp threads), and the 4 strategies run concurrently.

**Llama 3.1 70B:**

```bash
CONVERT_SCRIPT=examples/models/core/llama/convert_checkpoint.py
MODEL=/models/Llama-3.1-70B-Instruct
DTYPE=bfloat16

python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/llama3-70b/tp8_pp1 \
  --dtype $DTYPE --tp_size 8 --pp_size 1 --workers 8 &
python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/llama3-70b/tp4_pp2 \
  --dtype $DTYPE --tp_size 4 --pp_size 2 --workers 8 &
python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/llama3-70b/tp2_pp4 \
  --dtype $DTYPE --tp_size 2 --pp_size 4 --workers 8 &
python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/llama3-70b/tp1_pp8 \
  --dtype $DTYPE --tp_size 1 --pp_size 8 --workers 8 &
wait
echo "All Llama 3.1 70B converts done"
```

**Qwen3 32B:**

```bash
CONVERT_SCRIPT=examples/models/core/llama/convert_checkpoint.py
MODEL=/models/Qwen3-32B
DTYPE=bfloat16

python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/qwen3-32b/tp8_pp1 \
  --dtype $DTYPE --tp_size 8 --pp_size 1 --workers 8 &
python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/qwen3-32b/tp4_pp2 \
  --dtype $DTYPE --tp_size 4 --pp_size 2 --workers 8 &
python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/qwen3-32b/tp2_pp4 \
  --dtype $DTYPE --tp_size 2 --pp_size 4 --workers 8 &
python3 $CONVERT_SCRIPT --model_dir $MODEL --output_dir /trtllm/ckpt/qwen3-32b/tp1_pp8 \
  --dtype $DTYPE --tp_size 1 --pp_size 8 --workers 8 &
wait
echo "All Qwen3 32B converts done"
```

**Notes:**
- All 4 strategies share the same model via mmap — physical RAM ≈ 1× model size regardless of parallelism.
- Converted checkpoints are GPU-independent. Upload to S3 once and reuse across all instance types.
- `convert_checkpoint.py` is the Llama converter but works for Qwen3 as well (same architecture family in TRT-LLM).

### Step 3: Build Engine (Sequential per Strategy)

Build requires GPU — each rank occupies one GPU. Since all 8 GPUs are used per strategy,
strategies must be built sequentially.

**Llama 3.1 70B:**

```bash
DTYPE=bfloat16

for STRATEGY in "8 1" "4 2" "2 4" "1 8"; do
  set -- $STRATEGY; TP=$1; PP=$2
  echo "=== Building llama3-70b tp${TP}_pp${PP} ==="
  trtllm-build \
    --checkpoint_dir /trtllm/ckpt/llama3-70b/tp${TP}_pp${PP} \
    --output_dir /trtllm/engines/llama3-70b/tp${TP}_pp${PP} \
    --gemm_plugin $DTYPE \
    --gpt_attention_plugin $DTYPE \
    --max_batch_size 256 \
    --max_input_len 763 \
    --max_seq_len 995 \
    --workers $((TP * PP))
done
```

**Qwen3 32B:**

```bash
DTYPE=bfloat16

for STRATEGY in "8 1" "4 2" "2 4" "1 8"; do
  set -- $STRATEGY; TP=$1; PP=$2
  echo "=== Building qwen3-32b tp${TP}_pp${PP} ==="
  trtllm-build \
    --checkpoint_dir /trtllm/ckpt/qwen3-32b/tp${TP}_pp${PP} \
    --output_dir /trtllm/engines/qwen3-32b/tp${TP}_pp${PP} \
    --gemm_plugin $DTYPE \
    --gpt_attention_plugin $DTYPE \
    --max_batch_size 256 \
    --max_input_len 763 \
    --max_seq_len 995 \
    --workers $((TP * PP))
done
```

**Notes:**
- `--max_seq_len 995` = 763 (input) + 232 (output). Tight allocation maximizes available GPU memory for larger batch sizes.
- `--max_batch_size 256`: set to the largest batch size you plan to test.
- Build time: ~15–30 minutes per strategy for 70B.
- **Engine is GPU-architecture-specific.** Must build on the target GPU (A10G, L4, L40S, A100).
- `--workers` = tp×pp: each worker uses one GPU (`gpu_id = rank % workers`).

### Step 4: Run Static Batch Benchmark

```bash
TP=8; PP=1; WORLD=$((TP * PP))
BS=32  # exact batch size

mpirun --allow-run-as-root -n $WORLD ./benchmarks/cpp/gptManagerBenchmark \
  --engine_dir /trtllm/engines/llama3-70b/tp${TP}_pp${PP} \
  --static_emulated_batch_size $BS \
  --dataset /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.json \
  --streaming
```

- `--static_emulated_batch_size N`: waits for exactly N requests, submits as one batch, waits for completion, then next batch.
- `--streaming`: enables per-token ITL measurement.
- `--request_rate -1`: (optional) submit all requests immediately.

### Step 5: Batch Size Sweep

Run Step 4 for each batch size (num_requests = batch_size × 10):

```bash
TP=8; PP=1; WORLD=$((TP * PP))

for BS in 1 2 4 8 16 32 64 128 256 512 1024; do
  SAMPLES=$((BS * 10))
  echo "=== Running bs=$BS, samples=$SAMPLES ==="
  mpirun --allow-run-as-root -n $WORLD ./benchmarks/cpp/gptManagerBenchmark \
    --engine_dir /trtllm/engines/llama3-70b/tp${TP}_pp${PP} \
    --static_emulated_batch_size $BS \
    --dataset /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.json \
    --max_num_samples $SAMPLES \
    --streaming \
    --output_csv /workspace/llama3-70b/in763-out232/measured/csv/trtllm_tp${TP}_pp${PP}_bs${BS}.csv \
    2>&1 | tee /workspace/llama3-70b/in763-out232/measured/log/trtllm_tp${TP}_pp${PP}_bs${BS}.log
done
```

### Step 6: Repeat for All Strategies

Build and benchmark all 4 strategies:

| Strategy | TP | PP | mpirun -n |
|---|---|---|---|
| tp8_pp1 | 8 | 1 | 8 |
| tp4_pp2 | 4 | 2 | 8 |
| tp2_pp4 | 2 | 4 | 8 |
| tp1_pp8 | 1 | 8 | 8 |

Each requires separate checkpoint conversion + engine build.

## Output Metrics

`gptManagerBenchmark` with `--streaming` outputs to **stdout** (no `--report_json`). Save as log files, parse later.

| Metric | Description | Estimator Comparison |
|---|---|---|
| **TTFT (ms)** | Time to First Token = prefill latency | `prefill_latency_ms` |
| **ITL (ms)** | Inter-Token Latency = per-decode-step time | `decode_latency_ms / output_len` |
| **TPOT (ms)** | Time per Output Token (avg) | ≈ ITL average |
| **E2EL (ms)** | End-to-End Latency | `prefill + decode` |
| **Throughput** | Request throughput (req/sec) | `throughput` |

### Comparison with Estimator

```
Estimator TTFT   ↔ TRT-LLM TTFT
Estimator TPOT   ↔ TRT-LLM mean ITL (or TPOT)
Estimator E2E    ↔ TRT-LLM E2EL
Estimator RPS    ↔ TRT-LLM Request Throughput
```

## Model Configs

### Llama 3.1 70B

| Param | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-70B-Instruct` |
| Layers | 80 |
| Hidden dim | 8192 |
| Attention heads | 64 |
| KV heads | 8 |
| Intermediate dim | 28672 |
| Vocab size | 128256 |

### Qwen3 32B

| Param | Value |
|---|---|
| Model | `Qwen/Qwen3-32B` |
| Layers | 64 |
| Hidden dim | 5120 |
| Attention heads | 64 |
| KV heads | 8 |
| Head dim | 128 |
| Intermediate dim | 27648 |
| Vocab size | 151936 |

## GPU Compatibility

| Instance | GPU | Arch | CUDA CC | FP8 |
|---|---|---|---|---|
| g5.48xlarge | A10G | Ampere | SM86 | ❌ |
| g6.48xlarge | L4 | Ada | SM89 | ✅ |
| g6e.48xlarge | L40S | Ada | SM89 | ✅ |
| p4d.24xlarge | A100 40GB | Ampere | SM80 | ❌ |
| p5.48xlarge | H100 80GB | Hopper | SM90 | ✅ |

**Use BF16/FP16 for all benchmarks** (consistent with estimator assumptions).

## Important Notes (gptManagerBenchmark)

- **Engine is GPU-specific:** An engine built on A10G will NOT run on L4. Rebuild on each GPU type.
- **Checkpoint can be shared:** The converted checkpoint is GPU-independent. Only the engine build is architecture-specific.
- **Engines are large:** ~70B model engine ≈ 30–70GB per strategy. Use ephemeral storage, not workspace.
- **Static batching = no PagedAttention:** KV cache is contiguous, matching the estimator's assumption.

---

# Alternative: trtllm-bench (PyTorch Backend)

A faster alternative that **skips engine build entirely**. Uses PyTorch kernels (no TRT compilation)
with TRT-LLM's C++ scheduling — closer to roofline assumptions than TRT-optimized kernels.

## trtllm-bench vs gptManagerBenchmark

| | gptManagerBenchmark (static) | trtllm-bench (PyTorch) |
|---|---|---|
| Engine build | **Required** (~30 min/strategy) | **Not needed** |
| Kernels | TRT-optimized (fusion, custom GEMM) | Standard PyTorch (cuBLAS) |
| KV cache | Contiguous (no PagedAttention) | PagedAttention (~10% overhead) |
| Batch control | Exact (`--static_emulated_batch_size`) | Indirect (`--concurrency`) |
| Output format | stdout log (parse manually) | `--report_json` (native JSON) |
| Roofline match | KV cache ✅, kernels ❌ (too optimized) | KV cache ❌, kernels ✅ (standard) |
| Setup time | ~2 hours (convert + build all) | ~5 minutes |

**Recommendation:** Start with trtllm-bench PyTorch backend. If results are close to estimator (MAPE < 20%),
CPU overhead was the main gap. If not, use gptManagerBenchmark for deeper analysis.

## Dataset Preparation (trtllm-bench format)

trtllm-bench uses **JSONL format** (one JSON object per line), different from gptManagerBenchmark's JSON array.

```bash
# Inside Docker container
python3 benchmarks/cpp/prepare_dataset.py \
  --stdout \
  --tokenizer /models/Llama-3.1-70B-Instruct \
  token-norm-dist \
  --num-requests 10240 \
  --input-mean 763 --input-stdev 0 \
  --output-mean 232 --output-stdev 0 \
  > /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.jsonl

python3 benchmarks/cpp/prepare_dataset.py \
  --stdout \
  --tokenizer /models/Qwen3-32B \
  token-norm-dist \
  --num-requests 10240 \
  --input-mean 763 --input-stdev 0 \
  --output-mean 232 --output-stdev 0 \
  > /workspace/qwen3-32b/in763-out232/datasets/synthetic_763_232.jsonl
```

## GPU Setup (Before Benchmarking)

```bash
# On host, before Docker (or inside with --privileged)
sudo nvidia-smi -pm 1                    # Persistence mode
sudo nvidia-smi -rgc                     # Reset GPU clocks (let GPU auto-boost)
sudo nvidia-smi -pl $(nvidia-smi -q -d POWER | grep "Max Power" | head -1 | awk '{print $5}')  # Max power
```

## Running trtllm-bench (PyTorch Backend)

### Single Run

```bash
trtllm-bench \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --model_path /models/Llama-3.1-70B-Instruct \
  throughput \
  --backend pytorch \
  --dataset /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.jsonl \
  --tp 8 --pp 1 \
  --concurrency 32 \
  --num_requests 320 \
  --streaming \
  --report_json /workspace/llama3-70b/in763-out232/measured/trtllm-bench_tp8_pp1_c32.json
```

**Key options:**
- `--backend pytorch`: Use PyTorch kernels, skip engine build
- `--model_path`: Path to local HF model (avoids re-download)
- `--concurrency N`: Maximum concurrent requests (≈ batch size)
- `--num_requests`: Total requests to process (= concurrency × 10)
- `--streaming`: Enables per-token ITL measurement
- `--report_json`: Save detailed results as JSON

### Batch Size Sweep — Llama 3.1 70B

```bash
MODEL=meta-llama/Llama-3.1-70B-Instruct
MODEL_PATH=/models/Llama-3.1-70B-Instruct
DATASET=/workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.jsonl
OUTDIR=/workspace/llama3-70b/in763-out232/measured

for STRATEGY in "8 1" "4 2" "2 4" "1 8"; do
  set -- $STRATEGY; TP=$1; PP=$2
  for C in 1 2 4 8 16 32 64 128 256 512 1024; do
    N=$((C * 10))
    echo "=== tp${TP}_pp${PP} concurrency=$C requests=$N ==="
    trtllm-bench \
      --model $MODEL \
      --model_path $MODEL_PATH \
      throughput \
      --backend pytorch \
      --dataset $DATASET \
      --tp $TP --pp $PP \
      --concurrency $C \
      --num_requests $N \
      --streaming \
      --report_json ${OUTDIR}/trtllm-bench_tp${TP}_pp${PP}_c${C}.json
  done
done
```

### Batch Size Sweep — Qwen3 32B

```bash
MODEL=Qwen/Qwen3-32B
MODEL_PATH=/models/Qwen3-32B
DATASET=/workspace/qwen3-32b/in763-out232/datasets/synthetic_763_232.jsonl
OUTDIR=/workspace/qwen3-32b/in763-out232/measured

for STRATEGY in "8 1" "4 2" "2 4" "1 8"; do
  set -- $STRATEGY; TP=$1; PP=$2
  for C in 1 2 4 8 16 32 64 128 256 512 1024; do
    N=$((C * 10))
    echo "=== tp${TP}_pp${PP} concurrency=$C requests=$N ==="
    trtllm-bench \
      --model $MODEL \
      --model_path $MODEL_PATH \
      throughput \
      --backend pytorch \
      --dataset $DATASET \
      --tp $TP --pp $PP \
      --concurrency $C \
      --num_requests $N \
      --streaming \
      --report_json ${OUTDIR}/trtllm-bench_tp${TP}_pp${PP}_c${C}.json
  done
done
```

## trtllm-bench Output Metrics

With `--report_json`, output includes:

```json
{
  "metadata": {
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "tp_size": 8,
    "pp_size": 1,
    "dtype": "bfloat16"
  },
  "summary": {
    "request_throughput": 43.21,
    "total_output_throughput": 5530.74,
    "avg_ttft_ms": 512.03,
    "avg_tpot_ms": 18.96,
    "avg_e2el_ms": 4903.45
  },
  "percentiles": {
    "ttft_ms": { "p50": ..., "p90": ..., "p95": ..., "p99": ... },
    "tpot_ms": { "p50": ..., "p90": ..., "p95": ..., "p99": ... },
    "itl_ms":  { "p50": ..., "p90": ..., "p95": ..., "p99": ... },
    "e2el_ms": { "p50": ..., "p90": ..., "p95": ..., "p99": ... }
  },
  "request_metrics": [
    {
      "request_id": 0,
      "ttft_ms": 391.76,
      "e2el_ms": 4189.08,
      "output_tokens": 232,
      "inter_token_latencies_ms": [1.59, 1.61, ...]
    }
  ]
}
```

## Important Notes (trtllm-bench)

- **`--concurrency` ≠ exact batch size:** Inflight batching may schedule fewer or more requests per step depending on available KV cache. With fixed-length requests, concurrency ≈ batch size in practice.
- **PagedAttention is always on:** ~10% overhead vs contiguous KV cache (vAttention paper). This is a known difference from the estimator.
- **First run is slow:** PyTorch backend compiles CUDA graphs on first iteration. Subsequent runs are faster. trtllm-bench runs a warmup automatically.
- **Model stays in GPU memory between concurrency levels:** Within the same `trtllm-bench` invocation, the model is loaded once. But the sweep script above restarts per concurrency level, so model is reloaded each time. This is acceptable for correctness.
- **PP support in PyTorch backend:** Verify `--pp > 1` works; some versions may have limitations.
