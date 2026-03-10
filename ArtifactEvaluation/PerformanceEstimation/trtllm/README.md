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
# On host, before Docker
pip install huggingface_hub
huggingface-cli login

huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
  --local-dir /local/models/Llama-3.1-70B-Instruct

huggingface-cli download Qwen/Qwen3-32B \
  --local-dir /local/models/Qwen3-32B
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
python3 benchmarks/cpp/prepare_dataset.py \
  --tokenizer /models/Llama-3.1-70B-Instruct \
  --output /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.json \
  token-norm-dist \
  --num-requests 1000 \
  --input-mean 763 --input-stdev 0 \
  --output-mean 232 --output-stdev 0
```

- `--input-stdev 0` / `--output-stdev 0`: all requests have exactly 763 input tokens and 232 output tokens
- Random token IDs → no early EOS
- `output_tokens` field in JSON controls generation length (not `--max_seq_len`)

### Step 2: Convert Checkpoint

For each (tp, pp) strategy:

```bash
TP=8; PP=1; WORKERS=$((TP * PP))

python3 examples/models/core/llama/convert_checkpoint.py \
  --model_dir /models/Llama-3.1-70B-Instruct \
  --output_dir /trtllm/ckpt/llama3-70b/tp${TP}_pp${PP} \
  --dtype bfloat16 \
  --tp_size $TP --pp_size $PP \
  --workers $WORKERS
```

### Step 3: Build Engine

```bash
TP=8; PP=1; WORKERS=$((TP * PP))

trtllm-build \
  --checkpoint_dir /trtllm/ckpt/llama3-70b/tp${TP}_pp${PP} \
  --output_dir /trtllm/engines/llama3-70b/tp${TP}_pp${PP} \
  --gemm_plugin bfloat16 \
  --gpt_attention_plugin bfloat16 \
  --max_batch_size 256 \
  --max_input_len 763 \
  --max_seq_len 995 \
  --workers $WORKERS
```

**Notes:**
- `--max_seq_len 995` = 763 (input) + 232 (output). Tight allocation maximizes available GPU memory for larger batch sizes.
- `--max_batch_size 256`: set to the largest batch size you plan to test.
- Build time: ~15–30 minutes for 70B model.
- **Engine is GPU-architecture-specific.** Must build on the target GPU (A10G, L4, L40S, A100).

### Step 4: Run Static Batch Benchmark

```bash
TP=8; PP=1; WORLD=$((TP * PP))
BS=32  # exact batch size

mpirun -n $WORLD ./benchmarks/cpp/gptManagerBenchmark \
  --engine_dir /trtllm/engines/llama3-70b/tp${TP}_pp${PP} \
  --static_emulated_batch_size $BS \
  --static_emulated_timeout 5000 \
  --dataset /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.json \
  --streaming
```

- `--static_emulated_batch_size N`: waits for exactly N requests, submits as one batch, waits for completion, then next batch.
- `--static_emulated_timeout 5000`: timeout (ms) if N requests don't arrive (set high enough).
- `--streaming`: enables per-token ITL measurement.
- `--request_rate -1`: (optional) submit all requests immediately.

### Step 5: Batch Size Sweep

Run Step 4 for each batch size and save results:

```bash
TP=8; PP=1; WORLD=$((TP * PP))

for BS in 1 2 4 8 16 32 64 128; do
  echo "=== Running bs=$BS ==="
  mpirun -n $WORLD ./benchmarks/cpp/gptManagerBenchmark \
    --engine_dir /trtllm/engines/llama3-70b/tp${TP}_pp${PP} \
    --static_emulated_batch_size $BS \
    --static_emulated_timeout 10000 \
    --dataset /workspace/llama3-70b/in763-out232/datasets/synthetic_763_232.json \
    --streaming \
    2>&1 | tee /workspace/llama3-70b/in763-out232/measured/trtllm_tp${TP}_pp${PP}_bs${BS}.log
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

## Important Notes

- **Engine is GPU-specific:** An engine built on A10G will NOT run on L4. Rebuild on each GPU type.
- **Checkpoint can be shared:** The converted checkpoint is GPU-independent. Only the engine build is architecture-specific.
- **Engines are large:** ~70B model engine ≈ 30–70GB per strategy. Use ephemeral storage, not workspace.
- **Static batching = no PagedAttention:** KV cache is contiguous, matching the estimator's assumption.
