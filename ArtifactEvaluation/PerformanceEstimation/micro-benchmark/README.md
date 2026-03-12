# Hardware Micro-Benchmark

Measures **effective** HBM bandwidth, TFLOPS, FlashAttention throughput (prefill & decode), and inter-GPU communication bandwidth using saturated workloads. All benchmarks sweep batch sizes from 1 to `--max-batch` (powers of 2).

## Environment Setup

Run inside the NVIDIA PyTorch container to ensure compatible CUDA, NCCL, and FlashAttention versions:

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  -v ~/ShuntServe:/workspace/ShuntServe \
  nvcr.io/nvidia/pytorch:24.12-py3
```

Then inside the container:

```bash
cd /workspace/ShuntServe/ArtifactEvaluation/PerformanceEstimation/micro-benchmark
pip install flash-attn --no-build-isolation
```

## What It Measures

| Benchmark | Metric | Workload | Regime |
|-----------|--------|----------|--------|
| GEMV | Effective HBM BW (GB/s) | `[BS, H] × [H, I]` | Memory-bound (decode FFN) |
| GEMM | Effective TFLOPS | `[BS*seq, H] × [H, I]` | Compute-bound (prefill FFN) |
| FlashAttn prefill | Effective TFLOPS | `B=BS, S=seq_len, GQA` | Fused attention (prefill) |
| FlashAttn decode | Effective TFLOPS | `B=BS, Q=1, KV=kv_len, GQA` | Fused attention (decode w/ KV cache) |
| AllReduce | Effective Comm BW (GB/s) | `BS × H × dtype_bytes` | TP communication |

All benchmarks sweep batch sizes from 1, 2, 4, ..., `--max-batch`.  
`H` = hidden_dim, `I` = intermediate_dim (from `--hidden-dim`, `--intermediate-dim`).

## Usage

### Single GPU

```bash
python bench_hw.py --max-batch 256
```

### Multi-GPU (includes AllReduce)

```bash
torchrun --nproc_per_node=4 bench_hw.py --max-batch 256
```

### Custom Model Dimensions

```bash
# Llama-3.1-70B (default)
python bench_hw.py --hidden-dim 8192 --intermediate-dim 28672 --max-batch 256

# Qwen3-32B
python bench_hw.py --hidden-dim 5120 --intermediate-dim 25600 \
  --attn-heads 64 --attn-kv-heads 8 --attn-head-dim 128 --max-batch 256
```

### Save Results

```bash
torchrun --nproc_per_node=4 bench_hw.py --max-batch 256 --output results/g6_48xlarge.json
```

## Output Format

```json
{
  "gpu_name": "NVIDIA L4",
  "gpu_memory_gb": 22.5,
  "summary": {
    "effective_bw_GBs": 245.3,
    "effective_tflops": 98.2,
    "effective_comm_bw_GBs": 10.5
  },
  "benchmarks": { ... }
}
```

## Run on Each Instance Type

```bash
# g5.48xlarge (A10G)
torchrun --nproc_per_node=8 bench_hw.py --output results/g5_48xlarge.json

# g6.48xlarge (L4)
torchrun --nproc_per_node=8 bench_hw.py --output results/g6_48xlarge.json

# g6e.48xlarge (L40S)
torchrun --nproc_per_node=8 bench_hw.py --output results/g6e_48xlarge.json

# p4d.24xlarge (A100)
torchrun --nproc_per_node=8 bench_hw.py --output results/p4d_24xlarge.json

# p5.48xlarge (H100)
torchrun --nproc_per_node=8 bench_hw.py --output results/p5_48xlarge.json
```

## Results Directory

```
micro-benchmark/
├── bench_hw.py          # Benchmark script
├── README.md
└── results/             # Measured results (gitignored, regenerate per instance)
    ├── g5_48xlarge.json
    ├── g6_48xlarge.json
    └── ...
```
