# Hardware Micro-Benchmark

Measures **effective** HBM bandwidth, TFLOPS, and inter-GPU communication bandwidth
using saturated workloads. These values replace peak spec-sheet numbers in the
roofline estimator for more accurate cross-GPU performance predictions.

## What It Measures

| Benchmark | Metric | Workload | Regime |
|-----------|--------|----------|--------|
| GEMV | Effective HBM BW (GB/s) | `[1, 8192] × [8192, 28672]` | Memory-bound (decode FFN) |
| GEMM | Effective TFLOPS | `[2048, 8192] × [8192, 28672]` | Compute-bound (prefill FFN) |
| AllReduce | Effective Comm BW (GB/s) | 64MB all-reduce across GPUs | TP communication |

Each benchmark also sweeps multiple sizes to verify saturation stability.

## Usage

### Single GPU (BW + FLOPS only)

```bash
python bench_hw.py --dtype bfloat16
```

### Multi-GPU (includes AllReduce)

```bash
torchrun --nproc_per_node=8 bench_hw.py --dtype bfloat16
```

### Save Results

```bash
# Single GPU
python bench_hw.py --output results/g6_48xlarge.json

# Multi-GPU
torchrun --nproc_per_node=8 bench_hw.py --output results/g6_48xlarge.json
```

### Custom Parameters

```bash
python bench_hw.py \
  --warmup 20 \
  --repeat 100 \
  --dtype bfloat16 \
  --gemv-k 8192 --gemv-n 28672 \
  --gemm-m 4096 --gemm-k 8192 --gemm-n 28672 \
  --ar-size-mb 128
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
