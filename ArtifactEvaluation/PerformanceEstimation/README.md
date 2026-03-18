# Performance Estimation

Estimates and benchmarks model serving performance across instance types and parallelization strategies.

## Quick Start

```bash
# 1. Estimate performance (analytical model)
./run.sh estimate

# 2. Generate benchmark scripts
./run.sh generate

# 3. Set worker IPs
vim nodes.py

# 4. Run a benchmark
./run.sh bench llama3-70b g6_48xlarge tp8_pp1
```

Or run steps 1+2 together:

```bash
./run.sh all
```

## Files

| File | Description |
|---|---|
| `run.sh` | Entry point — estimate, generate, bench |
| `estimate.py` | Analytical estimator (batch sweep, pipeline bubble correction) |
| `generate_p_files.py` | Generates benchmark Python scripts from estimation JSONs |
| `nodes.py` | Worker IP addresses (edit before benchmarking) |
| `{model}/results_viewer.ipynb` | Jupyter notebook to view estimation results |

## Directory Structure

```
{model}/
├── results_viewer.ipynb            # Set WORKLOAD variable to select
└── in{input}-out{output}/          # Workload configuration (gitignored)
    ├── {instance_dir}/
    │   └── {strategy}.py           # Benchmark script
    └── results/data/
        ├── estimated/est_*.json    # Estimation results (cached)
        └── measured/bench_*.json   # Benchmark results
```

## Models

- `llama3-70b` — Meta Llama 3.1 70B Instruct (80 layers)
- `qwen3-32b` — Qwen3 32B (64 layers)

## Instances

| Instance | GPU | Count | VRAM |
|---|---|---|---|
| g5.48xlarge | A10G | 8 | 24GB |
| g6.48xlarge | L4 | 8 | 24GB |
| g6e.48xlarge | L40S | 8 | 48GB |
| p4d.24xlarge | A100 40GB | 8 | 40GB |
| p5.48xlarge | H100 | 8 | 80GB |

## Strategies

| Label | TP | PP | Description |
|---|---|---|---|
| tp1_pp8 | 1 | 8 | Max pipeline parallelism |
| tp2_pp4 | 2 | 4 | Balanced |
| tp4_pp2 | 4 | 2 | Balanced |
| tp8_pp1 | 8 | 1 | Max tensor parallelism |

## Workload

Default: `input_len=763, output_len=232` (Azure LLM trace average).
Change `WORKLOAD` in `estimate.py` and re-run to generate a new `in{N}-out{N}/` directory.

## Parallel Benchmarks

Multiple benchmarks can run simultaneously on one head node if each targets a different worker node. Ensure Ray head is running and each worker has the latest code (`git pull`).
