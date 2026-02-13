# Model Placement Optimizer

This module determines optimal model placement (pipeline parallelism configuration) for serving LLMs on heterogeneous GPU clusters. Given a cluster of spot instances, it decides how to partition model layers across instances to maximize serving throughput.

## Quick Start

```bash
cd ModelPlacement
python main.py --baseline shuntserve
python main.py --baseline hexgen alpaserve shuntserve
```

### Requirements

```bash
pip install -r requirements-for-baselines.txt
```

## Baselines

Three model placement algorithms are implemented.

### ShuntServe

Our approach. Uses dynamic programming with beam search to jointly optimize node configuration, parallelization strategy, and layer assignment on heterogeneous spot GPU clusters.

- **Algorithm**: Beam search DP (`BeamSearchDPOptimizer`)
- **Objective**: Maximize throughput per cost with soft SLO penalty (Eq. 7 in paper)
- **Key feature**: Iteratively generates pipelines until no more feasible placements exist, deducting used instances after each iteration
- **Output**: One or more `Pipeline` objects with throughput, cost, latency, and layer assignment

### HEXGEN

A state-of-the-art heterogeneous GPU serving system that uses a genetic algorithm for model placement.

- **Algorithm**: DEAP-based genetic algorithm (`run_hexgen_ga`)
- **Objective**: Minimize pipeline execution cost via GA with mutation and selection
- **Key feature**: Allows a single physical instance to host multiple pipeline stages (e.g., splitting a 4-GPU instance into two 2-GPU stages). Since this is not representable in our pipeline abstraction, we map these splits to virtual instance types (`g5.12xlarge(half)`, `g6.xlarge`, etc.) defined in `hardware_specs.py`.
- **Post-processing**: HEXGEN's layer partitioning is re-optimized using ShuntServe's DP optimizer (`optimization_mode="only_throughput"`) to obtain accurate throughput estimates under our profiling-free throughput model.
- **Output**: Per-pipeline GPU usage, TP degree plan, re-optimized PP partition, and throughput metrics

### AlpaServe

A state-of-the-art homogeneous GPU serving system that uses dynamic programming to equalize stage latencies.

- **Algorithm**: Recursive DP with memoization (`AlpaServeOptimizer`)
- **Objective**: Minimize the maximum per-stage latency (balanced pipeline partitioning)
- **Key feature**: Homogeneous baseline -- runs independently for each instance type in the cluster. Each instance type forms its own pipeline where all stages use the same GPU.
- **Output**: One `Pipeline` per instance type with layer assignment, throughput, and cost

## Example Output

With the default cluster configuration, ShuntServe produces two heterogeneous pipelines:

```
================================================================================
ShuntServe: Found 2 pipeline(s)
================================================================================

Pipeline 1:
  Pipeline(stages=[(spot)g6.12xlarge:20L, (spot)g6.12xlarge:21L, (spot)g6.12xlarge:21L, (spot)g6e.xlarge:10L, (spot)g6e.xlarge:8L],
	throughput=4.096,
	cost=$7.241,
	latency_per_global_batch=112791ms,
	single_request_latency=30803ms,
	num_blocks=28730,
	global_batch_size=461)
  - Throughput: 4.10 req/s
  - Cost: $7.24 USD/h

Pipeline 2:
  Pipeline(stages=[(spot)g6e.xlarge:13L, (spot)g5.12xlarge:27L, (spot)g5.12xlarge:28L, (spot)g6e.xlarge:12L],
	throughput=2.898,
	cost=$5.991,
	latency_per_global_batch=83501ms,
	single_request_latency=22444ms,
	num_blocks=15049,
	global_batch_size=241)
  - Throughput: 2.90 req/s
  - Cost: $5.99 USD/h
```

### Reading the Output

Each pipeline represents an independent model replica serving requests in parallel. The `stages` field shows the pipeline parallelism configuration:

- **`(spot)g6.12xlarge:20L`** means a spot `g6.12xlarge` instance (4x L4 GPUs with TP=4) hosting 20 transformer layers
- The layers across all stages sum to the total model layers (80 for Llama-3.1-70B)

Key metrics per pipeline:
- **throughput**: Sustained serving throughput in requests per second
- **cost**: Total hourly cost of all instances in the pipeline
- **single_request_latency**: End-to-end latency for a single request (no batching)
- **global_batch_size**: Maximum concurrent requests the pipeline can process
- **num_blocks**: Number of KV cache blocks available across all stages (determines max concurrent sequences)

In this example, Pipeline 1 uses all 3 `g6.12xlarge` and 2 of 4 `g6e.xlarge` instances. Pipeline 2 then uses the remaining 2 `g5.12xlarge` and 2 `g6e.xlarge` instances, yielding a combined throughput of ~7.0 req/s across the full cluster.

## Cluster Configuration

The default evaluation cluster matches the paper (Section VII):

| Instance Type | GPU | Count | Spot Price |
|---|---|---|---|
| g5.12xlarge | A10G (x4) | 2 | $2.29/hr |
| g6.12xlarge | L4 (x4) | 3 | $1.94/hr |
| g6e.xlarge | L40S (x1) | 4 | $0.70/hr |

Total: 24 GPUs, 672 GB GPU memory.

## Model Configuration

Default model: `meta-llama/Llama-3.1-70B-Instruct` (FP16)

Workload parameters are derived from the Azure Conversation Dataset trace:
- Average input length: 763 tokens
- Average output length: 232 tokens
- Max model length: 8192 tokens

## Module Structure

```
ModelPlacement/
  main.py                   # Unified CLI entry point
  shuntserve_optimizer.py   # ShuntServe beam search DP optimizer + Pipeline class
  hexgen_optimizer.py       # HEXGEN genetic algorithm optimizer
  alpaserve_optimizer.py    # AlpaServe DP layer partitioning
  estimator_utils.py        # Profiling-free throughput/latency estimation (roofline model)
  hardware_specs.py         # GPU, interconnect, and instance specifications
  cluster_pool.py           # Cluster resource and pricing management
  hexgen/                   # HEXGEN internal modules (cost model, simulator, plan generation)
```

## Throughput Estimation

All baselines share the same profiling-free serving performance estimator (`estimator_utils.py`), which uses a roofline model to predict throughput without requiring actual GPU profiling. The estimator computes:

1. **Per-layer computation latency** for both prefill and decode phases, considering whether each operation is compute-bound or memory-bound
2. **Communication latency** for tensor parallelism (AllReduce) and pipeline parallelism (point-to-point)
3. **Maximum batch size** given GPU memory constraints (model weights + KV cache + activations)
4. **End-to-end throughput** as `batch_size / total_latency`
