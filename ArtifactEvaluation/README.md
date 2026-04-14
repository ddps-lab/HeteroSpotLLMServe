# Experiment Scripts

End-to-end benchmark scripts for ShuntServe. Each script configures a `GlobalServer` with one or more pipelines on a heterogeneous GPU cluster, replays Azure LLM traces (or sends synthetic requests), and collects throughput/latency metrics. Spot interruptions are **simulated** at the application level — all experiments run on on-demand instances.

## Experiment Matrix

| Experiment | Directory | Llama-3.1-70B | Qwen3-32B |
|---|---|---|---|
| Model Placement Optimizer | `ModelPlacement/optimizer/` | ✓ | ✓ |
| Offline Throughput | `ModelPlacement/offline/` | ✓ | ✓ |
| Online Serving | `ModelPlacement/online/` | ✓ | ✓ |
| Per-Pipeline Ranking | `ModelPlacement/per_pipeline/` | ✓ | ✓ |
| Module Initialization Timing | `ModelPlacement/check_module_time/` | ✓ | — |
| Beam-Search Top-k | `ModelPlacement/top_k_beam/` | ✓ | ✓ |
| Spot Interruption — Offline | `SpotTolerance/{model}/offline/scenario_A/` | ✓ | ✓ |
| Spot Interruption — Online | `SpotTolerance/{model}/online/scenario_A/` | ✓ | ✓ |
| Minimum Functional Test | `SpotTolerance/UnitTest8B/` | Llama-3.1-8B |  — |
| Performance Estimation | `PerformanceEstimation/` | ✓ | ✓ |

## Step 0: Environment Setup

See the project root [README.md](../README.md) for CUDA, NCCL, Python, and vLLM installation.

## Step 1: Cluster Setup

Example cluster used in our development setup:

| Instance Type | GPU | Count | Price |
|---|---|---|---|
| g5.12xlarge | 4× NVIDIA A10G | 2 | $2.29/hr |
| g6.12xlarge | 4× NVIDIA L4 | 3 | $1.94/hr |
| g6e.xlarge | 1× NVIDIA L40S | 4 | $0.70/hr |

Total: 9 instances, 24 GPUs, 672 GB GPU memory.

`SpotTolerance` experiments additionally need replacement instances (simulating on-demand fallback when spot is interrupted); counts are listed in [`nodes_scenario_A.json`](SpotTolerance/nodes_scenario_A.json).

The supported GPU/instance types are enumerated in [`ModelPlacement/hardware_specs.py`](../ModelPlacement/hardware_specs.py). To target a different cluster, edit that file and re-run the optimizer (Step 4).

## Step 2: Prepare Model Weights

Model weights are served from S3 via the [TensorStore](../TensorStore/README.md) module. TensorStore downloads, pre-partitions for TP=1/2/4/8, converts to TRAW binary format, and uploads to S3.

1. Request HuggingFace access for any gated models you plan to use:
   - [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
   - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

2. Create an S3 bucket and upload the weights using `TensorStore/upload_model.sh`. This does not require a GPU:
   ```bash
   cd TensorStore
   # Edit upload_model.sh:
   #   BUCKET_NAME="s3://<YOUR_S3_BUCKET>"
   #   MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
   bash upload_model.sh
   ```
   Repeat for each model you intend to benchmark.

3. Ensure all EC2 instances have S3 read access (IAM instance profile, `~/.aws/credentials`, or env variables — resolved through the standard boto3 chain).

## Step 3: Prepare Dataset

A preprocessed [Azure LLM Inference Conversation Dataset (2023)](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) is included at `Datasets/AzureLLMInferenceConvTrace_pruned_2048.csv` and loaded automatically. No additional preparation needed.

## Step 4: Run the Model Placement Optimizer

The optimizer determines how to partition transformer layers across a heterogeneous cluster. Run it once per (baseline × model):

```bash
cd ArtifactEvaluation/ModelPlacement/optimizer/llama3-70b
python shuntserve.py            # ShuntServe beam-search DP
python hexgen.py                # HEXGEN genetic algorithm
python alpaserve.py             # AlpaServe homogeneous DP
python vllm.py                  # Single-pipeline vLLM baseline
```

Replace `llama3-70b` with `qwen3-32b` for Qwen3. Results are written to `optimizer/results/<model>/{estimated,measured}/predicted_<baseline>_<ModelName>.json`, containing per-pipeline `pp_layer_partition`, `parallel_strategy`, `num_gpu_blocks`, `max_batch_size`, and estimated throughput.

See [`ModelPlacement/README.md`](../ModelPlacement/README.md) for algorithm details.

## Step 5: Configure Node IPs

### ModelPlacement experiments

`ModelPlacement/nodes.py` is shared across all `offline/`, `online/`, `per_pipeline/`, and `check_module_time/` scripts. Fill in the private IPs of your EC2 instances:

```python
g6_12xlarge_node_ip_1 = ""   # 4× L4
g6_12xlarge_node_ip_2 = ""
g6_12xlarge_node_ip_3 = ""
g5_12xlarge_node_ip_1 = ""   # 4× A10G
g5_12xlarge_node_ip_2 = ""
g6e_xlarge_node_ip_1  = ""   # 1× L40S
g6e_xlarge_node_ip_2  = ""
g6e_xlarge_node_ip_3  = ""
g6e_xlarge_node_ip_4  = ""
```

### SpotTolerance experiments

Edit [`SpotTolerance/nodes_scenario_A.json`](SpotTolerance/nodes_scenario_A.json) with the initial (spot-simulated) and replacement (on-demand) instance IPs. `spot_*` and `on_demand_*` prefixes distinguish the two roles.

Then generate pipeline configs from the optimizer results (Step 4):

```bash
cd SpotTolerance
python generate_pipelines.py --model all
```

This writes `pipelines_{model}_scenario_A.json` files used by the offline/online scripts.

### UnitTest8B

[`SpotTolerance/UnitTest8B/nodes.json`](SpotTolerance/UnitTest8B/nodes.json) holds the 3 initial + 2 replacement `g6.xlarge` IPs.

## Step 6: Run Experiments

All experiment scripts accept the same pipeline configuration structure. See [Appendix A](#appendix-a-pipeline-configuration-reference) for field definitions.

### 6.1 Offline Throughput

**Paths:** `ModelPlacement/offline/{llama3-70b,qwen3-32b}/`

All requests submitted at once (`time_scale=0.0`) to measure maximum sustained throughput.

| Baseline | Command |
|---|---|
| ShuntServe | `python shuntserve.py` |
| HEXGEN | `python hexgen.py` |
| AlpaServe | `python alpaserve.py` |
| vLLM | `python vllm.py` |

### 6.2 Online Serving

**Paths:** `ModelPlacement/online/{llama3-70b,qwen3-32b}/`

Trace replay with `time_scale=5.0` (inter-arrival times stretched 5×). Each baseline has a separate warmup script:

| Baseline | Main | Warmup |
|---|---|---|
| ShuntServe | `shuntserve.py` | `warmup_shuntserve.py` |
| HEXGEN | `hexgen.py` | `warmup_hexgen.py` |
| AlpaServe | `alpaserve.py` | `warmup_alpaserve.py` |
| vLLM | `vllm.py` | `warmup_vllm.py` |

### 6.3 Per-Pipeline Ranking Evaluation

**Paths:** `ModelPlacement/per_pipeline/{llama3-70b,qwen3-32b}/{shuntserve,hexgen,alpaserve,vllm}/`

Evaluates the ranking accuracy of ShuntServe's profiling-free estimator. Each pipeline is benchmarked independently using synthetic fixed-length requests (input=763, output=232 tokens). Each baseline subdirectory holds one script per pipeline: `p1.py`, `p2.py`, …

```bash
cd ModelPlacement/per_pipeline/llama3-70b/shuntserve
python p1.py
python p2.py
```

`example/p1.py` is a template for adding new pipelines.

### 6.4 Spot Interruption — Offline

**Paths:** `SpotTolerance/{llama3-70b,qwen3-32b}/offline/scenario_A/`

The interruption/restore timeline is declared in [`SpotTolerance/spot_trace_events_scenario_A.json`](SpotTolerance/spot_trace_events_scenario_A.json). Use `show_events.py` in each scenario directory to print a human-readable summary. All trace requests are submitted at once (`time_scale=0`).

| Strategy | Script | `request_handler_mode` | Interruption Handling |
|---|---|---|---|
| ShuntServe | `shuntserve.py` | `"migration"` | `switch_nodes()` + request migration + concurrent init |
| Request Migration | `request_migration.py` | `"migration"` | `switch_nodes()` without concurrent init |
| Concurrent Init | `concurrent_initialization.py` | `"re-routing"` | `switch_nodes()` without request migration |
| No Handle | `no_handle.py` | `"re-routing"` | Stop pipeline, wait, recreate |
| No Interruption | `only_ondemand.py` | default | Baseline without events |
| Warmup | `warmup.py` | default | Pre-benchmark warmup |

### 6.5 Spot Interruption — Online

**Paths:** `SpotTolerance/{llama3-70b,qwen3-32b}/online/scenario_A/`

Same strategies as 6.4, but the trace is replayed with `time_scale=3.0` (inter-arrival times stretched 3×). The event timeline is shared with the offline variant.

### 6.6 UnitTest8B — Minimum Functional Test

**Path:** `SpotTolerance/UnitTest8B/`

A small-scale sanity test on 3× `g6.xlarge` (single L4 each) using Llama-3.1-8B-Instruct. Intended to verify that interruption handling mechanisms (stop-and-start, concurrent initialization, request migration) are operational without provisioning the full 70B cluster.

Config files:
- `pipelines_8b.json` — pipeline definitions
- `nodes.json` — initial + replacement IPs
- `spot_trace_events.json` — interruption event timeline

Scripts (all share the naming scheme of the 70B experiments):

| Strategy | Script | `request_handler_mode` |
|---|---|---|
| ShuntServe | `shuntserve.py` | `"migration"` |
| Request Migration | `request_migration.py` | `"migration"` |
| Concurrent Init | `concurrent_initialization.py` | `"re-routing"` |
| No Handle | `no_handle.py` | `"re-routing"` |
| No Interruption | `only_ondemand.py` | default |

Unlike the 70B scripts (which use `switch_nodes()` for in-place migration), the `no_handle` path here uses `stop_nodes()` + `create_pipeline()` to simulate a full pipeline recreation.

## Step 7: Collect and Verify Results

### Trace CSV

Each run saves a per-request trace CSV:

```
ArtifactEvaluation/Trace/{trace_output_prefix}_{YYYYMMDD_HHMM}.csv
```

Columns: RequestID, ArrivalTime, CompletionTime, InputTokens, OutputTokens, Latency, TTFT, TPOT, Success.

### Console Summary

Each benchmark prints a summary including request throughput, per-token throughput, end-to-end latency, TTFT, TPOT, and ITL with P10/P25/P50/P75/P90/P99 percentiles.

### Reference Data and Figures

[`ReferenceData/`](ReferenceData/) contains reference results and the figure-generating notebooks, organized per experiment module:

```
ReferenceData/
  Datasets/figures/                      # Dataset distribution figures
  ModelPlacement/
    offline/{llama3-70b,qwen3-32b}/      # Raw traces & logs
    offline/figures/                     # Offline throughput figures
    online/{llama3-70b,qwen3-32b}/
    online/figures/
    per_pipeline/{llama3-70b,qwen3-32b}/
    per_pipeline/figures/
    check_module_time/figures/
    top_k_beam/figures/
  SpotTolerance/
    llama3-70b/{offline,online}/         # Scenario A raw traces
    llama3-70b/figures/                  # Per-model aggregate figures
    qwen3-32b/{offline,online}/
    qwen3-32b/figures/
    UnitTest8B/
    UnitTest8B/figures/
    figures/                             # Cross-cutting figures (cost, legend)
  PerformanceEstimation/figures/
  SpotAvailabilityTrace/figures/
  ConcurrentInitialization/
  MigrationComparison/figures/
```

Each `figures/` directory ships a short README describing its notebooks and outputs.

## Notes

1. **HEXGEN mode**: HEXGEN scripts set `"mode": "hexgen"` in their pipeline config, enabling stage splitting where a single physical instance can host multiple pipeline stages.
2. **Request handler modes**: `GlobalServer(request_handler_mode=...)` accepts
   - `"migration"` — active request migration during node switch (continues in-flight requests on new nodes)
   - `"re-routing"` — re-routes failed requests to surviving pipelines (restarts from scratch)
   - default — standard round-robin with no interruption handling.
3. **Trace output location**: All trace CSVs go to `ArtifactEvaluation/Trace/` (created automatically).

## Appendix A: Pipeline Configuration Reference

| Field | Type | Description | Example |
|---|---|---|---|
| `model_name` | str | HuggingFace model identifier | `"meta-llama/Llama-3.1-70B-Instruct"` |
| `total_num_layers` | int | Total transformer layers | 80 (70B), 64 (32B), 32 (8B) |
| `pp_layer_partition` | str | Layers per pipeline stage | `"20,20,20,10,10"` |
| `parallel_strategy` | list[int] | TP degree per stage | `[4,4,4,1,1]` |
| `gpu_memory_utilization` | float | Fraction of GPU memory to allocate | `0.85` |
| `max_model_len` | int | Maximum sequence length (tokens) | `8192` |
| `max_num_batched_tokens` | int | Maximum tokens per batch | `8192` |
| `max_num_seqs` | int | Maximum concurrent sequences | `512` |
| `model_source` | str | Weight loading source (only `"s3"` supported) | `"s3"` |
| `s3_path` | str | S3 URI for model weights | `"s3://<bucket>/..."` |
| `num_gpu_blocks` | int | Available KV cache blocks | `27549` |
| `max_batch_size` | int | Maximum batch size for scheduling | `442` |
| `mode` | str | Optional; `"hexgen"` for HEXGEN baselines | `"hexgen"` |

`pp_layer_partition`, `parallel_strategy`, `num_gpu_blocks`, and `max_batch_size` are outputs of the Model Placement optimizer (Step 4). Re-run the optimizer whenever the cluster changes.

## Appendix B: Trace Parameters Reference

Trace-based experiments (`run_trace_benchmark`) accept:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `time_scale` | float | `1.0` | Inter-arrival multiplier. `0.0` = offline, `1.0` = real-time, `3.0` = 3× slower |
| `start_time` | float | `None` | Start time filter (seconds from first request) |
| `end_time` | float | `None` | End time filter (seconds from first request) |
| `num_requests` | int | `None` | Cap on requests loaded (`None` = full trace) |
| `run_initial_test` | bool | `True` | Send test requests before the benchmark |
| `test_requests_per_pipeline` | int | `2` | Number of test requests per pipeline |

The trace is `Datasets/AzureLLMInferenceConvTrace_pruned_2048.csv`. Each row is (timestamp, input tokens, output tokens).
