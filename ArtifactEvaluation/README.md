# Artifact Evaluation

End-to-end evaluation scripts for the ShuntServe paper. Each script configures a GlobalServer with one or more pipelines on a heterogeneous GPU cluster, replays Azure traces (or sends synthetic requests), and collects throughput/latency metrics. Results map directly to figures in the paper. Spot interruptions are **simulated** -- all experiments run on on-demand instances.

> **Note for reviewers:** If the full cluster setup (9 instances, 24 GPUs) is too costly, a simplified 8B test setup is available to verify basic executability. See [Appendix C](#appendix-c-simplified-8b-test-setup).

| Paper Figure | Experiment | Directory | Scripts |
|---|---|---|---|
| Figure 7 | Offline Throughput | `ModelPlacement/offline/` | `shuntserve.py`, `hexgen.py`, `alpaserve.py`, `vllm.py` |
| Figure 8 | Online Serving | `ModelPlacement/online/` | `shuntserve.py`, `hexgen.py`, `alpaserve.py`, `vllm.py`, `warmup.py` |
| Figures 9, 11 | Spot Interruption -- Offline (70B) | `SpotInterruption/offline/` | `shuntserve.py`, `request_migration.py`, `no_handle.py`, `only_ondemand.py`, `concurrent_initialization.py`, `warmup.py` |
| Figures 9, 11 | Spot Interruption -- Online (70B) | `SpotInterruption/online/` | `shuntserve.py`, `request_migration.py`, `no_handle.py`, `only_ondemand.py`, `concurrent_initialization.py`, `warmup.py` |

## Step 0: Environment Setup

See the project root [README.md](../README.md) for environment setup instructions (CUDA, NCCL, Python, vLLM installation).

## Step 1: Cluster Setup

### Full Cluster (Figures 7-11)

The default evaluation cluster matches Section VII of the paper:

| Instance Type | GPU | Count | Price |
|---|---|---|---|
| g5.12xlarge | 4x NVIDIA A10G | 2 | $2.29/hr |
| g6.12xlarge | 4x NVIDIA L4 | 3 | $1.94/hr |
| g6e.xlarge | 1x NVIDIA L40S | 4 | $0.70/hr |

Total: 9 instances, 24 GPUs, 672 GB GPU memory.

For **SpotInterruption** experiments (Figures 9-11), additional instances are required as replacement nodes (simulating on-demand fallback):

| Instance Type | Count | Purpose |
|---|---|---|
| g6.12xlarge | 2 | Replacement for interrupted g6.12xlarge |
| g5.12xlarge | 2 | Replacement for interrupted g5.12xlarge |
| g6e.xlarge | 4 | Replacement for interrupted g6e.xlarge |

### Simplified Test Setup

The **SpotInterruption/8B** experiments provide a simplified test setup to verify basic executability when full replication with the 70B model cluster is costly or infeasible. This uses a smaller cluster with Llama-3.1-8B-Instruct:

| Instance Type | Initial Count | Replacement Count |
|---|---|---|
| g6.xlarge (1x L4) | 3 | 2 |

This setup validates the core interruption handling mechanisms (stop-and-start, concurrent initialization) at reduced cost.

## Step 2: Prepare Model Weights

Model weights are loaded from S3 using our custom **TensorStore** module. TensorStore pre-partitions model tensors for each tensor parallelism (TP) degree and stores them in a compact binary format (TRAW), which is ~50% smaller and ~50% faster to load compared to `torch.save()`. See `TensorStore/README.md` for format details.

1. Request access to the Llama models on HuggingFace:
   - [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
   - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (for simplified test setup only)

2. Create an S3 bucket and upload the model weights using `TensorStore/upload_model.sh`. This step does not require a GPU -- it can be run on any instance with sufficient CPU memory and network bandwidth. Edit the script to set your bucket name and model, then run it. It downloads the model from HuggingFace, partitions tensors for TP sizes 1/2/4/8, converts to TRAW format, and uploads to S3:
   ```bash
   # From the project root
   cd TensorStore
   # Edit upload_model.sh:
   #   BUCKET_NAME="s3://<YOUR_S3_BUCKET>"
   #   MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
   bash upload_model.sh
   ```
   The weights are stored at `s3://<YOUR_S3_BUCKET>/meta-llama/Llama-3.1-70B-Instruct/` (the model name is appended automatically to the bucket path). Repeat with `MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"` for the simplified test setup.

3. Ensure all EC2 instances have AWS credentials to access your S3 bucket. The TensorStore server resolves credentials via the standard boto3 chain (IAM instance profile, `~/.aws/credentials`, or environment variables).

## Step 3: Prepare Dataset

Experiments use the [Azure LLM Inference Conversation Dataset (2023)](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md).

A preprocessed trace (`Datasets/AzureLLMInferenceConvTrace_pruned_2048.csv`) is included in the repository and loaded automatically by the experiment scripts. No additional dataset preparation is needed.

## Step 4: Run Model Placement Optimizer

The Model Placement optimizer determines how to partition model layers across the heterogeneous cluster. Run all three baselines to compare their placement strategies -- ShuntServe's beam search DP produces superior throughput over HEXGEN and AlpaServe on heterogeneous clusters (Section VI in the paper).

```bash
# From the project root
cd ArtifactEvaluation
python model_placement_optimizer.py --baseline shuntserve
python model_placement_optimizer.py --baseline hexgen alpaserve shuntserve
```

The optimizer outputs `pp_layer_partition`, `parallel_strategy`, `num_gpu_blocks`, `max_batch_size`, and `estimated_throughput` for each pipeline. The experiment scripts already include pre-computed configurations from the default cluster (Step 1). If you change the cluster, re-run the optimizer and update the experiment scripts accordingly.

Three baselines are supported: `shuntserve` (beam search DP), `hexgen` (genetic algorithm), and `alpaserve` (homogeneous DP). See `ModelPlacement/README.md` for detailed algorithm descriptions and output format.

## Step 5: Configure Node IPs

Each experiment subdirectory contains a `nodes.py` file with placeholder IP addresses. Fill in the private IPs of your EC2 instances before running. The pipeline configurations (from Step 4) determine which instance types host which layers.

### ModelPlacement experiments

`ModelPlacement/offline/nodes.py`, `online/nodes.py`, and `check_module_time/nodes.py` all share the same structure:

```python
g6_12xlarge_node_ip_1 = ""   # 4x L4
g6_12xlarge_node_ip_2 = ""   # 4x L4
g6_12xlarge_node_ip_3 = ""   # 4x L4
g5_12xlarge_node_ip_1 = ""   # 4x A10G
g5_12xlarge_node_ip_2 = ""   # 4x A10G
g6e_xlarge_node_ip_1  = ""   # 1x L40S
g6e_xlarge_node_ip_2  = ""   # 1x L40S
g6e_xlarge_node_ip_3  = ""   # 1x L40S
g6e_xlarge_node_ip_4  = ""   # 1x L40S
```

### SpotInterruption experiments

`SpotInterruption/offline/nodes.py` and `SpotInterruption/online/nodes.py` have both initial and replacement instance IPs (identical structure). Variable names use `spot_` and `on_demand_` prefixes to match the simulated roles:

```python
# Initial instances (simulating spot instances)
spot_g6_12xlarge_node_ip_1 = ""
spot_g6_12xlarge_node_ip_2 = ""
spot_g6_12xlarge_node_ip_3 = ""
spot_g5_12xlarge_node_ip_1 = ""
spot_g5_12xlarge_node_ip_2 = ""
spot_g6e_xlarge_node_ip_1  = ""
spot_g6e_xlarge_node_ip_2  = ""
spot_g6e_xlarge_node_ip_3  = ""
spot_g6e_xlarge_node_ip_4  = ""

# Replacement instances (simulating on-demand fallback)
# Note: on_demand_g6_12xlarge_node_ip_1 is omitted because
# spot_g6_12xlarge_node_ip_1 is consistently available in the experiment trace.
on_demand_g6_12xlarge_node_ip_2 = ""
on_demand_g6_12xlarge_node_ip_3 = ""
on_demand_g5_12xlarge_node_ip_1 = ""
on_demand_g5_12xlarge_node_ip_2 = ""
on_demand_g6e_xlarge_node_ip_1  = ""
on_demand_g6e_xlarge_node_ip_2  = ""
on_demand_g6e_xlarge_node_ip_3  = ""
on_demand_g6e_xlarge_node_ip_4  = ""
```

### SpotInterruption/8B experiments

`SpotInterruption/8B/nodes_8B.py`:

```python
spot_g6_xlarge_node_ip_1 = ""
spot_g6_xlarge_node_ip_2 = ""
spot_g6_xlarge_node_ip_3 = ""

ondemand_g6_xlarge_node_ip_1 = ""
ondemand_g6_xlarge_node_ip_2 = ""
```

## Step 6: Run Experiments

All experiment scripts share the same pipeline configuration structure. See [Appendix A](#appendix-a-pipeline-configuration-reference) for field descriptions. Example (ShuntServe):

```python
# Pipeline 1: 3x g6.12xlarge + 2x g6e.xlarge
pipeline_1_config = {
    "model_name": "meta-llama/Llama-3.1-70B-Instruct",
    "total_num_layers": 80,
    "pp_layer_partition": "20,20,20,10,10",   # layers per pipeline stage
    "parallel_strategy": [4,4,4,1,1],          # tensor parallelism per stage
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
    "model_source": "s3",
    "s3_path": "s3://<YOUR_S3_BUCKET>/meta-llama/Llama-3.1-70B-Instruct",
    "num_gpu_blocks": 27549,
    "max_batch_size": 442,
}
estimated_throughput_1 = 4.23  # req/s (from ModelPlacement optimizer)

node_layer_mapping_1 = [
    (g6_12xlarge_node_ip_1, 20),   # Stage 0: 20 layers
    (g6_12xlarge_node_ip_2, 20),   # Stage 1: 20 layers
    (g6_12xlarge_node_ip_3, 20),   # Stage 2: 20 layers
    (g6e_xlarge_node_ip_1, 10),    # Stage 3: 10 layers
    (g6e_xlarge_node_ip_2, 10),    # Stage 4: 10 layers
]
```

### 6.1 Offline Throughput (Figure 7)

**Path:** `ModelPlacement/offline/`

The trace is replayed at once (`time_scale=0.0`) to measure maximum throughput under each model placement strategy.

| Baseline | Command | Description |
|---|---|---|
| ShuntServe | `python shuntserve.py` | Heterogeneous pipelines via beam search DP |
| HEXGEN | `python hexgen.py` | Genetic algorithm placement |
| AlpaServe | `python alpaserve.py` | Homogeneous per-instance-type pipelines |
| vLLM | `python vllm.py` | Homogeneous balanced partitioning |

Key parameters:
- `time_scale=0.0` -- all requests sent at once (offline mode)
- `run_initial_test=True` -- sends 2 test requests per pipeline to verify connectivity before benchmark
- `num_requests=None` -- uses the full trace

### 6.2 Online Serving (Figure 8)

**Path:** `ModelPlacement/online/`

Requests from the first 3 minutes of the trace are replayed with `time_scale=5.0` (inter-arrival times stretched 5x). Measures throughput and latency under realistic arrival patterns.

| Baseline | Command |
|---|---|
| ShuntServe | `python shuntserve.py` |
| HEXGEN | `python hexgen.py` |
| AlpaServe | `python alpaserve.py` |
| vLLM | `python vllm.py` |
| Warmup | `python warmup.py` |

Key parameters:
- `time_scale=5.0` -- inter-arrival times stretched 5x (3-minute trace window replayed over ~15 minutes)
- `start_time=0`, `end_time=180` -- first 3 minutes of the trace
- `run_initial_test=False` -- skip test requests

Note: `warmup.py` uses different parameters (`time_scale=4.0`, `end_time=15`) for a short warmup measurement.

### 6.3 Spot Interruption -- Offline (Figures 9, 11)

**Path:** `SpotInterruption/offline/`

Evaluates how the system handles simulated spot interruptions and recoveries during offline serving of Llama-3.1-70B. Timed events simulate interruptions by switching nodes to replacement instances and vice versa. All trace requests are submitted at once (`time_scale=0`).

| Strategy | Script | `request_handler_mode` | Interruption Handling |
|---|---|---|---|
| ShuntServe | `shuntserve.py` | `"migration"` | `switch_nodes()` with request migration |
| Request Migration | `request_migration.py` | `"migration"` | `switch_nodes()` without concurrent init |
| No Handle | `no_handle.py` | `"re-routing"` | Stop pipeline, wait, recreate |
| No Interruption | `only_ondemand.py` | default | No events (baseline without interruptions) |
| Concurrent Init | `concurrent_initialization.py` | `"re-routing"` | `switch_nodes()` without request migration |
| Warmup | `warmup.py` | default | No events (baseline with warmup) |

Simulated interruption event timeline (from `shuntserve.py`):

| Event | Time | Description |
|---|---|---|
| 1 | 5 min | 1x g6.12xlarge + 2x g5.12xlarge interrupted |
| 2 | 15 min | 2x g6.12xlarge recovered, 4x g6e.xlarge interrupted |
| 3 | 25 min | 1x g5.12xlarge recovered |
| 4 | 35 min | 4x g6e.xlarge recovered |
| 5 | 45 min | 1x g5.12xlarge recovered |

Benchmark window: `start_time=0`, `end_time=1200` (first 20 min of the trace), `time_scale=0` (offline).

### 6.4 Spot Interruption -- Online (Figures 9, 11)

**Path:** `SpotInterruption/online/`

Same simulated interruption scenarios as 6.3, but with online trace replay. The first 20 minutes of the trace are replayed with `time_scale=3.0` (inter-arrival times stretched 3x), so the trace is delivered over approximately 60 minutes.

| Strategy | Script | `request_handler_mode` | Interruption Handling |
|---|---|---|---|
| ShuntServe | `shuntserve.py` | `"migration"` | `switch_nodes()` with request migration |
| Request Migration | `request_migration.py` | `"migration"` | `switch_nodes()` without concurrent init |
| No Handle | `no_handle.py` | `"re-routing"` | Stop pipeline, wait, recreate |
| No Interruption | `only_ondemand.py` | default | No events (baseline without interruptions) |
| Concurrent Init | `concurrent_initialization.py` | `"re-routing"` | `switch_nodes()` without request migration |
| Warmup | `warmup.py` | default | No events (baseline with warmup) |

Benchmark window: `start_time=0`, `end_time=1200` (first 20 min of the trace), `time_scale=3.0` (online). The event timeline is the same as 6.3.

## Step 7: Collect and Verify Results

### Trace CSV

Each benchmark saves a detailed per-request trace CSV to `ArtifactEvaluation/Trace/`:

```
ArtifactEvaluation/Trace/{trace_output_prefix}_{YYYYMMDD_HHMM}.csv
```

For example: `ArtifactEvaluation/Trace/modelplacement_offline_shuntserve_20250215_1430.csv`

The CSV contains: RequestID, ArrivalTime, CompletionTime, InputTokens, OutputTokens, Latency, TTFT, TPOT, Success.

### Console Output

Each benchmark prints a summary to the console:

```
==================================================
            Serving Benchmark Result
==================================================
Successful requests:                     xxxxx
Benchmark duration (s):                  xxx.xx
Total input tokens:                      xxxxxx
Total generated tokens:                  xxxxxx
Request throughput (req/s):              x.xx
Output token throughput (tok/s):         xxx.xx
Total Token throughput (tok/s):          xxxx.xx
----------------End-to-end Latency----------------
Mean E2EL (ms):                          xxxxx.xx
Median E2EL (ms):                        xxxxx.xx
P10/P25/P50/P75/P90/P99 E2EL (ms):      ...
---------------Time to First Token----------------
Mean TTFT (ms):                          xxxx.xx
...
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xx.xx
...
---------------Inter-token Latency----------------
Mean ITL (ms):                           xx.xx
...
==================================================
```

### Expected Results

Reference results from our experiments are provided in `ReferenceData/` for comparison:

| Directory | Contents | Paper Figure |
|---|---|---|
| `ReferenceData/MigrationComparison/` | KV cache migration vs recomputing latency | Figure 5 |
| `ReferenceData/ModelPlacement/` | Offline throughput and online latency logs per baseline | Figures 7, 8 |
| `ReferenceData/SpotTolerance/` | Online and offline CSV traces under simulated interruptions | Figures 9, 11 |
| `ReferenceData/SpotAvailabilityTrace/` | Instance availability trace used in experiments | Figure 10 |
| `ReferenceData/ConcurrentInitialization/` | Pipeline ready-time logs | Figure 12 |

## Appendix A: Pipeline Configuration Reference

Each experiment script defines pipeline configs as Python dicts. The key fields:

| Field | Type | Description | Example |
|---|---|---|---|
| `model_name` | str | HuggingFace model identifier | `"meta-llama/Llama-3.1-70B-Instruct"` |
| `total_num_layers` | int | Total transformer layers in the model | 80 (70B), 32 (8B) |
| `pp_layer_partition` | str | Comma-separated layer count per pipeline stage | `"20,20,20,10,10"` |
| `parallel_strategy` | list[int] | Tensor parallelism degree per stage | `[4,4,4,1,1]` |
| `gpu_memory_utilization` | float | Fraction of GPU memory to allocate | `0.85` |
| `max_model_len` | int | Maximum sequence length (tokens) | `8192` |
| `max_num_batched_tokens` | int | Maximum tokens per batch | `8192` |
| `max_num_seqs` | int | Maximum concurrent sequences | `512` |
| `model_source` | str | Weight loading source (only `"s3"` is supported) | `"s3"` |
| `s3_path` | str | S3 URI for model weights | `"s3://<YOUR_S3_BUCKET>/..."` |
| `num_gpu_blocks` | int | KV cache blocks available | `27549` |
| `max_batch_size` | int | Maximum batch size for scheduling | `442` |
| `mode` | str | Optional; `"hexgen"` for HEXGEN baselines | `"hexgen"` |

`num_gpu_blocks`, `max_batch_size`, `pp_layer_partition`, `parallel_strategy`, and `estimated_throughput` are outputs from running the Model Placement optimizer (see [Step 4](#step-4-run-model-placement-optimizer)). If you change the cluster configuration, re-run the optimizer and update these values accordingly.

## Appendix B: Trace Parameters Reference

Trace-based experiments (`run_trace_benchmark`) accept these parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `time_scale` | float | `1.0` | Inter-arrival time multiplier. `0.0` = offline (all at once), `1.0` = real-time, `3.0` = 3x slower (stretched) |
| `start_time` | float | `None` | Start time filter (seconds from first trace request) |
| `end_time` | float | `None` | End time filter (seconds from first trace request) |
| `num_requests` | int | `None` | Max requests to load from trace (`None` = full trace) |
| `run_initial_test` | bool | `True` | Send test requests to verify connectivity before the benchmark |
| `test_requests_per_pipeline` | int | `2` | Number of test requests per pipeline |

The trace is loaded from `Datasets/AzureLLMInferenceConvTrace_pruned_2048.csv`. Each row contains a timestamp, input token count, and output token count. The `time_scale` multiplier stretches the inter-arrival times: `time_scale=3.0` means a 20-minute trace is replayed over approximately 60 minutes.

## Directory Structure

```
ArtifactEvaluation/
  model_placement_optimizer.py    # Model placement optimizer entry point
  ReferenceData/                # Reference results from our experiments
    ModelPlacement/               #   Offline throughput & online latency logs
    SpotTolerance/                #   Interruption traces (online & offline)
    ConcurrentInitialization/     #   Pipeline ready-time logs
    MigrationComparisn/          #   KV cache migration vs recomputing
    SpotAvailabilityTrace/        #   Instance availability trace
  Datasets/
    AzureLLMInferenceConvTrace_pruned_2048.csv  # Pruned trace (default)
    AzureLLMInferenceTrace_conv.csv             # Full Azure trace
    figure.py                                    # Trace distribution visualization
  ModelPlacement/
    offline/                        # Figure 7: Offline throughput
      shuntserve.py                 #   ShuntServe (ours)
      hexgen.py                     #   HEXGEN baseline
      alpaserve.py                  #   AlpaServe baseline
      vllm.py                       #   vLLM baseline
      nodes.py                      #   Node IP configuration
    online/                         # Figure 8: Online serving
      shuntserve.py, hexgen.py, alpaserve.py, vllm.py, warmup.py
      nodes.py
    check_module_time/              # Module initialization timing
      test_warmup.py, test_time.py, run_test.sh
      nodes.py
  SpotInterruption/
    offline/                        # Figures 9, 11: Offline interruption simulation
      shuntserve.py, request_migration.py, no_handle.py
      only_ondemand.py, concurrent_initialization.py, warmup.py
      nodes.py
    online/                         # Figures 9, 11: Online interruption simulation
      shuntserve.py, request_migration.py, no_handle.py
      only_ondemand.py, concurrent_initialization.py, warmup.py
      nodes.py
    8B/                             # Appendix C: Simplified 8B test setup
      8B_stop_and_start.py, 8B_concurrent_initialization.py
      8B_only_ondemand.py, 8B_warmup.py
      nodes_8B.py
```

## Notes

1. **HEXGEN mode**: HEXGEN baseline scripts include `"mode": "hexgen"` in their pipeline config. This enables stage splitting where a single physical instance can host multiple pipeline stages.

2. **Request handler modes**: The `GlobalServer` constructor accepts a `request_handler_mode` parameter:
   - `"migration"` -- active request migration during node switch (continues in-flight requests on new nodes)
   - `"re-routing"` -- re-routes failed requests to surviving pipelines (restarts from scratch)
   - Default (no argument) -- standard round-robin scheduling with no interruption handling

3. **switch_nodes vs stop-and-restart**: The 70B SpotInterruption experiments use `global_server.switch_nodes(old_ips, new_ips)` for atomic in-place node migration. The 8B `8B_stop_and_start.py` uses `global_server.stop_nodes(old_ips)` followed by `global_server.create_pipeline()` with a 5-second wait, simulating a full pipeline recreation.

4. **Trace output location**: All trace CSVs are saved to `ArtifactEvaluation/Trace/`. The directory is created automatically.

## Appendix C: Simplified 8B Test Setup

This section provides a simplified test setup for reviewers who cannot afford the full 70B cluster. It uses Llama-3.1-8B-Instruct on g6.xlarge instances (single L4 GPU each) to verify basic executability of the interruption handling mechanisms.

**Path:** `SpotInterruption/8B/`

**Cluster:** See [Step 1 — Simplified Test Setup](#simplified-test-setup) for instance requirements, and [Step 5 — SpotInterruption/8B experiments](#spotinterruption8b-experiments) for node IP configuration.

| Strategy | Script | `request_handler_mode` |
|---|---|---|
| No Handle | `8B_no_handle.py` | `"re-routing"` |
| Concurrent Init | `8B_concurrent_initialization.py` | `"re-routing"` |
| ShuntServe | `8B_shuntserve.py` | `"migration"` |
| No Interruption | `8B_only_ondemand.py` | default |
| Warmup | `8B_warmup.py` | default |

8B pipeline configuration example:

```python
pipeline_1_config = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "total_num_layers": 32,
    "pp_layer_partition": "16,16",
    "parallel_strategy": [1,1],
    "num_gpu_blocks": 10074,
    "max_batch_size": 162,
    ...
}

node_layer_mapping_1 = [
    (spot_g6_xlarge_node_ip_1, 16),
    (spot_g6_xlarge_node_ip_2, 16),
]
```

Key difference from 70B: `8B_stop_and_start.py` uses `stop_nodes()` followed by a 5-second wait and then `create_pipeline()` (full pipeline recreation), rather than `switch_nodes()` (in-place node migration).

Event timeline (from `8B_stop_and_start.py`):

| Event | Time | Description |
|---|---|---|
| 1 | 5 min | Pipeline 1 stage + Pipeline 2 stage interrupted |
| 2 | 10 min | Pipeline 1 recovered |
| 3 | 15 min | Pipeline 2 recovered |
| 4 | 20 min | Pipeline 1 (both stages) interrupted |
| 5 | 25 min | Pipeline 1 recovered |
