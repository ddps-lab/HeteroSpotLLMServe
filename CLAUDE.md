# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeteroSpotLLMServe is a distributed LLM inference serving system designed to run on heterogeneous spot clusters. It separates model loading from inference by using a TensorStore server architecture and leverages vLLM (v0.8.1) as the inference engine.

### Key Architecture Components

1. **VNode (Virtual Node)**: Abstraction for GPU nodes that manages:
   - Tensor Parallelism (TP) within a node across multiple GPUs
   - Pipeline Parallelism (PP) across different nodes
   - Each VNode runs tensor store servers
   - Only the first VNode (pipeline rank 0) runs the API server

2. **TensorStore Server**: Separate server that loads and manages model weights
   - Multi-threaded implementation in `TensorStore/mt_tensor_store_server.py`
   - Supports layer partitioning for pipeline parallelism
   - Uses Ray for distributed process management
   - **Important**: One TensorStore server instance must be run per GPU

3. **Global Server**: Orchestrates the distributed system
   - `GlobalServer/VNode.py`: Main abstraction for node management
   - `GlobalServer/command.py`: Command generation for starting services

### Parallelism Strategy

- **Tensor Parallelism (TP)**: Applied within a single node across multiple GPUs
- **Pipeline Parallelism (PP)**: Applied across different nodes
- Model layers are partitioned across nodes, with each node handling a subset of layers

## Common Development Commands

### Running TensorStore Server

**Important**: TensorStore servers must be run separately for each GPU on a node. The `--local-rank` parameter determines which CUDA device is used (e.g., local-rank=1 maps to cuda:1). The `--status-port` parameter is required and must be unique for each TensorStore instance.

#### Single GPU Example
```bash
python TensorStore/mt_tensor_store_server.py \
    --model-name=meta-llama/Llama-3.1-8B \
    --tensor-parallel-size=1 \
    --local-rank=0 \
    --start-layer-id=0 \
    --end-layer-id=32 \
    --status-port=10001 \
    --dtype=float16
```

#### Multi-GPU Example (4 GPUs on one node)
For a node with 4 GPUs, you need to run 4 separate TensorStore instances:

```bash
# GPU 0 (cuda:0)
python TensorStore/mt_tensor_store_server.py \
    --model-name=meta-llama/Llama-3.1-8B \
    --tensor-parallel-size=4 \
    --local-rank=0 \
    --start-layer-id=0 \
    --end-layer-id=32 \
    --dtype=float16 \
    --status-port=10001

# GPU 1 (cuda:1)
python TensorStore/mt_tensor_store_server.py \
    --model-name=meta-llama/Llama-3.1-8B \
    --tensor-parallel-size=4 \
    --local-rank=1 \
    --start-layer-id=0 \
    --end-layer-id=32 \
    --dtype=float16 \
    --status-port=10002

# GPU 2 (cuda:2)
python TensorStore/mt_tensor_store_server.py \
    --model-name=meta-llama/Llama-3.1-8B \
    --tensor-parallel-size=4 \
    --local-rank=2 \
    --start-layer-id=0 \
    --end-layer-id=32 \
    --dtype=float16 \
    --status-port=10003

# GPU 3 (cuda:3)
python TensorStore/mt_tensor_store_server.py \
    --model-name=meta-llama/Llama-3.1-8B \
    --tensor-parallel-size=4 \
    --local-rank=3 \
    --start-layer-id=0 \
    --end-layer-id=32 \
    --dtype=float16 \
    --status-port=10004
```

Note: Each instance needs a unique `--status-port` to avoid conflicts.

### Running API Server

**Important**: The API server should only be run once per pipeline, on the first node (pipeline rank 0). All nodes run TensorStore servers, but only the first node runs the API server.

#### Important Arguments Explained

**`--pp-layer-partition`**: Specifies how many layers each node should handle in pipeline parallelism.
- Format: Comma-separated string values
- Example: `--pp-layer-partition="6,20,6"` means:
  - Node 1: handles 6 layers
  - Node 2: handles 20 layers  
  - Node 3: handles 6 layers
- Total must equal the model's total layers (e.g., 6+20+6=32)

**`--parallel-strategy`**: Specifies the tensor parallelism degree for each node.
- Format: Space-separated integers (without quotes)
- Example: `--parallel-strategy 1 4 1` means:
  - Node 1: uses 1 GPU (no tensor parallelism)
  - Node 2: uses 4 GPUs (4-way tensor parallelism)
  - Node 3: uses 1 GPU (no tensor parallelism)

**`--node-rank-mapping` or `--node-rank-mapping-path`**: Maps node IPs to Ray ranks (required).
- `--node-rank-mapping`: Pass as JSON string
- `--node-rank-mapping-path`: Pass as path to JSON file
- Determines which Ray ranks are assigned to which nodes

#### Example Scenario
Consider 3 nodes with different GPU counts:
- Node 1 (192.168.10.1): 1 GPU
- Node 2 (192.168.10.2): 4 GPUs
- Node 3 (192.168.10.3): 1 GPU

For a 32-layer model, you might allocate layers proportionally:
- Node 1: 6 layers (using 1 GPU)
- Node 2: 20 layers (using 4 GPUs with tensor parallelism)
- Node 3: 6 layers (using 1 GPU)

The node-rank-mapping.json would be:
```json
{
    "192.168.10.1": [0],
    "192.168.10.2": [1, 2, 3, 4],
    "192.168.10.3": [5]
}
```

Complete command (run only on the first node in the pipeline):
```bash
python InferenceServer/api_server.py \
    --model=meta-llama/Llama-3.1-8B \
    --host=127.0.0.1 \
    --port=8000 \
    --dtype=float16 \
    --max_model_len=4096 \
    --pp-layer-partition="6,20,6" \
    --parallel-strategy 1 4 1 \
    --node-rank-mapping-path=../node_rank_mapping.json
```

**Important**: The API server should only be started on the node with pipeline rank 0 (the first node in the pipeline). Other nodes only run TensorStore servers.

### Testing
```bash
# Run benchmarks
cd benchmark
./run_benchmark_example.sh

# Test query
./query_example.sh
```

## Important Design Decisions

1. **Model Loading Separation**: vLLM's model loading and engine initialization are coupled, so this project separates them using TensorStore servers

2. **Layer Assignment**: Each VNode is assigned a range of layers [start_layer_id, end_layer_id). The total assigned layers must exactly match the model's total layers.

3. **Port Management**: 
   - TensorStore: Base port + node_index * 10 + gpu_index (to ensure unique ports per GPU)
   - API Server: Only runs on the first node (pipeline rank 0)

4. **Ray Integration**: Uses Ray for distributed process management and resource allocation

5. **GPU-TensorStore Mapping**: Each GPU requires its own TensorStore server instance with the appropriate local-rank

6. **API Server Strategy**: Only one API server per pipeline, running on the first node (pipeline rank 0)

## Key Files to Understand

- `GlobalServer/VNode.py`: Core abstraction for distributed nodes
- `TensorStore/mt_tensor_store_server.py`: Model weight loading and serving
- `InferenceServer/api_server.py`: vLLM-based inference server
- `GlobalServer/command.py`: Command generation utilities

## Development Notes

- Always specify `total_layers` when creating pipelines - it's a required parameter
- Layer validity is checked with assertions to ensure correct partitioning
- Use `--use-cpu-loading` flag for safer multi-threaded tensor loading
- Status servers run on each component for health checking
- The `node-rank-mapping` is crucial for Ray to correctly assign processes to nodes
- Remember to start one TensorStore server per GPU with the correct local-rank
- API server should only be started on the first node in the pipeline (pipeline rank 0)