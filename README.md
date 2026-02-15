# ShuntServe: LLM Serving on Heterogeneous Spot GPU Clusters

> **Note:** Detailed environment setup documentation is planned and will be added here.

## Environment Setup

Key requirements:

- **CUDA Toolkit**: 12.8+
- **GPU Driver**: 570+
- **NCCL**: 2.26.2+
- **Python**: 3.12 (via conda/miniconda)
- **vLLM**: v0.8.1 (included as git submodule)

### Quick Install

```bash
git submodule update --init --recursive
cd submodules/vLLM
VLLM_USE_PRECOMPILED=1 pip install --editable .
export VLLM_USE_V1=0
```

## Artifact Evaluation

For artifact evaluation and experiment reproduction, see [ArtifactEvaluation/README.md](ArtifactEvaluation/README.md).

## Model Placement Optimizer

For model placement algorithm details, see [ModelPlacement/README.md](ModelPlacement/README.md).
