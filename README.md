# ShuntServe: Cost-Efficient LLM Serving on Heterogeneous Spot GPU Clusters

ShuntServe is a distributed LLM serving system that maximizes cost-efficiency by leveraging heterogeneous spot GPU instances. It jointly optimizes model placement across GPUs with different capabilities, balances load via weighted round-robin scheduling, and tolerates spot interruptions through output-preserving request migration and concurrent initialization.

### Modules

Each module has its own README with detailed documentation.

| Module | Path | Description |
|---|---|---|
| **Global Server** | [`GlobalServer/`](GlobalServer/) | Central orchestrator — manages instances, routes requests via weighted round-robin load balancing, and handles spot interruptions through request migration and concurrent initialization. |
| **Model Placement Optimizer** | [`ModelPlacement/`](ModelPlacement/) | Profiling-free serving performance estimator (roofline model) and beam-search dynamic programming optimizer that determines optimal layer partitioning and parallelism configuration across heterogeneous GPUs. See [ModelPlacement/README.md](ModelPlacement/README.md). |
| **Inference Engine** | [`submodules/vLLM/`](submodules/vLLM/) | Modified [vLLM](https://github.com/vllm-project/vllm) v0.8.1 with support for heterogeneous pipeline parallelism, Shared Tensor Store integration, uneven layer partitioning, and per-stage tensor parallelism. |
| **Shared Tensor Store** | [`TensorStore/`](TensorStore/) | Downloads model weights from HuggingFace, partitions them for tensor parallelism, serializes in a custom binary format (TRAW), uploads to S3, and serves pre-partitioned weights directly to GPU memory. See [TensorStore/README.md](TensorStore/README.md). |
| **API Server** | [`InferenceServer/`](InferenceServer/) | FastAPI-based OpenAI-compatible serving endpoint that wraps the vLLM engine. Each pipeline instance runs its own API server. |
| **Infrastructure as Code** | [`IaC/`](IaC/) | Terraform configuration for provisioning the evaluation cluster on AWS (VPC, security groups, IAM, EC2 GPU instances). See [IaC/README.md](IaC/README.md). |
| **Artifact Evaluation** | [`ArtifactEvaluation/`](ArtifactEvaluation/) | End-to-end scripts for reproducing experiments, including offline/online throughput, per-pipeline ranking, and spot interruption scenarios. See [ArtifactEvaluation/README.md](ArtifactEvaluation/README.md). |

### Directory Structure

```
ShuntServe/
├── GlobalServer/                        # Global Server (Instance Manager, Load Balancer, Request Scheduler)
│   ├── global_server.py
│   ├── VNode.py
│   ├── request_handler.py
│   ├── benchmark_utils.py
│   ├── evaluation_utils.py
│   └── ...
├── ModelPlacement/                      # Serving Performance Estimator + Model Placement Optimizer
│   ├── README.md
│   ├── shuntserve_optimizer.py          # ShuntServe beam search DP optimizer
│   ├── hexgen_optimizer.py              # HEXGEN genetic algorithm optimizer
│   ├── alpaserve_optimizer.py           # AlpaServe DP layer partitioning (uses official code)
│   ├── estimator_utils.py              # Profiling-free throughput/latency estimation (roofline model)
│   ├── hardware_specs.py               # GPU, interconnect, and instance specifications
│   ├── cluster_pool.py                 # Cluster resource and pricing management
│   ├── alpaserve_lib/                  # AlpaServe official code (DP, simulator, placement policy)
│   └── hexgen/                         # HEXGEN internal modules (cost model, GA, simulator)
├── submodules/
│   └── vLLM/                            # Inference Engine (modified vLLM v0.8.1)
├── TensorStore/                         # Shared Tensor Store + Remote Storage (S3)
│   ├── README.md
│   ├── raw_s3_model_uploader.py
│   ├── raw_s3_tensor_store_server.py
│   └── upload_model.sh
├── InferenceServer/                     # API Server (FastAPI + vLLM)
│   ├── api_server.py
│   └── launch_server_example.sh
├── ArtifactEvaluation/                  # Experiment reproduction scripts
│   ├── README.md
│   ├── model_placement_optimizer.py
│   ├── Datasets/
│   ├── ReferenceData/
│   ├── ModelPlacement/
│   │   ├── offline/llama3-70b/          # Offline throughput benchmark
│   │   ├── online/llama3-70b/           # Online serving benchmark
│   │   ├── per_pipeline/                # Per-pipeline ranking evaluation
│   │   │   ├── llama3-70b/              #   Llama-3.1-70B-Instruct
│   │   │   └── qwen3-32b/              #   Qwen3-32B
│   │   └── check_module_time/           # Module initialization timing
│   ├── PerformanceEstimation/           # Estimator accuracy evaluation
│   └── SpotTolerance/
│       ├── offline/                     # Offline interruption simulation
│       ├── online/                      # Online interruption simulation
│       └── 8B/                          # Simplified 8B test setup
├── IaC/                                 # Infrastructure as Code (Terraform)
│   ├── README.md
│   ├── main.tf
│   └── ec2-cluster-module/
├── profiling/                           # GPU profiling utilities
├── install.sh                           # Inference engine installer
├── protocols.py                         # Inter-component communication protocols
└── utils.py                             # SSH and Ray placement group utilities
```

## Supported Models

| Model | Parameters | Layers | Architecture |
|---|---|---|---|
| Llama-3.1-70B-Instruct | 70B | 80 | GQA, SiLU, RoPE, RMSNorm |
| Qwen3-32B | 32B | 64 | GQA, SiLU, RoPE, RMSNorm |
| Llama-3.1-8B-Instruct | 8B | 32 | GQA, SiLU, RoPE, RMSNorm (simplified test only) |

Other models with the same architecture family (GQA + SiLU + RoPE + RMSNorm) should work with minimal changes to the estimator configuration.

## Prerequisites

> **Note:** The following describes the environment we used for development and evaluation. ShuntServe may work in other configurations, but you may need to modify the code for compatibility.

- **CUDA Toolkit** 12.8+
- **GPU Driver** 570+
- **NCCL** 2.26.2+
- **Python** 3.12
- **AWS account** with S3 access (for model weight storage via Tensor Store)
- **SSH client** (the Global Server manages remote nodes over SSH)
- **Terraform** (only needed if using the IaC module to provision the cluster)

> **Tip:** If you are using AWS, the [Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) (e.g., Deep Learning OSS Nvidia Driver AMI) satisfies CUDA, GPU Driver, NCCL, and Python requirements out of the box.

### Tested Cluster Configuration

| Instance Type | GPU | Count |
|---|---|---|
| g5.12xlarge | 4x NVIDIA A10G | 2 |
| g6.12xlarge | 4x NVIDIA L4 | 3 |
| g6e.xlarge | 1x NVIDIA L40S | 4 |

This gives a total of 9 instances with 24 GPUs. Other GPU types and instance configurations are supported — see [`ModelPlacement/hardware_specs.py`](ModelPlacement/hardware_specs.py) for the full list.

## Installation

### Step 0 — Install NVIDIA Driver (Optional)

If you are using an AWS Deep Learning AMI (e.g., Deep Learning OSS Nvidia Driver AMI), CUDA, GPU Driver, NCCL, and Python are already installed — skip to [Step 1](#step-1--environment-setup).

We used **Ubuntu 24.04** with **CUDA 12.8**. If you use a different OS, you may need to adjust the commands below and potentially modify parts of the codebase.

- CUDA 12.8 download page: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

sudo apt-get install -y nvidia-open-570

echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

sudo apt update
sudo apt install libnccl2=2.26.2-1+cuda12.8 libnccl-dev=2.26.2-1+cuda12.8

sudo reboot
```

### Step 1 — Environment Setup

Python 3.12 is assumed to be already installed. Our code sends `python` commands over SSH, so the `python-is-python3` package is required:

```bash
sudo apt install -y python-is-python3 python3-pip
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**SSH key sharing:** ShuntServe requires all instances in the cluster to communicate with each other over SSH without interactive authentication. We recommend generating a dedicated SSH key pair, placing both the public and private keys in `~/.ssh/` on every node, and then creating an AMI from this setup. If your cluster already allows passwordless SSH between nodes, this step is not necessary.

### Step 2 — Install ShuntServe and Inference Engine (custom vLLM 0.8.1)

```bash
git clone https://github.com/ddps-lab/ShuntServe.git
cd ShuntServe
bash install.sh
```

The install script installs the modified vLLM in editable mode using a precompiled wheel. After installation, set the following environment variable:

```bash
export VLLM_USE_V1=0
```

### Step 3 — Install Baseline Packages (Optional)

To run the HEXGEN baseline in the Model Placement Optimizer (uses a genetic algorithm):

```bash
pip install deap
```

### Step 4 — Provision Cluster with Terraform (Optional)

We provide Terraform configurations for automated cluster provisioning on AWS. To use this, create an AMI from an instance where Steps 0–2 are completed, then follow the instructions in [IaC/README.md](IaC/README.md).

### Step 5 — Start Ray Head and Run Experiments

On the head node, start two Ray head processes. Worker nodes do not need any Ray configuration — they are managed automatically.

```bash
ray start --head --port=6379 --disable-usage-stats
ray start --head --port=6380 --disable-usage-stats
```

Once the cluster is ready, follow the [Artifact Evaluation guide](ArtifactEvaluation/README.md) to reproduce the experiments.

## Getting Started

1. **Upload model weights** — Use the [Tensor Store](TensorStore/README.md) to partition and upload model weights to S3. This step does not require a GPU.
2. **Run the placement optimizer** — Use the [Model Placement Optimizer](ModelPlacement/README.md) to determine optimal pipeline configurations for your cluster.
3. **Provision the cluster** — Use the [IaC module](IaC/README.md) or set up GPU instances manually based on the placement result.
4. **Run experiments** — Follow the [Artifact Evaluation guide](ArtifactEvaluation/README.md) to reproduce the experiments.

> **Tip:** The full evaluation uses Llama-3.1-70B on 9 instances (24 GPUs), which can be costly. A simplified 8B test setup using Llama-3.1-8B-Instruct on 3x g6.xlarge (single L4 GPU each) is available to verify basic executability at reduced cost. See [Appendix C in ArtifactEvaluation/README.md](ArtifactEvaluation/README.md#appendix-c-simplified-8b-test-setup) for details.
