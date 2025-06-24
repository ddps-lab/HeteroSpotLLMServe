# LLM Serving on Heterogeneous Spot Cluster
Currently used version of vllm : v0.8.1

### Install build essential & cmake
```bash
sudo apt update
sudo apt install -y build-essential
sudo apt install -y cmake
```
```bash
gcc --version   # gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
cmake --version # cmake version 3.28.3
```

### Install cuda toolkit (12.8) & GPU driver (570)
You can check comparability from https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

sudo apt-get install -y nvidia-open-570

echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
sudo reboot
```
```bash
nvcc --version
nvidia-smi --version
```

### Install nccl (https://developer.nvidia.com/nccl/nccl-download)
keyring 은 위에서 이미 받았으므로 과정에서 제외
```bash
sudo apt update
sudo apt install libnccl2=2.26.2-1+cuda12.8 libnccl-dev=2.26.2-1+cuda12.8
```

I installed miniconda from : https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation
```bash
conda create -n vllm-example python=3.12 -y && conda activate vllm-example
```

### Install vLLM
```bash
cd HeteroSpotLLMServe
git submodule update --init --recursive
cd submodules/vLLM
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

I am using vscode. If your vscode can't detect vllm package, add `"python.analysis.extraPaths": ["./submodules/vllm"]` to `settings.json` for debugging.

I don't use V1 Engine. So I commanded `export VLLM_USE_V1=0`
