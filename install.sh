#!/bin/bash
set -euo pipefail

# install.sh — Install the modified vLLM for ShuntServe.
# Version metadata was captured from git before .git was removed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/submodules/vLLM"

echo "Installing vLLM (editable mode with precompiled wheel)..."
SETUPTOOLS_SCM_PRETEND_VERSION="0.8.1.dev30+g6ae43b8a8" \
VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/61c7a1b856e32ef8b12c70abcb9fd9ad22619a13/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl" \
pip install --editable . --break-system-packages

export VLLM_USE_V1=0

echo ""
echo "Installation complete."
echo "Note: set VLLM_USE_V1=0 in your environment before running."
