#!/bin/bash

# 현재 노드의 local ip 주소 가져오기
local_ip=$(hostname -I | awk '{print $1}')

# python api_server.py --model="meta-llama/Llama-3.2-3B-Instruct" --host=127.0.0.1 --port=8000 \
#     --dtype=float16 --max_model_len=9182 --node-rank-mapping='{"172.31.20.124": [0]}' \
#     --pp-layer-partition="28" --parallel-strategy 1

python api_server.py --model=meta-llama/Llama-3.1-8B --host=127.0.0.1 --port=8000  \
    --dtype=float16 --max_model_len=4096 --node-rank-mapping-path=./node_rank_mapping.json \
    --pp-layer-partition="32" --parallel-strategy=1 --gpu-memory-utilization=0.25 \
    --max-num-batched-tokens=4096 --max-num-seqs=16
