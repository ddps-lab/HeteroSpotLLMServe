#!/bin/bash

# 실제 데이터셋을 활용할 경우
# python benchmark_serving.py --backend vllm \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --dataset-name sharegpt \
#     --dataset-path=/home/ubuntu/vllm-example/benchmark/dataset/sharegpt/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --save-result --save-detailed \
#     --num-prompts 1000 \
#     --seed 42 \
#     --metric-percentiles "25,50,75,99" \
#     --percentile-metrics ttft,tpot,itl,e2el \
#     # --request-rate=0.1 \
#     # --max-concurrency=4 \
#     --result-dir=./results --metadata pipeline-parallel-strategy=1,4 nodes=g4dn.xlarge,g4dn.12xlarge

# 시스템 성능 측정을 위해서 고정된 input length 와 output length 를 사용할 경우
python benchmark_serving.py --backend vllm \
    --model=meta-llama/Llama-3.1-8B \
    --dataset-name=random \
    --random-input-len=1024 \
    --random-output-len=128 \
    --num-prompts 1024 \
    --ignore-eos \
    --metric-percentiles="25,50,75,99" \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --save-result --save-detailed \
    --result-dir=./results --metadata parallel-strategy=1 layer-partition="32" nodes=1xg6.xlarge
