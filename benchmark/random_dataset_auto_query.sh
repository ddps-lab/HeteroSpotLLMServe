#!/bin/bash

# 세 개의 인자를 받아야 함
# 1. num-prompts
# 2. input-length
# 3. output-length

# 인자 확인
if [ $# -ne 3 ]; then
    echo "Usage: $0 <num-prompts> <input-length> <output-length>"
    exit 1
fi

# 기본 명령어 템플릿
BASE_CMD="python benchmark_serving.py --backend vllm \
    --model=meta-llama/Llama-3.1-8B \
    --dataset-name=random \
    --num-prompts=$1 \
    --random-input-len=$2 \
    --random-output-len=$3 \
    --ignore-eos \
    --metric-percentiles="25,50,75,99" \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --save-result --save-detailed \
    --result-dir=./results"

CONCURRENCY_VALUES=(1 2 4 8 16 32 64 128 256)

# --max-concurrency 값을 1부터 8까지 반복
for i in "${CONCURRENCY_VALUES[@]}"
do
  echo "=================================================="
  echo "Running benchmark with --max-concurrency=$i"
  echo "=================================================="

  # 현재 concurrency 값으로 명령어 완성
  FULL_CMD="$BASE_CMD --max-concurrency=$i --metadata parallel-strategy=1 layer-partition=32 nodes=1xg6.xlarge max-concurrency=$i"

  # 명령어 실행
  eval $FULL_CMD

  # 각 실행 후 잠시 대기 (선택 사항, 시스템 안정화 시간)
  # sleep 5

  echo "--------------------------------------------------"
  echo "Finished benchmark with --max-concurrency=$i"
  echo "--------------------------------------------------"
done

echo "All benchmarks completed."