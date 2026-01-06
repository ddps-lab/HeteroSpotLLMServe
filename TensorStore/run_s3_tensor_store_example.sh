python s3_tensor_store_server.py \
    --model-name=meta-llama/Llama-3.1-8B-Instruct \
    --s3-path=s3://hetero-spot-llm-serve-models/meta-llama/Llama-3.1-8B-Instruct \
    --start-layer-id=0 \
    --end-layer-id=32 \
    --status-port=10001 \
    --tensor-parallel-size=1 \
    --tensor-parallel-rank=0 \
    --block-size=16 \
    --gpu-memory-utilization=0.9 \
    --swap-space=4.0 \
    --cache-dtype=auto \
    --pipeline-parallel-size=1 \
    --pipeline-parallel-rank=0 \
    --max-model-len=4096
    # --dtype=float16 \