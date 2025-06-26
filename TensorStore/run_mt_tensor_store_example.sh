python mt_tensor_store_server.py --model-name=meta-llama/Llama-3.1-8B \
    --dtype=float16 \
    --start-layer-id=0 \
    --end-layer-id=32 \
    --status-port=10001 \
    --tensor-parallel-size=1 \
    --use-cpu-loading