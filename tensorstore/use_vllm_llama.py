from vllm.model_executor.model_loader.loader import _initialize_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.engine.arg_utils import AsyncEngineArgs, UsageContext
from vllm.worker.worker import init_worker_distributed_environment
from vllm.config import set_current_vllm_config
import os
import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from vllm.forward_context import set_forward_context

import time

def get_safetensor_iterator(st_file: str):
    with safe_open(st_file, framework="pt", device="cuda") as f:
        for key in f.keys():
            param = f.get_tensor(key)
            yield key, param

if __name__ == "__main__":
    model = "meta-llama/Llama-3.2-1B-Instruct"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
        "parallel_strategy": [1],
        # "tensor_parallel_size": 1,
        # "pipeline_parallel_size": 1,
    }
    engine_args = AsyncEngineArgs(**engine_args)
    usage_context = UsageContext.ENGINE_CONTEXT

    os.environ["VLLM_USE_V1"] = "0"
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    vllm_config.model_config.max_model_len = 4096

    with set_current_vllm_config(vllm_config):
        rank = 0
        distributed_init_method = "tcp://127.0.0.1:12345"
        local_rank = 0
        init_worker_distributed_environment(vllm_config, rank, distributed_init_method, local_rank)

        target_device = torch.device(vllm_config.device_config.device)
        with set_default_torch_dtype(vllm_config.model_config.dtype):
            with target_device:
                model_executable = _initialize_model(vllm_config=vllm_config)

            st_file = "/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/model.safetensors"
            st_it = get_safetensor_iterator(st_file)
            load_start = time.time()
            model_executable.load_weights(st_it)
            load_end = time.time()
            print(f"TOTAL MODEL LOAD TIME: {load_end - load_start} s")
