import time

init_end_to_end_start = time.time()

module_loading_start = time.time()
import os
from typing import List, Dict
vllm_loading_start = time.time()
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from transformers import LlamaConfig
vllm_loading_end = time.time()
print(f"vllm loading time: {vllm_loading_end - vllm_loading_start} seconds")
from vllm.usage.usage_lib import UsageContext
from torch.distributed import destroy_process_group
from utils import create_placement_group_and_bundle_indices, get_computation_latency, get_computation_latency_hexgen
import json
module_loading_end = time.time()
print(f"total module loading time: {module_loading_end - module_loading_start} seconds")
# 각 노드 IP에 할당할 rank 리스트를 정의합니다.
# 예시: node_rank_mapping = {
#     "172.31.20.243": [0],
#     "172.31.10.100": [1, 2], # 172.31.10.100 노드에는 rank 1과 2 할당
#     "172.31.30.50": [3]    # 172.31.30.50 노드에는 rank 3 할당
# }
node_rank_mapping = json.load(open("node_rank_mapping.json"))

def get_latency_estimation(config: LlamaConfig, input_sequence_length: int, output_sequence_length: int):
    
    hidden_dim_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads",
                                  config.num_attention_heads)
    intermediate_dim_size = config.intermediate_size
    data_byte_size = 2 # 그냥 일단 하드코딩
    device_FLOPS = 242 * 10**12 # 그냥 일단 하드코딩 242 TFlops (L4)
    device_memory_bandwidth = 300 * 10**9 # 그냥 일단 하드코딩 300GB/s

    total_layer_num = config.num_hidden_layers

    hexgen_compute_latency, hexgen_memory_scan_latency = get_computation_latency_hexgen(
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        hidden_dim_size=hidden_dim_size,
        data_byte_size=data_byte_size,
        device_FLOPS=device_FLOPS,
        device_memory_bandwidth=device_memory_bandwidth,
    )

    print(f"hexgen_compute_latency: {hexgen_compute_latency:.3f} seconds / hexgen_memory_scan_latency: {hexgen_memory_scan_latency:.3f} seconds")
    hexgen_estimation = hexgen_compute_latency + hexgen_memory_scan_latency

    compute_latency, memory_scan_latency = get_computation_latency(
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        hidden_dim_size=hidden_dim_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_dim_size=intermediate_dim_size,
        data_byte_size=data_byte_size,
        device_FLOPS=device_FLOPS,
        device_memory_bandwidth=device_memory_bandwidth,
    )

    print(f"compute_latency: {compute_latency:.3f} seconds / memory_scan_latency: {memory_scan_latency:.3f} seconds")
    estimation = compute_latency + memory_scan_latency

    return hexgen_estimation * total_layer_num, estimation * total_layer_num

async def main():
    # V0 버전 사용
    os.environ["VLLM_USE_V1"] = "0"

    model = "meta-llama/Llama-3.1-8B"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
        "parallel_strategy": [1],
        "enforce_eager": True,
        "gpu_memory_utilization": 0.25,
        # "tensor_parallel_size": 1,
        # "pipeline_parallel_size": 1,
    }
    start_time = time.time()
    engine_args = AsyncEngineArgs(**engine_args)
    end_time = time.time()
    print(f"engine_args loading time: {end_time - start_time} seconds")
    usage_context = UsageContext.ENGINE_CONTEXT

    # 해당 코드에서 config.json 을 받아오며 engine 버전을 어떤 것을 사용할 것인지 정하게 됨.
    # 예를 들어 V1 을 사용하려 했으나 Compute Capacity 가 80아래면 V0 로 Fall back 하게 됨
    start_time = time.time()
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    end_time = time.time()
    print(f"vllm_config loading time by create_engine_config(): {end_time - start_time} seconds")
    vllm_config.model_config.max_model_len = 4096

    start_time = time.time()
    placement_group = create_placement_group_and_bundle_indices(node_rank_mapping)
    end_time = time.time()
    print(f"placement_group loading time by create_placement_group_and_bundle_indices(): {end_time - start_time} seconds")
    vllm_config.parallel_config.placement_group = placement_group
    os.environ["VLLM_PP_LAYER_PARTITION"] = "32"

    # 아래 코드에서 tokenizer 와 model 의 safetensor 를 받아오며
    # CUDA graph shape capturing 을 한다.
    engine = AsyncLLMEngine.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_stats=engine_args.disable_log_stats,
            )
    init_end_to_end_end = time.time()
    print(f"Inference Process Init Time: {init_end_to_end_end - init_end_to_end_start} seconds")


    # 초기에 한번 추론 수행해봄
    initial_input = "What is LLM?"
    example_input = {
        "prompt": initial_input,
        "stream": False,
        "temperature": 0.0,
        "request_id": 0,
    }

    results_generator = engine.generate(
        example_input["prompt"],
        SamplingParams(temperature=example_input["temperature"]),
        example_input["request_id"])

    async for request_output in results_generator:
        for text_output in request_output.outputs:
            print(f"Output: {text_output.text.strip()}\n")


    # 토크나이저 가져오기 (engine 객체에 tokenizer가 있다고 가정)
    tokenizer = engine.engine.tokenizer # engine.tokenizer 가 아니라 engine.engine.tokenizer 일 수 있음 (vLLM 내부 구조에 따라 다름)
    
    input_sequence_length = 512
    output_sequence_length = 512
    original_prompt = "What is LLM?"

    # 원본 프롬프트를 토큰화
    original_tokens = tokenizer.encode(original_prompt)
    
    # 목표 토큰 수에 도달할 때까지 프롬프트 반복
    repeated_tokens = []
    while len(repeated_tokens) < input_sequence_length:
        repeated_tokens.extend(original_tokens)
        # 반복 후 바로 다음 토큰이 추가될 때 512를 넘는지 확인하기 위해 공백 토큰 추가 고려 (모델/토크나이저에 따라 다름)
        # 여기서는 단순 반복 후 자르는 방식으로 진행
    
    # 목표 토큰 수에 맞게 자르기
    final_tokens = repeated_tokens[:input_sequence_length]
    
    # 토큰 ID 리스트를 다시 문자열로 디코딩
    prompt = tokenizer.tokenizer.decode(final_tokens)

    example_input = {
        "prompt": prompt,
        "stream": False, # assume the non-streaming case
        "temperature": 0.0,
        "request_id": 0,
    }

    inference_start = time.perf_counter()
    # start the generation
    results_generator = engine.generate(
        example_input["prompt"],
        SamplingParams(temperature=example_input["temperature"],
                       min_tokens=output_sequence_length,
                       max_tokens=output_sequence_length,
                       ignore_eos=True),
        example_input["request_id"])

    # get the results
    final_output = None

    # 실행한 메인 스레드에서만 출력하게 설정되어 있음
    async for request_output in results_generator:
        for text_output in request_output.outputs:
            continue
    #         print(f"Prompt: {prompt}")
    #         print(f"Output: {text_output.text.strip()}\n")
    inference_end = time.perf_counter()

    hexgen_estimation, estimation = get_latency_estimation(vllm_config.model_config.hf_config, input_sequence_length, output_sequence_length)
    print(f"Inference Time: {inference_end - inference_start:.3f} seconds")
    print(f"input_sequence_length: {input_sequence_length} / output_sequence_length: {output_sequence_length}")
    print(f"Hexgen Estimation: {hexgen_estimation:.3f} seconds / Modified Estimation: {estimation:.3f} seconds")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    destroy_process_group()
