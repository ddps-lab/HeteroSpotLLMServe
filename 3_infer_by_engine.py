import argparse

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from torch.distributed import destroy_process_group
from vllm.usage.usage_lib import UsageContext

def initialize_engine() -> LLMEngine:
    """
    engine_args = EngineArgs(
    model=model,
    task=task,
    tokenizer=tokenizer,
    tokenizer_mode=tokenizer_mode,
    skip_tokenizer_init=skip_tokenizer_init,
    trust_remote_code=trust_remote_code,
    allowed_local_media_path=allowed_local_media_path,
    tensor_parallel_size=tensor_parallel_size,
    dtype=dtype,
    quantization=quantization,
    revision=revision,
    tokenizer_revision=tokenizer_revision,
    seed=seed,
    gpu_memory_utilization=gpu_memory_utilization,
    swap_space=swap_space,
    cpu_offload_gb=cpu_offload_gb,
    enforce_eager=enforce_eager,
    max_seq_len_to_capture=max_seq_len_to_capture,
    disable_custom_all_reduce=disable_custom_all_reduce,
    disable_async_output_proc=disable_async_output_proc,
    hf_overrides=hf_overrides,
    mm_processor_kwargs=mm_processor_kwargs,
    override_pooler_config=override_pooler_config,
    compilation_config=compilation_config_instance,
    **kwargs,
    )   
    """
    model = "meta-llama/Llama-3.2-3B"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
    }

    engine_args = EngineArgs(**engine_args)
    usage_context = UsageContext.ENGINE_CONTEXT
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    vllm_config.model_config.max_model_len = 4096
    return LLMEngine.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_stats=engine_args.disable_log_stats,
            )

def create_test_prompts() -> list[tuple[str, SamplingParams]]:
    prompts = [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2,
                        temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
    ]
    return prompts

def process_requests(engine: LLMEngine, prompts: list[tuple[str, SamplingParams]]):
    request_id = 0

    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params = prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)

            print(f"Add request {request_id} with prompt: {prompt!r} to engine")
            if len(prompts) == 0:
                print("All requests have been added to engine")

            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                print(f"Request id : {request_output.request_id} finished")
                continue
            for i, output in enumerate(request_output.outputs):
                print(f"Request id : {request_output.request_id}, output[{i}] : {output.text}")


def main():
    engine = initialize_engine()
    prompts = create_test_prompts()
    process_requests(engine, prompts)

if __name__ == '__main__':
    main()
    destroy_process_group()