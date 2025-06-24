from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
tensor_parallel_size = 2

# Compute Capacity 가 80 보다 작은 경우 Bfloat16 을 지원하지 않는다. Llama-3.2-1B 는 Bfloat16 이 default 인듯
llm = LLM(model="meta-llama/Meta-Llama-3-8B", 
          dtype="float16",
          tensor_parallel_size=tensor_parallel_size)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # !r 은 repr() 를 호출한다. 이는 단순하게 따옴표까지 출력하는것이라고 생각하면 된다.
    # 그냥 출력하는 경우 str() 를 출력하게 된다. 이 경우에는 따옴표를 출력하지 않는다.
    # !r 를 붙여서 repr() 를 호출하는 것이 디버깅 상 편하다고 한다.
    print(f"Prompt: {prompt!r} \n\tGenerated text: {generated_text!r}")