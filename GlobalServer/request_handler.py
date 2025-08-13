"""
Request handler for global server.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List
import aiohttp
import numpy as np
from transformers import AutoTokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# Global counter for request IDs
_request_counter = 0

@dataclass
class Request:
    """Represents a unique request with tracking information."""
    request_id: int
    input: 'RequestInput'
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0
    output: Optional['RequestOutput'] = None  # Output object (exists if request was halted)
    sended_at: Optional[float] = None  # When forwarded to inference pipeline
    halted_at: Optional[float] = None  # When request was halted (if applicable)
    _completion_event: Optional[asyncio.Event] = field(default=None, init=False, repr=False)
    
    @classmethod
    def create(cls, request_input: 'RequestInput') -> 'Request':
        """Create a new request with a unique ID."""
        global _request_counter
        request_id = _request_counter
        _request_counter += 1
        request = cls(
            request_id=request_id,
            input=request_input
        )
        request._completion_event = asyncio.Event()
        return request
    
    async def wait_for_completion(self):
        """Wait for this request to complete."""
        if self._completion_event:
            await self._completion_event.wait()
        return self

@dataclass
class RequestInput:
    """Represents a single inference request."""
    prompt: str
    prompt_len: int
    expected_output_len: int
    api_url: str = ""
    model: str = ""
    ignore_eos: bool = False

@dataclass
class RequestOutput:
    """Output from async request function."""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # Inter-token latencies
    prompt_len: int = 0
    error: str = ""

def generate_random_requests(
    num_prompts: int,
    input_len: int, 
    output_len: int,
    model_name: str,
    seed: int = 0,
    ignore_eos: bool = False
) -> List[RequestInput]:
    """
    Generate random requests for benchmarking.
    
    Args:
        num_prompts: Number of prompts to generate
        input_len: Input sequence length (number of tokens)
        output_len: Expected output length
        model_name: Model name for tokenizer
        seed: Random seed
        ignore_eos: Whether to ignore EOS token during generation
        
    Returns:
        List of RequestInput objects
    """
    # Set random seed
    np.random.seed(seed)
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    
    requests = []
    
    for i in range(num_prompts):
        # Generate random token IDs
        # Use different offset for each prompt to ensure variety
        offset = np.random.randint(0, vocab_size)
        token_ids = ((offset + i + np.arange(input_len)) % vocab_size).tolist()
        
        # Decode to text
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Create request
        request = RequestInput(
            prompt=prompt,
            prompt_len=input_len,
            expected_output_len=output_len,
            model=model_name,
            ignore_eos=ignore_eos
        )
        requests.append(request)
    
    return requests

async def async_request(request: Request, logger: logging.Logger) -> RequestOutput:
    """Send async request to OpenAI-compatible API.
    
    Args:
        request: The request object containing input and possibly existing output
    
    Returns:
        RequestOutput with the results
    """
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        request_input = request.input
        
        # If continuing from existing output, adjust the prompt and remaining tokens
        if request.output and request.output.generated_text:
            # Continue generation from where it left off
            prompt = request_input.prompt + request.output.generated_text
            remaining_tokens = request_input.expected_output_len - request.output.output_tokens
        else:
            prompt = request_input.prompt
            remaining_tokens = request_input.expected_output_len
            
        payload = {
            "model": request_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": remaining_tokens,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        
        # Add ignore_eos if specified
        if request_input.ignore_eos:
            payload["ignore_eos"] = True
        
        headers = {"Content-Type": "application/json"}

        first_chunk_received = False
        
        # Use existing output or create new one
        if request.output:
            output = request.output
            first_chunk_received = True
        else:
            output = RequestOutput()
            output.prompt_len = request_input.prompt_len
        
        most_recent_timestamp = request.sended_at
        
        try:
            async with session.post(url=request_input.api_url, 
                                  json=payload, 
                                  headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        
                        chunk = chunk_bytes.decode("utf-8")
                        if chunk.startswith("data: "):
                            chunk = chunk.removeprefix("data: ")
                        
                        if chunk == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(chunk)
                            
                            if choices := data.get("choices"):
                                text = choices[0].get("text", "")
                                if text:
                                    output.generated_text += text
                                    
                                timestamp = time.time()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    output.ttft = timestamp - most_recent_timestamp
                                    if output.output_tokens == 0:
                                        output.output_tokens = 1
                                    else:
                                        output.output_tokens += 1
                                else:
                                    # Inter-token latency
                                    output.itl.append(timestamp - most_recent_timestamp)
                                    output.output_tokens += 1
                                most_recent_timestamp = timestamp

                                if choices[0].get("finish_reason") is not None:
                                    output.latency = time.time() - request.sended_at
                                    output.output_tokens = len(output.itl) + 1
                                    
                            # Check for usage stats
                            # 여기서 이번 request 에서 얼마나 토큰이 생성되었는지 알 수 있음 (completion_tokens)
                            # 일단 지금 필요하지 않으니 주석처리한다.
                            # if usage := data.get("usage"):
                            #     output.output_tokens = usage.get("completion_tokens", output.output_tokens)
                                
                        except json.JSONDecodeError:
                            continue
                    
                    output.success = True
                    output.latency = time.time() - request.sended_at
                        
                else:
                    output.error = f"HTTP {response.status}: {await response.text()}"
                    output.success = False
                    
        except Exception as e:
            output.success = False
            output.error = str(e)
    return output