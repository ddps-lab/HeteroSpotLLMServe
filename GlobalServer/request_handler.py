"""
Request handler for global server.
"""
import json
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
    
    @classmethod
    def create(cls, request_input: 'RequestInput') -> 'Request':
        """Create a new request with a unique ID."""
        global _request_counter
        request_id = _request_counter
        _request_counter += 1
        return cls(
            request_id=request_id,
            input=request_input
        )

@dataclass
class RequestInput:
    """Represents a single inference request."""
    prompt: str
    prompt_len: int
    expected_output_len: int
    api_url: str = ""
    model: str = ""

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
    seed: int = 0
) -> List[RequestInput]:
    """
    Generate random requests for benchmarking.
    
    Args:
        num_prompts: Number of prompts to generate
        input_len: Input sequence length (number of tokens)
        output_len: Expected output length
        model_name: Model name for tokenizer
        seed: Random seed
        
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
            model=model_name
        )
        requests.append(request)
    
    return requests

async def async_request(request: Request) -> RequestOutput:
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
        
        headers = {"Content-Type": "application/json"}
        
        # Use existing output or create new one
        if request.output:
            output = request.output
            generated_text = output.generated_text
        else:
            output = RequestOutput()
            output.prompt_len = request_input.prompt_len
            generated_text = ""
        
        st = time.perf_counter()
        most_recent_timestamp = st
        
        # For continued requests, we don't have a new TTFT
        first_chunk_received = request.output is not None
        
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
                                    generated_text += text
                                    
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    output.ttft = timestamp - st
                                else:
                                    # Inter-token latency
                                    output.itl.append(timestamp - most_recent_timestamp)
                                most_recent_timestamp = timestamp
                                
                                if choices[0].get("finish_reason") is not None:
                                    output.latency = time.perf_counter() - st
                                    output.output_tokens = len(output.itl) + 1
                                    
                            # Check for usage stats
                            if usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens", output.output_tokens)
                                
                        except json.JSONDecodeError:
                            continue
                    
                    output.generated_text = generated_text
                    output.success = True
                    
                    # Ensure latency is set if not already
                    if output.latency == 0:
                        output.latency = time.perf_counter() - st
                        
                else:
                    output.error = f"HTTP {response.status}: {await response.text()}"
                    output.success = False
                    
        except Exception as e:
            output.success = False
            output.error = str(e)
            
    return output