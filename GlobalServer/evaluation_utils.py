"""
Evaluation utilities for GlobalServer benchmarking.
Contains dataset loading, trace-based request generation, and trace replay functions.
"""
import asyncio
import csv
import os
import time
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer

from request_handler import RequestInput, Request
from benchmark_utils import BenchmarkMetrics, calculate_benchmark_metrics


# ============================================================================
# Section 1: Dataset Loading
# ============================================================================

def load_azure_trace(
    csv_path: str,
    max_requests: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
    max_generated_tokens: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> List[Tuple[float, int, int]]:
    """
    Load Azure LLM Inference Conversation Trace dataset.

    Args:
        csv_path: Path to the CSV file
        max_requests: Maximum number of requests to load (None for all)
        max_context_tokens: Filter out requests with context tokens > this value
        max_generated_tokens: Filter out requests with generated tokens > this value
        start_time: Start time in seconds from the first request (None for no lower bound)
        end_time: End time in seconds from the first request (None for no upper bound)

    Returns:
        List of (arrival_time, prompt_tokens, output_tokens) tuples
        where arrival_time is relative to the first request (in seconds, float)

    CSV Format:
        TIMESTAMP,ContextTokens,GeneratedTokens
        2023-11-16 18:15:46.6805900,374,44
        ...
    """
    trace_data = []
    first_timestamp = None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            # Apply max_requests filter
            if max_requests is not None and i >= max_requests:
                break

            # Parse timestamp
            timestamp_str = row['TIMESTAMP']
            # Truncate fractional seconds to 6 digits (microseconds) for Python 3.10 compatibility
            if '.' in timestamp_str:
                parts = timestamp_str.split('.')
                parts[1] = parts[1][:6]
                timestamp_str = '.'.join(parts)
            timestamp = datetime.fromisoformat(timestamp_str)

            # Set first timestamp as reference
            if first_timestamp is None:
                first_timestamp = timestamp

            # Calculate relative arrival time in seconds (float for ms precision)
            arrival_time = (timestamp - first_timestamp).total_seconds()

            # Parse token counts
            context_tokens = int(row['ContextTokens'])
            generated_tokens = int(row['GeneratedTokens'])

            # Apply filters
            if max_context_tokens is not None and context_tokens > max_context_tokens:
                continue
            if max_generated_tokens is not None and generated_tokens > max_generated_tokens:
                continue
            if start_time is not None and arrival_time < start_time:
                continue
            if end_time is not None and arrival_time > end_time:
                continue

            trace_data.append((arrival_time, context_tokens, generated_tokens))

    # Print statistics
    if trace_data:
        arrival_times = [t[0] for t in trace_data]
        context_tokens = [t[1] for t in trace_data]
        generated_tokens = [t[2] for t in trace_data]

        print("\n" + "=" * 60)
        print(" " * 15 + "Azure Trace Dataset Loaded")
        print("=" * 60)
        print(f"{'Total requests:':<40} {len(trace_data)}")
        print(f"{'Time range (s):':<40} {min(arrival_times):.3f} - {max(arrival_times):.3f}")
        print(f"{'Duration (s):':<40} {max(arrival_times) - min(arrival_times):.3f}")
        print(f"{'Average request rate (req/s):':<40} {len(trace_data) / max(arrival_times) if max(arrival_times) > 0 else 0:.2f}")
        print()
        print(f"{'Context tokens - Min:':<40} {min(context_tokens)}")
        print(f"{'Context tokens - Max:':<40} {max(context_tokens)}")
        print(f"{'Context tokens - Mean:':<40} {np.mean(context_tokens):.1f}")
        print(f"{'Context tokens - Median (P50):':<40} {np.percentile(context_tokens, 50):.1f}")
        print(f"{'Context tokens - P99:':<40} {np.percentile(context_tokens, 99):.1f}")
        print()
        print(f"{'Generated tokens - Min:':<40} {min(generated_tokens)}")
        print(f"{'Generated tokens - Max:':<40} {max(generated_tokens)}")
        print(f"{'Generated tokens - Mean:':<40} {np.mean(generated_tokens):.1f}")
        print(f"{'Generated tokens - Median (P50):':<40} {np.percentile(generated_tokens, 50):.1f}")
        print(f"{'Generated tokens - P99:':<40} {np.percentile(generated_tokens, 99):.1f}")
        print("=" * 60 + "\n")

    return trace_data


# ============================================================================
# Section 2: Trace-based Request Generation
# ============================================================================

def generate_requests_from_trace(
    trace_data: List[Tuple[float, int, int]],
    model_name: str,
    seed: int = 0,
    ignore_eos: bool = True
) -> List[Tuple[float, RequestInput]]:
    """
    Generate RequestInput objects from trace data.

    Args:
        trace_data: List of (arrival_time, prompt_tokens, output_tokens)
        model_name: Model name for tokenizer
        seed: Random seed for prompt generation
        ignore_eos: Whether to ignore EOS token during generation

    Returns:
        List of (arrival_time, RequestInput) tuples
    """
    # Set random seed
    np.random.seed(seed)

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size

    requests = []

    for i, (arrival_time, prompt_tokens, output_tokens) in enumerate(trace_data):
        # Generate random token IDs for the prompt
        offset = np.random.randint(0, vocab_size)
        token_ids = ((offset + i + np.arange(prompt_tokens)) % vocab_size).tolist()

        # Decode to text
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Create request input
        request_input = RequestInput(
            prompt=prompt,
            prompt_len=prompt_tokens,
            expected_output_len=output_tokens,
            model=model_name,
            ignore_eos=ignore_eos
        )

        requests.append((arrival_time, request_input))

    return requests


# ============================================================================
# Section 3: Trace Replay Benchmark
# ============================================================================

async def run_trace_replay_benchmark(
    global_server,
    trace_requests: List[Tuple[float, RequestInput]],
    time_scale: float = 1.0,
    percentiles: Optional[List[float]] = None,
    disable_tqdm: bool = False,
    save_trace_path: Optional[str] = None
) -> BenchmarkMetrics:
    """
    Run trace replay benchmark by sending requests according to their arrival times.

    Args:
        global_server: GlobalServer instance
        trace_requests: List of (arrival_time, RequestInput) tuples
        time_scale: Time scale multiplier (1.0 = original, 0.5 = 2x faster, 2.0 = 2x slower)
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 99])
        disable_tqdm: Whether to disable progress bar
        save_trace_path: Path to save detailed request trace CSV (optional)

    Returns:
        BenchmarkMetrics object with results
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 99]

    # Print header
    if not disable_tqdm:
        print("\n" + "=" * 60)
        print(" " * 15 + "Starting Trace Replay Benchmark")
        print("=" * 60)
        print(f"{'Total requests:':<40} {len(trace_requests)}")
        print(f"{'Time scale:':<40} {time_scale}x")
        if trace_requests:
            original_duration = trace_requests[-1][0] - trace_requests[0][0]
            scaled_duration = original_duration * time_scale
            print(f"{'Original trace duration (s):':<40} {original_duration:.3f}")
            print(f"{'Scaled duration (s):':<40} {scaled_duration:.3f}")
        print("=" * 60 + "\n")

    # Start benchmark timer
    benchmark_start = time.time()

    # Prepare progress tracking
    pbar = None
    if not disable_tqdm:
        from tqdm.asyncio import tqdm
        pbar = tqdm(total=len(trace_requests), desc="Benchmark progress")

    # Create tasks for all requests
    tasks = []
    request_inputs = []

    # Helper function to send request at scheduled time
    async def send_request_at_time(req_input, delay):
        # Wait until the scheduled time
        await asyncio.sleep(delay)
        # Record arrival time (when request is sent)
        arrival_time_ts = time.time()
        # Send request and wait for completion
        request = await global_server.add_request_and_wait(req_input)
        # Record completion time
        completion_time_ts = time.time()
        if pbar:
            pbar.update(1)
        return (request, arrival_time_ts, completion_time_ts)

    # Schedule all requests based on their arrival times
    for arrival_time, request_input in trace_requests:
        request_inputs.append(request_input)

        # Scale the arrival time
        scaled_arrival_time = arrival_time * time_scale

        # Create async task
        task = asyncio.create_task(send_request_at_time(request_input, scaled_arrival_time))
        tasks.append(task)

    # Wait for all requests to complete
    request_datas = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    # Calculate total duration
    benchmark_duration = time.time() - benchmark_start

    # Extract requests for metrics calculation
    requests = [data[0] for data in request_datas]

    # Calculate metrics
    if not disable_tqdm:
        print("\nCalculating metrics...")

    metrics = calculate_benchmark_metrics(
        requests, request_inputs, benchmark_duration, percentiles
    )

    # Save request trace if path is provided
    if save_trace_path:
        save_request_trace(request_datas, save_trace_path)
        if not disable_tqdm:
            print(f"Request trace saved to: {save_trace_path}")

    return metrics


def save_request_trace(
    request_datas: List[Tuple[Request, float, float]],
    output_path: str
) -> None:
    """
    Save detailed request trace to CSV file for post-analysis.

    Args:
        request_datas: List of (request, arrival_time, completion_time) tuples
        output_path: Path to save the CSV file

    CSV Format:
        RequestID,ArrivalTime,CompletionTime,InputTokens,OutputTokens,Latency,QueueingDelay,TTFT,TPOT,Success
        - ArrivalTime, CompletionTime: Unix timestamps (float, seconds)
        - QueueingDelay: (CompletionTime - ArrivalTime) - Latency
        - TTFT: Time To First Token (seconds)
        - TPOT: Time Per Output Token (seconds), calculated as (Latency - TTFT) / (OutputTokens - 1)
              Excludes first token, same calculation as benchmark_utils.py
    """
    import csv as csv_module
    from pathlib import Path

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv_module.writer(f)

        # Write header (CamelCase for column names)
        writer.writerow([
            'RequestID',
            'ArrivalTime',
            'CompletionTime',
            'InputTokens',
            'OutputTokens',
            'Latency',
            'TTFT',
            'TPOT',
            'Success'
        ])

        # Write data rows
        for request, arrival_time_ts, completion_time_ts in request_datas:
            if request.output:
                # Calculate queueing delay
                total_time = completion_time_ts - arrival_time_ts
                latency = request.output.latency

                # Get TTFT
                ttft = request.output.ttft

                # Calculate TPOT (time per output token, excluding first token)
                # Same calculation as in benchmark_utils.py
                if request.output.output_tokens > 1:
                    latency_minus_ttft = latency - ttft
                    tpot = latency_minus_ttft / (request.output.output_tokens - 1)
                else:
                    tpot = 0.0

                writer.writerow([
                    request.request_id,
                    f"{arrival_time_ts:.6f}",
                    f"{completion_time_ts:.6f}",
                    request.input.prompt_len,
                    request.output.output_tokens,
                    f"{latency:.6f}",
                    f"{ttft:.6f}",
                    f"{tpot:.6f}",
                    request.output.success
                ])
            else:
                # Request failed before completion
                total_time = completion_time_ts - arrival_time_ts
                writer.writerow([
                    request.request_id,
                    f"{arrival_time_ts:.6f}",
                    f"{completion_time_ts:.6f}",
                    request.input.prompt_len,
                    0,
                    f"{total_time:.6f}",
                    0.0,
                    0.0,
                    False
                ])


# Unittest
if __name__ == "__main__":
    from benchmark_utils import ARTIFACT_EVALUATION_DIR, DEFAULT_DATASET_PATH

    # 원본 데이터
    datas = load_azure_trace(
        csv_path=os.path.join(ARTIFACT_EVALUATION_DIR, "Datasets", "AzureLLMInferenceTrace_conv.csv"),
    )

    # pruned 데이터
    pruned_datas = load_azure_trace(
        csv_path=DEFAULT_DATASET_PATH,
    )