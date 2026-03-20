"""
Benchmark utilities for GlobalServer testing.
Contains metrics calculation and display functions.
"""
import asyncio
import os
import time
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm.asyncio import tqdm

from request_handler import Request, RequestInput


def find_project_root(marker=".git"):
    """Find the project root directory by searching for a marker (e.g., .git)."""
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"Project root not found (marker: {marker})")
        current = parent


PROJECT_ROOT = find_project_root()
GLOBAL_SERVER_DIR = os.path.join(PROJECT_ROOT, "GlobalServer")
ARTIFACT_EVALUATION_DIR = os.path.join(PROJECT_ROOT, "ArtifactEvaluation")
DEFAULT_DATASET_PATH = os.path.join(
    ARTIFACT_EVALUATION_DIR, "Datasets",
    "AzureLLMInferenceConvTrace_pruned_2048.csv"
)


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark run."""
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    mean_e2el_ms: float
    median_e2el_ms: float
    benchmark_duration: float
    # Store percentiles as lists of tuples (percentile, value)
    percentiles_ttft_ms: List[Tuple[float, float]] = None
    percentiles_tpot_ms: List[Tuple[float, float]] = None
    percentiles_itl_ms: List[Tuple[float, float]] = None
    percentiles_e2el_ms: List[Tuple[float, float]] = None


def calculate_benchmark_metrics(
    requests: List[Request],
    request_inputs: List[RequestInput],
    benchmark_duration: float,
    percentiles: List[float] = None
) -> BenchmarkMetrics:
    """
    Calculate benchmark metrics from completed requests.
    
    Args:
        requests: List of completed Request objects
        request_inputs: Original request inputs
        benchmark_duration: Total benchmark duration in seconds
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 99])
        
    Returns:
        BenchmarkMetrics object with calculated values
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 99]
    completed = 0
    total_input = 0
    total_output = 0
    ttfts = []  # Time to first token (seconds)
    tpots = []  # Time per output token (seconds)
    itls = []   # Inter-token latencies (seconds)
    e2els = []  # End-to-end latencies (seconds)
    
    for i, request in enumerate(requests):
        if request and request.output and request.output.success:
            completed += 1
            
            # Input/output tokens
            total_input += request.output.prompt_len
            total_output += request.output.output_tokens
            
            # TTFT (already in seconds)
            ttfts.append(request.output.ttft)
            
            # E2E latency (seconds)
            e2e_latency = request.output.latency
            e2els.append(e2e_latency)
            
            # TPOT calculation (time per output token, excluding first token)
            if request.output.output_tokens > 1:
                latency_minus_ttft = e2e_latency - request.output.ttft
                tpot = latency_minus_ttft / (request.output.output_tokens - 1)
                tpots.append(tpot)
            
            # ITL (inter-token latencies, already in seconds)
            if request.output.itl:
                itls.extend(request.output.itl)
    
    # Calculate throughput
    request_throughput = completed / benchmark_duration if benchmark_duration > 0 else 0
    output_throughput = total_output / benchmark_duration if benchmark_duration > 0 else 0
    total_token_throughput = (total_input + total_output) / benchmark_duration if benchmark_duration > 0 else 0
    
    # Calculate percentiles for each metric
    def calculate_percentiles_for_metric(data):
        if not data:
            return [(p, 0) for p in percentiles]
        return [(p, np.percentile(data, p)) for p in percentiles]
    
    ttft_percentiles = calculate_percentiles_for_metric(ttfts)
    tpot_percentiles = calculate_percentiles_for_metric(tpots)
    itl_percentiles = calculate_percentiles_for_metric(itls)
    e2el_percentiles = calculate_percentiles_for_metric(e2els)
    
    return BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=request_throughput,
        output_throughput=output_throughput,
        total_token_throughput=total_token_throughput,
        # Mean and median metrics (convert to milliseconds)
        mean_ttft_ms=np.mean(ttfts) * 1000 if ttfts else 0,
        median_ttft_ms=np.median(ttfts) * 1000 if ttfts else 0,
        mean_tpot_ms=np.mean(tpots) * 1000 if tpots else 0,
        median_tpot_ms=np.median(tpots) * 1000 if tpots else 0,
        mean_itl_ms=np.mean(itls) * 1000 if itls else 0,
        median_itl_ms=np.median(itls) * 1000 if itls else 0,
        mean_e2el_ms=np.mean(e2els) * 1000 if e2els else 0,
        median_e2el_ms=np.median(e2els) * 1000 if e2els else 0,
        benchmark_duration=benchmark_duration,
        # Percentile metrics (convert to milliseconds)
        percentiles_ttft_ms=[(p, v * 1000) for p, v in ttft_percentiles],
        percentiles_tpot_ms=[(p, v * 1000) for p, v in tpot_percentiles],
        percentiles_itl_ms=[(p, v * 1000) for p, v in itl_percentiles],
        percentiles_e2el_ms=[(p, v * 1000) for p, v in e2el_percentiles]
    )


def print_benchmark_results(metrics: BenchmarkMetrics):
    """Print benchmark results in a formatted manner similar to benchmark_serving.py."""
    print("\n" + "=" * 50)
    print(" " * 12 + "Serving Benchmark Result")
    print("=" * 50)
    
    # Basic statistics
    print(f"{'Successful requests:':<40} {metrics.completed:<10}")
    print(f"{'Benchmark duration (s):':<40} {metrics.benchmark_duration:<10.2f}")
    print(f"{'Total input tokens:':<40} {metrics.total_input:<10}")
    print(f"{'Total generated tokens:':<40} {metrics.total_output:<10}")
    print(f"{'Request throughput (req/s):':<40} {metrics.request_throughput:<10.2f}")
    print(f"{'Output token throughput (tok/s):':<40} {metrics.output_throughput:<10.2f}")
    print(f"{'Total Token throughput (tok/s):':<40} {metrics.total_token_throughput:<10.2f}")
    
    # End-to-end Latency
    print("-" * 16 + "End-to-end Latency" + "-" * 16)
    print(f"{'Mean E2EL (ms):':<40} {metrics.mean_e2el_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40} {metrics.median_e2el_ms:<10.2f}")
    if metrics.percentiles_e2el_ms:
        for p, v in metrics.percentiles_e2el_ms:
            print(f"{f'P{int(p)} E2EL (ms):':<40} {v:<10.2f}")
    
    # Time to First Token
    print("-" * 15 + "Time to First Token" + "-" * 16)
    print(f"{'Mean TTFT (ms):':<40} {metrics.mean_ttft_ms:<10.2f}")
    print(f"{'Median TTFT (ms):':<40} {metrics.median_ttft_ms:<10.2f}")
    if metrics.percentiles_ttft_ms:
        for p, v in metrics.percentiles_ttft_ms:
            print(f"{f'P{int(p)} TTFT (ms):':<40} {v:<10.2f}")
    
    # Time per Output Token
    print("-" * 5 + "Time per Output Token (excl. 1st token)" + "-" * 6)
    print(f"{'Mean TPOT (ms):':<40} {metrics.mean_tpot_ms:<10.2f}")
    print(f"{'Median TPOT (ms):':<40} {metrics.median_tpot_ms:<10.2f}")
    if metrics.percentiles_tpot_ms:
        for p, v in metrics.percentiles_tpot_ms:
            print(f"{f'P{int(p)} TPOT (ms):':<40} {v:<10.2f}")
    
    # Inter-token Latency
    print("-" * 15 + "Inter-token Latency" + "-" * 16)
    print(f"{'Mean ITL (ms):':<40} {metrics.mean_itl_ms:<10.2f}")
    print(f"{'Median ITL (ms):':<40} {metrics.median_itl_ms:<10.2f}")
    if metrics.percentiles_itl_ms:
        for p, v in metrics.percentiles_itl_ms:
            print(f"{f'P{int(p)} ITL (ms):':<40} {v:<10.2f}")
    
    print("=" * 50)


async def run_benchmark_requests(
    global_server,
    request_inputs: List[RequestInput],
    request_rate: float,
    max_concurrency: Optional[int] = None,
    disable_tqdm: bool = False
) -> Tuple[List[Request], float]:
    """
    Send requests and wait for completion with progress tracking.
    
    Args:
        global_server: The GlobalServer instance
        request_inputs: List of request inputs to send
        request_rate: Requests per second (use float('inf') for no rate limit)
        max_concurrency: Maximum number of concurrent requests (None for no limit)
        disable_tqdm: Whether to disable progress bar
        
    Returns:
        Tuple of (List of Request objects, duration)
    """
    start_time = time.time()
    pbar = None if disable_tqdm else tqdm(total=len(request_inputs), desc="Benchmark progress")
    
    # Create semaphore if needed
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    
    async def limited_request_func(req_input):
        """Send request with optional concurrency limit and wait for completion."""
        if semaphore is None:
            request = await global_server.add_request_and_wait(req_input)
        else:
            async with semaphore:
                request = await global_server.add_request_and_wait(req_input)
        
        if pbar:
            pbar.update(1)
        return request
    
    # Create tasks based on request rate
    tasks = []
    
    if request_rate == float('inf'):
        # No rate limit - create all tasks at once
        for req_input in request_inputs:
            task = asyncio.create_task(limited_request_func(req_input))
            tasks.append(task)
    else:
        # Rate-limited - create tasks with delays
        interval = 1.0 / request_rate
        for i, req_input in enumerate(request_inputs):
            task = asyncio.create_task(limited_request_func(req_input))
            tasks.append(task)
            
            # Wait before creating next task
            if i < len(request_inputs) - 1:
                await asyncio.sleep(interval)
    
    # Wait for all tasks to complete
    requests = await asyncio.gather(*tasks)
    
    if pbar:
        pbar.close()
    
    duration = time.time() - start_time
    return requests, duration


async def run_benchmark_requests_with_trace(
    global_server,
    request_inputs: List[RequestInput],
    request_rate: float,
    max_concurrency: Optional[int] = None,
    disable_tqdm: bool = False,
    save_trace_path: Optional[str] = None
) -> Tuple[List[Request], float]:
    """
    Send requests and wait for completion with progress tracking and trace saving.

    Args:
        global_server: The GlobalServer instance
        request_inputs: List of request inputs to send
        request_rate: Requests per second (use float('inf') for no rate limit)
        max_concurrency: Maximum number of concurrent requests (None for no limit)
        disable_tqdm: Whether to disable progress bar
        save_trace_path: Optional path to save request trace CSV

    Returns:
        Tuple of (List of Request objects, duration)
    """
    start_time = time.time()
    pbar = None if disable_tqdm else tqdm(total=len(request_inputs), desc="Benchmark progress")

    # Create semaphore if needed
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(req_input):
        """Send request with optional concurrency limit and wait for completion."""
        arrival_time_ts = time.time()  # Record arrival time

        if semaphore is None:
            request = await global_server.add_request_and_wait(req_input)
        else:
            async with semaphore:
                request = await global_server.add_request_and_wait(req_input)

        completion_time_ts = time.time()  # Record completion time

        if pbar:
            pbar.update(1)
        return (request, arrival_time_ts, completion_time_ts)

    # Create tasks based on request rate
    tasks = []

    if request_rate == float('inf'):
        # No rate limit - create all tasks at once
        for req_input in request_inputs:
            task = asyncio.create_task(limited_request_func(req_input))
            tasks.append(task)
    else:
        # Rate-limited - create tasks with delays
        interval = 1.0 / request_rate
        for i, req_input in enumerate(request_inputs):
            task = asyncio.create_task(limited_request_func(req_input))
            tasks.append(task)

            # Wait before creating next task
            if i < len(request_inputs) - 1:
                await asyncio.sleep(interval)

    # Wait for all tasks to complete
    request_datas = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    duration = time.time() - start_time

    # Save trace if path is provided
    if save_trace_path:
        from evaluation_utils import save_request_trace
        save_request_trace(request_datas, save_trace_path)
        if not disable_tqdm:
            print(f"Request trace saved to: {save_trace_path}")

    # Extract requests for return (maintain compatibility)
    requests = [data[0] for data in request_datas]

    return requests, duration


async def run_trace_benchmark(
    global_server,
    dataset_path: str,
    trace_output_prefix: str,
    trace_base_dir: str = None,
    num_requests: int = None,
    time_scale: float = 1.0,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    percentiles: List[float] = None,
    disable_tqdm: bool = False,
    run_initial_test: bool = True,
    test_requests_per_pipeline: int = 2,
    start_time: float = None,
    end_time: float = None
):
    """
    Run a trace-based benchmark test on the GlobalServer using Azure dataset.

    Args:
        global_server: The GlobalServer instance
        dataset_path: Path to Azure trace dataset CSV file
        trace_output_prefix: Prefix for trace output filename (e.g. "request_migration", "alpaserve_throughput")
        trace_base_dir: Base directory where "Trace/" subdirectory will be created
        num_requests: Maximum number of requests to load from dataset (None for all)
        time_scale: Time scale multiplier (1.0 = original, 0.5 = 2x faster, 2.0 = 2x slower)
        model_name: Model name for generating requests
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 99])
        disable_tqdm: Whether to disable progress bar
        run_initial_test: Whether to run initial test requests
        test_requests_per_pipeline: Number of test requests per pipeline
        start_time: Start time in seconds from the first request (None for no lower bound)
        end_time: End time in seconds from the first request (None for no upper bound)

    Returns:
        BenchmarkMetrics object with results
    """
    from request_handler import generate_random_requests
    from evaluation_utils import (
        load_azure_trace,
        generate_requests_from_trace,
        run_trace_replay_benchmark,
    )

    if trace_base_dir is None:
        trace_base_dir = ARTIFACT_EVALUATION_DIR

    # Run initial test if requested
    if run_initial_test:
        num_pipelines = len(global_server.cluster.pipelines)
        print(f"\nRunning initial test on {num_pipelines} pipeline(s)...")

        # Generate test requests with fixed length for stability
        test_count = test_requests_per_pipeline * num_pipelines
        test_inputs = generate_random_requests(
            num_prompts=test_count,
            input_len=512,
            output_len=128,
            model_name=model_name,
            ignore_eos=True
        )

        # Send test requests
        print(f"Sending {test_count} test requests ({test_requests_per_pipeline} per pipeline)...")
        test_requests = []
        for test_input in test_inputs:
            request = await global_server.add_request_and_wait(test_input)
            test_requests.append(request)

        # Check results
        failed_count = 0
        for i, request in enumerate(test_requests):
            if not (request.output and request.output.success):
                failed_count += 1
                error_msg = request.output.error if request.output else "No output"
                print(f"  Test request {i+1} failed: {error_msg}")

        if failed_count > 0:
            raise ValueError(
                f"Initial test failed - {failed_count}/{test_count} requests failed. "
                "Please check pipeline configuration and server status."
            )
        else:
            print(f"Initial test completed successfully - all {test_count} requests succeeded!")
            print("Starting trace benchmark...\n")

    # Load Azure trace dataset
    print(f"Loading trace dataset from: {dataset_path}")
    trace_data = load_azure_trace(
        csv_path=dataset_path,
        max_requests=num_requests,
        start_time=start_time,
        end_time=end_time
    )

    if not trace_data:
        raise ValueError("No trace data loaded. Check dataset path and filters.")

    # Generate requests from trace
    print("Generating requests from trace data...")
    trace_requests = generate_requests_from_trace(
        trace_data=trace_data,
        model_name=model_name,
        seed=0,
        ignore_eos=True
    )

    print(f"Generated {len(trace_requests)} requests from trace\n")

    # Prepare trace output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    trace_dir = os.path.join(trace_base_dir, "Trace")
    os.makedirs(trace_dir, exist_ok=True)
    trace_output_path = os.path.join(trace_dir, f"{trace_output_prefix}_{timestamp}.csv")

    # Run trace replay benchmark
    metrics = await run_trace_replay_benchmark(
        global_server=global_server,
        trace_requests=trace_requests,
        time_scale=time_scale,
        percentiles=percentiles,
        disable_tqdm=disable_tqdm,
        save_trace_path=trace_output_path
    )

    return metrics


async def run_latency_benchmark(
    global_server,
    num_requests: int = 100,
    input_len: int = 1024,
    output_len: int = 128,
    request_rate: float = float('inf'),
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_concurrency: int = None,
    percentiles: List[float] = None,
    disable_tqdm: bool = False,
    run_initial_test: bool = True,
    test_requests_per_pipeline: int = 2
):
    """
    Run a benchmark test on the GlobalServer.

    Args:
        global_server: The GlobalServer instance
        num_requests: Number of requests to send
        input_len: Input token length
        output_len: Expected output token length
        request_rate: Requests per second (inf for no limit)
        model_name: Model name for generating requests
        max_concurrency: Maximum number of concurrent requests (None for no limit)
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 99])
        disable_tqdm: Whether to disable progress bar
        run_initial_test: Whether to run initial test requests
        test_requests_per_pipeline: Number of test requests per pipeline

    Returns:
        BenchmarkMetrics object with results
    """
    from request_handler import generate_random_requests

    if not disable_tqdm:
        print("\n" + "=" * 50)
        print("Starting GlobalServer Benchmark")
        print(f"  Requests: {num_requests}")
        print(f"  Input length: {input_len} tokens")
        print(f"  Output length: {output_len} tokens")
        print(f"  Request rate: {request_rate if request_rate != float('inf') else 'unlimited'} req/s")
        print(f"  Model: {model_name}")
        print("=" * 50 + "\n")

    # Generate random requests
    if not disable_tqdm:
        print("Generating requests...")
    request_inputs = generate_random_requests(
        num_prompts=num_requests,
        input_len=input_len,
        output_len=output_len,
        model_name=model_name,
        ignore_eos=True  # Ensure consistent output length
    )

    # Run initial test if requested
    if run_initial_test:
        num_pipelines = len(global_server.cluster.pipelines)
        print(f"\nRunning initial test on {num_pipelines} pipeline(s)...")

        # Generate test requests (2 per pipeline)
        test_count = test_requests_per_pipeline * num_pipelines
        test_inputs = request_inputs[:test_count] if test_count <= len(request_inputs) else generate_random_requests(
            num_prompts=test_count,
            input_len=input_len,
            output_len=output_len,
            model_name=model_name,
            ignore_eos=True
        )

        # Send test requests
        print(f"Sending {test_count} test requests ({test_requests_per_pipeline} per pipeline)...")
        test_requests = []
        for test_input in test_inputs:
            request = await global_server.add_request_and_wait(test_input)
            test_requests.append(request)

        # Check results
        failed_count = 0
        for i, request in enumerate(test_requests):
            if not (request.output and request.output.success):
                failed_count += 1
                error_msg = request.output.error if request.output else "No output"
                print(f"  Test request {i+1} failed: {error_msg}")

        if failed_count > 0:
            raise ValueError(
                f"Initial test failed - {failed_count}/{test_count} requests failed. "
                "Please check pipeline configuration and server status."
            )
        else:
            print(f"Initial test completed successfully - all {test_count} requests succeeded!")
            print("Starting main benchmark run...\n")

    # Run benchmark (send requests and wait for completion)
    requests, actual_duration = await run_benchmark_requests(
        global_server, request_inputs, request_rate, max_concurrency, disable_tqdm
    )

    # Calculate metrics
    if not disable_tqdm:
        print("\nCalculating metrics...")
    metrics = calculate_benchmark_metrics(
        requests, request_inputs, actual_duration, percentiles
    )

    return metrics
