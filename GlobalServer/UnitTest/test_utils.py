"""
Common utilities for GlobalServer unit tests.
"""
import asyncio
import logging
import time
from typing import List, Optional, Callable
from dataclasses import dataclass

from request_handler import Request, RequestInput


@dataclass
class TestMetrics:
    """Performance metrics for a completed request."""
    request_id: int
    e2e_latency: float
    ttft: float
    avg_itl: float  # Average inter-token latency in ms
    output_tokens: int
    prompt_len: int
    halted_duration: Optional[float] = None


def setup_test_logger(name: str) -> logging.Logger:
    """Set up a logger for test scripts with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    test_logger = logging.getLogger(name)
    test_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    test_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    test_logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    test_logger.propagate = False
    
    return test_logger


async def send_request_with_delay(
    global_server, 
    request_input: RequestInput, 
    index: int, 
    delay: float,
    logger: logging.Logger,
    urgent: bool = False
) -> Request:
    """Send a request after a delay.
    
    Args:
        global_server: The GlobalServer instance
        request_input: The request input to send
        index: Request index for logging
        delay: Delay in seconds before sending
        logger: Logger instance
        urgent: Whether to add to urgent queue
        
    Returns:
        The created Request object
    """
    await asyncio.sleep(delay)
    request = await global_server.add_request(request_input, urgent=urgent)
    logger.info(f"[{index}] Added request {request.request_id}")
    return request


def calculate_metrics(request: Request) -> TestMetrics:
    """Calculate performance metrics for a completed request.
    
    Args:
        request: Completed request with output
        
    Returns:
        TestMetrics object with calculated values
    """
    # Calculate e2e latency
    e2e_latency = time.time() - request.created_at
    
    # Calculate average ITL (inter-token latency)
    avg_itl = sum(request.output.itl) / len(request.output.itl) if request.output.itl else 0
    
    # Check if request was halted
    halted_duration = None
    if request.halted_at is not None and request.sended_at is not None:
        halted_duration = request.halted_at - request.sended_at
    
    return TestMetrics(
        request_id=request.request_id,
        e2e_latency=e2e_latency,
        ttft=request.output.ttft,
        avg_itl=avg_itl * 1000,  # Convert to ms
        output_tokens=request.output.output_tokens,
        prompt_len=request.output.prompt_len,
        halted_duration=halted_duration
    )


def log_request_metrics(logger: logging.Logger, index: int, metrics: TestMetrics):
    """Log performance metrics for a request.
    
    Args:
        logger: Logger instance
        index: Request index
        metrics: Calculated metrics
    """
    logger.info(f"[{index}] Request {metrics.request_id} completed!")
    logger.info(f"[{index}] Performance Metrics:")
    logger.info(f"[{index}]   - E2E Latency: {metrics.e2e_latency:.3f}s")
    logger.info(f"[{index}]   - TTFT: {metrics.ttft:.3f}s")
    logger.info(f"[{index}]   - Avg ITL (TPOT): {metrics.avg_itl:.2f}ms")
    logger.info(f"[{index}]   - Output Tokens: {metrics.output_tokens}")
    logger.info(f"[{index}]   - Prompt Len: {metrics.prompt_len}")
    
    if metrics.halted_duration is not None:
        logger.info(f"[{index}]   - Request was halted after {metrics.halted_duration:.3f}s")


async def monitor_requests_completion(
    requests: List[Request],
    logger: logging.Logger,
    timeout: int = 480,
    status_interval: int = 30,
    status_callback: Optional[Callable[[], None]] = None,
    request_tasks: Optional[List[asyncio.Task]] = None
) -> int:
    """Monitor requests until completion or timeout.
    
    Args:
        requests: List of requests to monitor
        logger: Logger instance
        timeout: Timeout in seconds
        status_interval: Interval for status updates in seconds
        status_callback: Optional callback for periodic status updates
        
    Returns:
        Number of completed requests
    """
    start_time = time.time()
    completed_count = 0
    last_status_time = 0
    num_requests = len(requests)
    
    while completed_count < num_requests:
        await asyncio.sleep(1)
        
        # Log status periodically
        elapsed = time.time() - start_time
        if elapsed - last_status_time >= status_interval:
            last_status_time = elapsed
            logger.info(f"Progress at {elapsed:.0f}s: {completed_count}/{num_requests} completed")
            
            # Call status callback if provided
            if status_callback:
                status_callback()
        
        # Check which requests are completed
        for i, req in enumerate(requests):
            if req.output and req.output.success and not hasattr(req, '_logged'):
                completed_count += 1
                req._logged = True  # Mark as logged to avoid duplicate logs
                
                # Calculate and log metrics
                metrics = calculate_metrics(req)
                log_request_metrics(logger, i, metrics)
        
        # Timeout check
        if elapsed > timeout:
            logger.warning(f"Timeout reached after {timeout}s, stopping...")
            break
    
    return completed_count


def create_request_tasks(
    global_server,
    request_inputs: List[RequestInput],
    delay_between_requests: float,
    logger: logging.Logger,
    urgent: bool = False
) -> List[asyncio.Task]:
    """Create tasks to send requests with delays, but don't wait for them.
    
    Args:
        global_server: The GlobalServer instance
        request_inputs: List of request inputs
        delay_between_requests: Delay in seconds between each request
        logger: Logger instance
        urgent: Whether to add to urgent queue
        
    Returns:
        List of tasks that will return Request objects when complete
    """
    tasks = []
    
    for i, request_input in enumerate(request_inputs):
        delay = i * delay_between_requests
        task = asyncio.create_task(
            send_request_with_delay(global_server, request_input, i, delay, logger, urgent)
        )
        tasks.append(task)
    
    return tasks



async def send_and_monitor_requests(
    global_server,
    request_inputs: List[RequestInput],
    delay_between_requests: float,
    logger: logging.Logger,
    timeout: int = 480,
    status_interval: int = 30,
    status_callback: Optional[Callable[[], None]] = None,
    urgent: bool = False
) -> int:
    """Send requests with delays and monitor them concurrently.
    
    This function starts sending requests and immediately begins monitoring,
    allowing requests to be processed as they are sent rather than waiting
    for all to be sent first.
    
    Args:
        global_server: The GlobalServer instance
        request_inputs: List of request inputs
        delay_between_requests: Delay in seconds between each request
        logger: Logger instance
        timeout: Timeout in seconds
        status_interval: Interval for status updates in seconds
        status_callback: Optional callback for periodic status updates
        urgent: Whether to add to urgent queue
        
    Returns:
        Number of completed requests
    """
    # Create tasks for sending requests
    request_tasks = create_request_tasks(
        global_server, request_inputs, delay_between_requests, logger, urgent
    )
    
    start_time = time.time()
    completed_count = 0
    last_status_time = 0
    num_requests = len(request_inputs)
    requests = []
    
    while completed_count < num_requests:
        await asyncio.sleep(1)
        
        # Collect completed request tasks
        for i, task in enumerate(request_tasks):
            if task.done() and i >= len(requests):
                try:
                    request = await task
                    requests.append(request)
                    logger.info(f"Request {i} submitted (total submitted: {len(requests)}/{num_requests})")
                except Exception as e:
                    logger.error(f"Failed to submit request {i}: {e}")
                    requests.append(None)
        
        # Log status periodically
        elapsed = time.time() - start_time
        if elapsed - last_status_time >= status_interval:
            last_status_time = elapsed
            logger.info(f"Progress at {elapsed:.0f}s: {completed_count}/{num_requests} completed, {len(requests)}/{num_requests} submitted")
            
            # Call status callback if provided
            if status_callback:
                status_callback()
        
        # Check which requests are completed
        for i, req in enumerate(requests):
            if req and req.output and req.output.success and not hasattr(req, '_logged'):
                completed_count += 1
                req._logged = True  # Mark as logged to avoid duplicate logs
                
                # Calculate and log metrics
                metrics = calculate_metrics(req)
                log_request_metrics(logger, i, metrics)
        
        # Timeout check
        if elapsed > timeout:
            logger.warning(f"Timeout reached after {timeout}s, stopping...")
            break
    
    # Cancel any remaining tasks
    for task in request_tasks:
        if not task.done():
            task.cancel()
    
    # Wait for all tasks to be cancelled properly
    if any(not task.done() for task in request_tasks):
        try:
            await asyncio.gather(*request_tasks, return_exceptions=True)
        except Exception:
            pass  # Ignore exceptions during cleanup
    
    return completed_count


def log_pipeline_status(global_server, logger: logging.Logger):
    """Log the current status of all pipelines.
    
    Args:
        global_server: The GlobalServer instance
        logger: Logger instance
    """
    logger.info("Pipeline status:")
    for i, pipeline in enumerate(global_server.cluster.pipelines):
        logger.info(f"  Pipeline {i}: ready={pipeline.is_ready}, throughput={pipeline.ideal_throughput}")
        
        # If verbose, also log node details
        if hasattr(pipeline, 'vnodes'):
            logger.info(f"    Nodes: {len(pipeline.vnodes)} nodes")
            for j, vnode in enumerate(pipeline.vnodes):
                logger.info(f"      Node {j} ({vnode.node_ip}): layers=[{vnode.layer_start_id}, {vnode.layer_end_id})")