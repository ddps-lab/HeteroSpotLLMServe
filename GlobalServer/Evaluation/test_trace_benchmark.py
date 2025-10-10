"""
Trace-based benchmark test for GlobalServer using Azure LLM Inference Conversation Dataset.
Replays real workload traces to measure system performance under realistic conditions.
"""
import asyncio
import concurrent.futures
import sys
import os
from typing import Dict, List, Tuple

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from global_server import GlobalServer
from evaluation_utils import (
    load_azure_trace,
    generate_requests_from_trace,
    run_trace_replay_benchmark
)
from benchmark_utils import print_benchmark_results
from UnitTest.test_utils import setup_test_logger


async def test_trace_benchmark():
    """Test trace-based benchmark with Azure dataset."""
    logger = setup_test_logger(__name__)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    # Dataset configuration
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "Datasets",
        "AzureLLMInferenceConvTrace_pruned_2048.csv"
    )

    # Benchmark configuration
    max_requests = 1000  # Start with a small number for testing
    time_scale = 0  # 1.0 = original speed, 0.1 = 10x faster, 10.0 = 10x slower, 0 = offline
    run_initial_test = True
    test_requests_per_pipeline = 2

    # Create GlobalServer instance
    global_server = GlobalServer()

    # Pipeline configuration
    pipeline_node_ip = "172.31.30.165"  # g6.xlarge
    pipeline_config = {
        "model_name": model_name,
        "total_num_layers": 32,
        "pp_layer_partition": "32",
        "parallel_strategy": [1],  # Single GPU
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 1741,
        "max_batch_size": 28,
    }
    estimated_throughput = 100
    node_layer_mapping = [
        (pipeline_node_ip, 32)
    ]

    # Create pipeline in background
    async def create_pipeline_async(config: Dict, node_layer_mapping: List[Tuple[str, int]], throughput: int):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                global_server.create_pipeline,
                node_layer_mapping,
                config,
                throughput
            )
        logger.info("Pipeline creation completed")

    # Start pipeline creation
    pipeline_task = asyncio.create_task(
        create_pipeline_async(pipeline_config, node_layer_mapping, estimated_throughput)
    )

    # Start global server
    server_task = asyncio.create_task(global_server.run_global_server())

    try:
        # Wait for pipeline creation to complete
        logger.info("Waiting for pipeline creation to complete...")
        await pipeline_task
        logger.info("Pipeline is ready!")

        # Run initial test if requested
        if run_initial_test:
            from request_handler import generate_random_requests
            num_pipelines = len(global_server.cluster.pipelines)
            test_count = test_requests_per_pipeline * num_pipelines

            print(f"\nRunning initial test on {num_pipelines} pipeline(s)...")
            print(f"Sending {test_count} test requests ({test_requests_per_pipeline} per pipeline)...")

            test_inputs = generate_random_requests(
                num_prompts=test_count,
                input_len=512,
                output_len=128,
                model_name=model_name,
                ignore_eos=True
            )

            test_requests = []
            for test_input in test_inputs:
                request = await global_server.add_request_and_wait(test_input)
                test_requests.append(request)

            # Check results
            failed_count = sum(
                1 for req in test_requests
                if not (req.output and req.output.success)
            )

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
            max_requests=max_requests,
        )

        if not trace_data:
            raise ValueError("No trace data loaded. Check dataset path and filters.")

        # Generate requests from trace
        print("Generating requests from trace data...")
        trace_requests = generate_requests_from_trace(
            trace_data=trace_data,
            model_name=model_name,
            seed=0,
            ignore_eos=True  # Ensure consistent output length
        )

        print(f"Generated {len(trace_requests)} requests from trace\n")

        # Run trace replay benchmark
        metrics = await run_trace_replay_benchmark(
            global_server=global_server,
            trace_requests=trace_requests,
            time_scale=time_scale,
            percentiles=[10, 25, 50, 75, 90, 99],
            disable_tqdm=False
        )

        # Print results
        print_benchmark_results(metrics)

        # Additional trace-specific analysis
        print("\n" + "=" * 60)
        print(" " * 15 + "Trace Replay Analysis")
        print("=" * 60)

        context_tokens = [t[1] for t in trace_data]
        generated_tokens = [t[2] for t in trace_data]

        print(f"{'Expected input tokens (total):':<40} {sum(context_tokens)}")
        print(f"{'Expected output tokens (total):':<40} {sum(generated_tokens)}")
        print(f"{'Actual input tokens (total):':<40} {metrics.total_input}")
        print(f"{'Actual output tokens (total):':<40} {metrics.total_output}")
        print(f"{'Success rate:':<40} {metrics.completed / len(trace_requests) * 100:.1f}%")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        server_task.cancel()
        pipeline_task.cancel()

        try:
            await asyncio.gather(server_task, pipeline_task, return_exceptions=True)
        except:
            pass

        # Stop all pipelines
        logger.info("Stopping pipelines...")
        try:
            global_server.cluster.stop_all_pipelines()
            logger.info("All pipelines stopped")
        except Exception as e:
            logger.error(f"Error stopping pipelines: {e}")


if __name__ == "__main__":
    asyncio.run(test_trace_benchmark())
