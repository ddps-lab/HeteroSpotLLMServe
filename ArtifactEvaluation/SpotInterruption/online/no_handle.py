"""
Spot interruption benchmark using on-demand instances with real Azure trace.
"""
import asyncio
import logging
import concurrent.futures
import sys
import os
import time
from typing import Dict, List, Tuple
from nodes import *

# Add GlobalServer to path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, ".git")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "GlobalServer"))
del _d

from global_server import GlobalServer
from benchmark_utils import print_benchmark_results, run_trace_benchmark, DEFAULT_DATASET_PATH

logger = logging.getLogger(__name__)


async def run_benchmark(
    global_server: GlobalServer,
    dataset_path: str,
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
    return await run_trace_benchmark(
        global_server=global_server,
        dataset_path=dataset_path,
        trace_output_prefix="spotinterruption_no_handle",
        num_requests=num_requests,
        time_scale=time_scale,
        model_name=model_name,
        percentiles=percentiles,
        disable_tqdm=disable_tqdm,
        run_initial_test=run_initial_test,
        test_requests_per_pipeline=test_requests_per_pipeline,
        start_time=start_time,
        end_time=end_time,
    )


async def main():
    """Test node switching functionality."""
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    global_server = GlobalServer(request_handler_mode="re-routing") # default is migration
    model_name = "meta-llama/Llama-3.1-70B-Instruct"

    tasks = []

    # Create pipeline in background
    async def create_pipeline_async(config:Dict, node_layer_mapping:List[Tuple[str, int]], throughput:int):
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

    # Our Pipeline 1
    pipeline_1_stage_0_node_ip = spot_g6_12xlarge_node_ip_1
    pipeline_1_stage_1_node_ip = spot_g6_12xlarge_node_ip_2
    pipeline_1_stage_2_node_ip = on_demand_g6_12xlarge_node_ip_3
    pipeline_1_stage_3_node_ip = spot_g6e_xlarge_node_ip_1
    pipeline_1_stage_4_node_ip = spot_g6e_xlarge_node_ip_2
    pipeline_1_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "20,20,20,10,10",
        "parallel_strategy": [4,4,4,1,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 27549,
        "max_batch_size": 442,
    }
    estimated_throughput_1 = 4.23
    node_layer_mapping_1 = [
        (pipeline_1_stage_0_node_ip, 20),
        (pipeline_1_stage_1_node_ip, 20),
        (pipeline_1_stage_2_node_ip, 20),
        (pipeline_1_stage_3_node_ip, 10),
        (pipeline_1_stage_4_node_ip, 10),
    ]

    # Pipeline 2
    pipeline_2_stage_0_node_ip = spot_g6e_xlarge_node_ip_3
    pipeline_2_stage_1_node_ip = spot_g5_12xlarge_node_ip_1
    pipeline_2_stage_2_node_ip = spot_g5_12xlarge_node_ip_2
    pipeline_2_stage_3_node_ip = spot_g6e_xlarge_node_ip_4
    pipeline_2_config = {
        "model_name": model_name,
        "total_num_layers": 80,
        "gpu_memory_utilization": 0.85,
        "pp_layer_partition": "13,28,28,11",
        "parallel_strategy": [1,4,4,1],
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "model_source": "s3",
        "s3_path": f"s3://hetero-spot-llm-serve-models/{model_name}",
        "num_gpu_blocks": 15049,
        "max_batch_size": 241,
    }
    estimated_throughput_2 = 3.49
    node_layer_mapping_2 = [
        (pipeline_2_stage_0_node_ip, 13),
        (pipeline_2_stage_1_node_ip, 28),
        (pipeline_2_stage_2_node_ip, 28),
        (pipeline_2_stage_3_node_ip, 11),
    ]
    
    # Start pipeline creation
    pipeline_task_1 = asyncio.create_task(create_pipeline_async(pipeline_1_config, node_layer_mapping_1, estimated_throughput_1))
    pipeline_task_2 = asyncio.create_task(create_pipeline_async(pipeline_2_config, node_layer_mapping_2, estimated_throughput_2))
    tasks.append(pipeline_task_1)
    tasks.append(pipeline_task_2)
    await asyncio.gather(pipeline_task_1, pipeline_task_2)
    
    # Start the global server in the background
    server_task = asyncio.create_task(global_server.run_global_server())
    tasks.append(server_task)

    # Schedule stop-and-restart (optional - comment out to disable)
    async def stop_and_restart_after_delay(event_time: float = 0, old_node_ips: List[str] = None, new_node_ips: List[str] = None):
        """Stop old node and create new pipeline after a delay"""
        nonlocal node_layer_mapping_1, node_layer_mapping_2

        await asyncio.sleep(event_time)
        try:
            logger.info(f"Starting stop-and-restart test: stopping {old_node_ips}, then creating with {new_node_ips}")

            # Total time measurement start
            total_start_time = time.time()

            # 1. Stop the pipeline containing old_node_ip
            # Stop time measurement
            stop_start_time = time.time()

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    global_server.stop_nodes,
                    old_node_ips
                )

            stop_end_time = time.time()
            stop_duration = stop_end_time - stop_start_time

            logger.info(f"Pipeline with {old_node_ips} stopped (was at index {new_node_ips})")
            logger.info(f"⏱️  Stop Duration: {stop_duration:.2f} seconds")
            
            pipeline_1_nodes = [] # (old, new)
            pipeline_2_nodes = [] # (old, new)

            for node_ip, _ in node_layer_mapping_1:
                for i, old_ip in enumerate(old_node_ips):
                    if node_ip == old_ip:
                        pipeline_1_nodes.append((old_ip, new_node_ips[i]))
            
            for node_ip, _ in node_layer_mapping_2:
                for i, old_ip in enumerate(old_node_ips):
                    if node_ip == old_ip:
                        pipeline_2_nodes.append((old_ip, new_node_ips[i]))
            
            # Create new node_layer_mapping
            new_node_layer_mapping_1 = []
            for node_ip, layers in node_layer_mapping_1:
                if node_ip in old_node_ips:
                    for old_ip, new_ip in pipeline_1_nodes:
                        if node_ip == old_ip:
                            new_node_layer_mapping_1.append((new_ip, layers))
                            break
                else:
                    new_node_layer_mapping_1.append((node_ip, layers))
            new_node_layer_mapping_2 = []
            for node_ip, layers in node_layer_mapping_2:
                if node_ip in old_node_ips:
                    for old_ip, new_ip in pipeline_2_nodes:
                        if node_ip == old_ip:
                            new_node_layer_mapping_2.append((new_ip, layers))
                            break
                else:
                    new_node_layer_mapping_2.append((node_ip, layers))

            node_layer_mapping_1 = new_node_layer_mapping_1
            node_layer_mapping_2 = new_node_layer_mapping_2

            # 2. Wait a bit for clean shutdown
            await asyncio.sleep(5)

            # 3. Create new pipeline with new_node_ip
            # Create time measurement
            create_start_time = time.time()
            # Reuse create_pipeline_async function
            pipeline_tasks = []
            if len(pipeline_1_nodes) > 0:
                logger.info(f"Creating new pipeline for Pipeline 1 with nodes: {new_node_layer_mapping_1}")
                logger.info(f"pipeline_1_config: {pipeline_1_config}")
                pipeline_tasks.append(
                    asyncio.create_task(
                        create_pipeline_async(pipeline_1_config, new_node_layer_mapping_1, estimated_throughput_1)
                    )
                )
            if len(pipeline_2_nodes) > 0:
                logger.info(f"Creating new pipeline for Pipeline 2 with nodes: {new_node_layer_mapping_2}")
                logger.info(f"pipeline_2_config: {pipeline_2_config}")
                pipeline_tasks.append(
                    asyncio.create_task(
                        create_pipeline_async(pipeline_2_config, new_node_layer_mapping_2, estimated_throughput_2)
                    )
                )
            await asyncio.gather(*pipeline_tasks)

            create_end_time = time.time()
            create_duration = create_end_time - create_start_time

            logger.info(f"New pipeline with {new_node_ips} created successfully")
            logger.info(f"⏱️  Create Duration: {create_duration:.2f} seconds")

            # Total time logging
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time

            logger.info(f"⏱️  Total Stop-and-Restart Duration: {total_duration:.2f} seconds")
            logger.info(f"    - Stop: {stop_duration:.2f}s")
            logger.info(f"    - Wait: 1.00s")
            logger.info(f"    - Create: {create_duration:.2f}s")

        except Exception as e:
            logger.error(f"Stop-and-restart test failed: {e}")


    # Event 1: 
    # 시작 5분 후 1개의 g6.12xlarge 와 2개의 g5.12xlarge spot interruption 발생
    event_1_time = 5 * 60  # 5 minutes
    tasks.append(asyncio.create_task(
        stop_and_restart_after_delay(
            event_time=event_1_time,
            old_node_ips=[
                spot_g6_12xlarge_node_ip_2,
                spot_g5_12xlarge_node_ip_1,
                spot_g5_12xlarge_node_ip_2,
            ],
            new_node_ips=[
                on_demand_g6_12xlarge_node_ip_2,
                on_demand_g5_12xlarge_node_ip_1,
                on_demand_g5_12xlarge_node_ip_2,
            ]
        )
    ))

    # Event 2:
    # 시작 15분 후 4개의 g6e.xlarge Spot Interruption 과 2개의 g6.12xlarge Spot 복구
    event_2_time = 15 * 60  # 15 minutes
    tasks.append(asyncio.create_task(
        stop_and_restart_after_delay(
            event_time=event_2_time,
            old_node_ips=[
                on_demand_g6_12xlarge_node_ip_2,
                on_demand_g6_12xlarge_node_ip_3,
                spot_g6e_xlarge_node_ip_1,
                spot_g6e_xlarge_node_ip_2,
                spot_g6e_xlarge_node_ip_3,
                spot_g6e_xlarge_node_ip_4,
            ],
            new_node_ips=[
                spot_g6_12xlarge_node_ip_2,
                spot_g6_12xlarge_node_ip_3,
                on_demand_g6e_xlarge_node_ip_1,
                on_demand_g6e_xlarge_node_ip_2,
                on_demand_g6e_xlarge_node_ip_3,
                on_demand_g6e_xlarge_node_ip_4,
            ]
        )
    ))

    # Event 3:
    # 시작 25분 후 1개의 g5.12xlarge Spot 복구
    event_3_time = 25 * 60  # 25 minutes
    tasks.append(asyncio.create_task(
        stop_and_restart_after_delay(
            event_time=event_3_time,
            old_node_ips=[
                on_demand_g5_12xlarge_node_ip_1,
            ],
            new_node_ips=[
                spot_g5_12xlarge_node_ip_1,
            ]
        )
    ))

    # Event 4:
    # 시작 35분 후 4개의 g6e.xlarge Spot 복구
    event_4_time = 35 * 60  # 35 minutes
    tasks.append(asyncio.create_task(
        stop_and_restart_after_delay(
            event_time=event_4_time,
            old_node_ips=[
                on_demand_g6e_xlarge_node_ip_1,
                on_demand_g6e_xlarge_node_ip_2,
                on_demand_g6e_xlarge_node_ip_3,
                on_demand_g6e_xlarge_node_ip_4,
            ],
            new_node_ips=[
                spot_g6e_xlarge_node_ip_1,
                spot_g6e_xlarge_node_ip_2,
                spot_g6e_xlarge_node_ip_3,
                spot_g6e_xlarge_node_ip_4,
            ]
        )
    ))

    # Event 5:
    # 시작 45분 후 1개의 g5.12xlarge Spot 복구
    event_5_time = 45 * 60  # 45 minutes
    tasks.append(asyncio.create_task(
        stop_and_restart_after_delay(
            event_time=event_5_time,
            old_node_ips=[
                on_demand_g5_12xlarge_node_ip_2,
            ],
            new_node_ips=[
                spot_g5_12xlarge_node_ip_2,
            ]
        )
    ))


    try:
        # Set dataset path
        dataset_path = DEFAULT_DATASET_PATH

        start_time = 0      # Start from beginning
        end_time = 20 * 60  # Run for 20 minutes

        # Run benchmark using helper function
        metrics = await run_benchmark(
            global_server,
            dataset_path=dataset_path,
            num_requests=None,  # Use all requests from trace
            time_scale=3.0,  # Original trace speed (0.0 = Offline, 1.0 = original speed)
            model_name=model_name,
            percentiles=[10, 25, 50, 75, 90, 95],
            disable_tqdm=False,  # Show progress bar
            run_initial_test=True,
            test_requests_per_pipeline=0,
            start_time=start_time,
            end_time=end_time
        )

        # Print results
        print_benchmark_results(metrics)
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
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
    asyncio.run(main())