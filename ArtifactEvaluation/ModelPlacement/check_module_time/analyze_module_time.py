#!/usr/bin/env python3
"""
Analyze module loading times from withfsr and withoutfsr logs.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


# Node IP mappings
WITHFSR_NODE_MAPPING = {
    "192.168.0.126": ("g6_12xlarge", 1),
    "192.168.0.253": ("g6_12xlarge", 2),
    "192.168.0.70": ("g6_12xlarge", 3),
    "192.168.0.127": ("g5_12xlarge", 1),
    "192.168.0.134": ("g5_12xlarge", 2),
    "192.168.0.73": ("g6e_xlarge", 1),
    "192.168.0.189": ("g6e_xlarge", 2),
    "192.168.0.15": ("g6e_xlarge", 3),
    "192.168.0.65": ("g6e_xlarge", 4),
}

WITHOUTFSR_NODE_MAPPING = {
    "192.168.0.64": ("g5_12xlarge", 1),
    "192.168.0.13": ("g5_12xlarge", 2),
    "192.168.0.240": ("g6_12xlarge", 1),
    "192.168.0.206": ("g6_12xlarge", 2),
    "192.168.0.253": ("g6_12xlarge", 3),
    "192.168.0.194": ("g6e_xlarge", 1),
    "192.168.0.67": ("g6e_xlarge", 2),
    "192.168.0.225": ("g6e_xlarge", 3),
    "192.168.0.99": ("g6e_xlarge", 4),
}

ITERATIONS_10_NODE_MAPPING = {
    "192.168.0.226": ("g6_12xlarge", 1),
    "192.168.0.195": ("g6_12xlarge", 2),
    "192.168.0.127": ("g6_12xlarge", 3),
    "192.168.0.200": ("g5_12xlarge", 1),
    "192.168.0.240": ("g5_12xlarge", 2),
    "192.168.0.36": ("g6e_xlarge", 1),
    "192.168.0.143": ("g6e_xlarge", 2),
    "192.168.0.66": ("g6e_xlarge", 3),
    "192.168.0.5": ("g6e_xlarge", 4),
}

NEW_ITERATIONS_10_NODE_MAPPING = {
    "192.168.0.103": ("g6_12xlarge", 1),
    "192.168.0.248": ("g6_12xlarge", 2),
    "192.168.0.198": ("g6_12xlarge", 3),
    "192.168.0.115": ("g5_12xlarge", 1),
    "192.168.0.202": ("g5_12xlarge", 2),
    "192.168.0.8": ("g6e_xlarge", 1),
    "192.168.0.144": ("g6e_xlarge", 2),
    "192.168.0.30": ("g6e_xlarge", 3),
    "192.168.0.140": ("g6e_xlarge", 4),
}

# Pipeline to Node IP mappings
# Pipeline 1: stages 0,1,2,3,4
# Pipeline 2: stages 0,1,2,3
WITHFSR_PIPELINE_MAPPING = {
    # Pipeline 1
    "192.168.0.126": 1,  # pipeline_1_stage_0_node_ip = g6_12xlarge_node_ip_1
    "192.168.0.253": 1,  # pipeline_1_stage_1_node_ip = g6_12xlarge_node_ip_2
    "192.168.0.70": 1,   # pipeline_1_stage_2_node_ip = g6_12xlarge_node_ip_3
    "192.168.0.73": 1,   # pipeline_1_stage_3_node_ip = g6e_xlarge_node_ip_1
    "192.168.0.189": 1,  # pipeline_1_stage_4_node_ip = g6e_xlarge_node_ip_2
    # Pipeline 2
    "192.168.0.15": 2,   # pipeline_2_stage_0_node_ip = g6e_xlarge_node_ip_3
    "192.168.0.127": 2,  # pipeline_2_stage_1_node_ip = g5_12xlarge_node_ip_1
    "192.168.0.134": 2,  # pipeline_2_stage_2_node_ip = g5_12xlarge_node_ip_2
    "192.168.0.65": 2,   # pipeline_2_stage_3_node_ip = g6e_xlarge_node_ip_4
}

WITHOUTFSR_PIPELINE_MAPPING = {
    # Pipeline 1
    "192.168.0.240": 1,  # pipeline_1_stage_0_node_ip = g6_12xlarge_node_ip_1
    "192.168.0.206": 1,  # pipeline_1_stage_1_node_ip = g6_12xlarge_node_ip_2
    "192.168.0.253": 1,  # pipeline_1_stage_2_node_ip = g6_12xlarge_node_ip_3
    "192.168.0.194": 1,  # pipeline_1_stage_3_node_ip = g6e_xlarge_node_ip_1
    "192.168.0.67": 1,   # pipeline_1_stage_4_node_ip = g6e_xlarge_node_ip_2
    # Pipeline 2
    "192.168.0.225": 2,  # pipeline_2_stage_0_node_ip = g6e_xlarge_node_ip_3
    "192.168.0.64": 2,   # pipeline_2_stage_1_node_ip = g5_12xlarge_node_ip_1
    "192.168.0.13": 2,   # pipeline_2_stage_2_node_ip = g5_12xlarge_node_ip_2
    "192.168.0.99": 2,   # pipeline_2_stage_3_node_ip = g6e_xlarge_node_ip_4
}

# Stage mapping for each IP
WITHFSR_STAGE_MAPPING = {
    # Pipeline 1
    "192.168.0.126": 0,  # pipeline_1_stage_0
    "192.168.0.253": 1,  # pipeline_1_stage_1
    "192.168.0.70": 2,   # pipeline_1_stage_2
    "192.168.0.73": 3,   # pipeline_1_stage_3
    "192.168.0.189": 4,  # pipeline_1_stage_4
    # Pipeline 2
    "192.168.0.15": 0,   # pipeline_2_stage_0
    "192.168.0.127": 1,  # pipeline_2_stage_1
    "192.168.0.134": 2,  # pipeline_2_stage_2
    "192.168.0.65": 3,   # pipeline_2_stage_3
}

WITHOUTFSR_STAGE_MAPPING = {
    # Pipeline 1
    "192.168.0.240": 0,  # pipeline_1_stage_0
    "192.168.0.206": 1,  # pipeline_1_stage_1
    "192.168.0.253": 2,  # pipeline_1_stage_2
    "192.168.0.194": 3,  # pipeline_1_stage_3
    "192.168.0.67": 4,   # pipeline_1_stage_4
    # Pipeline 2
    "192.168.0.225": 0,  # pipeline_2_stage_0
    "192.168.0.64": 1,   # pipeline_2_stage_1
    "192.168.0.13": 2,   # pipeline_2_stage_2
    "192.168.0.99": 3,   # pipeline_2_stage_3
}

ITERATIONS_10_PIPELINE_MAPPING = {
    # Pipeline 1
    "192.168.0.226": 1,  # pipeline_1_stage_0_node_ip = g6_12xlarge_node_ip_1
    "192.168.0.195": 1,  # pipeline_1_stage_1_node_ip = g6_12xlarge_node_ip_2
    "192.168.0.127": 1,  # pipeline_1_stage_2_node_ip = g6_12xlarge_node_ip_3
    "192.168.0.36": 1,   # pipeline_1_stage_3_node_ip = g6e_xlarge_node_ip_1
    "192.168.0.143": 1,  # pipeline_1_stage_4_node_ip = g6e_xlarge_node_ip_2
    # Pipeline 2
    "192.168.0.66": 2,   # pipeline_2_stage_0_node_ip = g6e_xlarge_node_ip_3
    "192.168.0.200": 2,  # pipeline_2_stage_1_node_ip = g5_12xlarge_node_ip_1
    "192.168.0.240": 2,  # pipeline_2_stage_2_node_ip = g5_12xlarge_node_ip_2
    "192.168.0.5": 2,    # pipeline_2_stage_3_node_ip = g6e_xlarge_node_ip_4
}

ITERATIONS_10_STAGE_MAPPING = {
    # Pipeline 1
    "192.168.0.226": 0,  # pipeline_1_stage_0
    "192.168.0.195": 1,  # pipeline_1_stage_1
    "192.168.0.127": 2,  # pipeline_1_stage_2
    "192.168.0.36": 3,   # pipeline_1_stage_3
    "192.168.0.143": 4,  # pipeline_1_stage_4
    # Pipeline 2
    "192.168.0.66": 0,   # pipeline_2_stage_0
    "192.168.0.200": 1,  # pipeline_2_stage_1
    "192.168.0.240": 2,  # pipeline_2_stage_2
    "192.168.0.5": 3,    # pipeline_2_stage_3
}
NEW_ITERATIONS_10_PIPELINE_MAPPING = {
    # Pipeline 1
    "192.168.0.103": 1,  # pipeline_1_stage_0_node_ip = g6_12xlarge_node_ip_1
    "192.168.0.248": 1,  # pipeline_1_stage_1_node_ip = g6_12xlarge_node_ip_2
    "192.168.0.198": 1,  # pipeline_1_stage_2_node_ip = g6_12xlarge_node_ip_3
    "192.168.0.8": 1,   # pipeline_2_stage_0_node_ip = g6e_xlarge_node_ip_1
    "192.168.0.144": 1,  # pipeline_2_stage_1_node_ip = g6e_xlarge_node_ip_2

    # Pipeline 2
    "192.168.0.30": 2,  # pipeline_2_stage_2_node_ip = g6e_xlarge_node_ip_3
    "192.168.0.115": 2,  # pipeline_1_stage_3_node_ip = g5_12xlarge_node_ip_1
    "192.168.0.202": 2,  # pipeline_1_stage_4_node_ip = g5_12xlarge_node_ip_2
    "192.168.0.140": 2,  # pipeline_2_stage_3_node_ip = g6e_xlarge_node_ip_4
}

NEW_ITERATIONS_10_STAGE_MAPPING = {
    # Pipeline 1
    "192.168.0.103": 0,  # pipeline_1_stage_0
    "192.168.0.248": 1,  # pipeline_1_stage_1
    "192.168.0.198": 2,  # pipeline_1_stage_2
    "192.168.0.8": 3,   # pipeline_1_stage_3
    "192.168.0.144": 4,  # pipeline_1_stage_4
    # Pipeline 2
    "192.168.0.30": 0,  # pipeline_2_stage_0
    "192.168.0.115": 1,  # pipeline_2_stage_1
    "192.168.0.202": 2,  # pipeline_2_stage_2
    "192.168.0.140": 3,  # pipeline_2_stage_3
}

def parse_cluster_log(log_path):
    """Parse Cluster log to extract base timestamps."""
    results = {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Pattern: [Pipeline X] Ray Cluster Started at TIMESTAMP and Ended at TIMESTAMP
    ray_pattern = r'\[Pipeline (\d+)\] Ray Cluster Started at ([\d.]+) and Ended at ([\d.]+)'
    for match in re.finditer(ray_pattern, content):
        pipeline = int(match.group(1))
        start_time = float(match.group(2))
        end_time = float(match.group(3))

        if pipeline not in results:
            results[pipeline] = {}
        results[pipeline]['ray_cluster_start'] = start_time
        results[pipeline]['ray_cluster_end'] = end_time

    # Pattern: [Pipeline X] Tensor Store Started at TIMESTAMP
    ts_pattern = r'\[Pipeline (\d+)\] Tensor Store Started at ([\d.]+)'
    for match in re.finditer(ts_pattern, content):
        pipeline = int(match.group(1))
        ts_start = float(match.group(2))

        if pipeline not in results:
            results[pipeline] = {}
        results[pipeline]['tensorstore_start'] = ts_start

    # Pattern: [Pipeline X] API Server Started at TIMESTAMP
    api_pattern = r'\[Pipeline (\d+)\] API Server Started at ([\d.]+)'
    for match in re.finditer(api_pattern, content):
        pipeline = int(match.group(1))
        api_start = float(match.group(2))

        if pipeline not in results:
            results[pipeline] = {}
        results[pipeline]['apiserver_start'] = api_start

    return results


def parse_tensorstore_log(log_path):
    """Parse TensorStore log to extract start and end times."""
    with open(log_path, 'r') as f:
        content = f.read()

    result = {}

    # Pattern: Raw S3 Tensor Store Server Started at TIMESTAMP
    start_match = re.search(r'Raw S3 Tensor Store Server Started at ([\d.]+)', content)
    if start_match:
        result['start'] = float(start_match.group(1))

    # Pattern: Start time: TIMESTAMP and End time: TIMESTAMP
    end_match = re.search(r'Start time: ([\d.]+) and End time: ([\d.]+)', content)
    if end_match:
        result['loading_start'] = float(end_match.group(1))
        result['loading_end'] = float(end_match.group(2))
        result['loading_time'] = result['loading_end'] - result['loading_start']

    return result


def parse_apiserver_log(log_path):
    """Parse API Server log to extract start and end times."""
    with open(log_path, 'r') as f:
        content = f.read()

    result = {}

    # Pattern: run_server start time: TIMESTAMP
    start_match = re.search(r'run_server start time: ([\d.]+)', content)
    if start_match:
        result['start'] = float(start_match.group(1))

    # Pattern: run_server end time: TIMESTAMP
    end_match = re.search(r'run_server end time: ([\d.]+)', content)
    if end_match:
        result['end'] = float(end_match.group(1))
        if 'start' in result:
            result['total_time'] = result['end'] - result['start']

    return result


def analyze_logs(base_path, node_mapping, pipeline_mapping, stage_mapping, experiment_name, max_runs=None):
    """Analyze all logs in the given path."""
    base_path = Path(base_path)
    results = []
    cluster_timing_results = []

    # Determine available run indices
    if max_runs is None:
        # Auto-detect by finding Cluster_*.log files
        cluster_logs = list(base_path.glob("Cluster_*.log"))
        run_indices = []
        for log in cluster_logs:
            match = re.search(r'Cluster_(\d+)\.log', log.name)
            if match:
                run_indices.append(int(match.group(1)))
        run_indices = sorted(run_indices)
    else:
        run_indices = list(range(max_runs))

    print(f"  Found {len(run_indices)} runs: {run_indices}")

    # Parse Cluster logs for each run
    for run_idx in run_indices:
        cluster_log = base_path / f"Cluster_{run_idx}.log"
        if not cluster_log.exists():
            print(f"Warning: {cluster_log} not found")
            continue

        cluster_data = parse_cluster_log(cluster_log)

        # Get Ray Cluster start times for each pipeline
        ray_cluster_starts = {}
        for pipeline, data in cluster_data.items():
            ray_cluster_starts[pipeline] = data.get('ray_cluster_start')

        # Parse remote logs
        remote_path = base_path / "remote"
        if not remote_path.exists():
            print(f"Warning: {remote_path} not found")
            continue

        # Parse TensorStore logs
        for ts_log in remote_path.glob("tensorstore_*.log"):
            # Extract IP, GPU, and run from filename
            # Format: tensorstore_IP_GPU_RUN.log
            parts = ts_log.stem.split('_')
            if len(parts) >= 4:
                ip = parts[1]
                gpu = parts[2]
                log_run = int(parts[3])

                if log_run != run_idx:
                    continue

                ts_data = parse_tensorstore_log(ts_log)

                if ip in node_mapping:
                    node_type, node_num = node_mapping[ip]

                    # Determine which pipeline this node belongs to
                    node_pipeline = pipeline_mapping.get(ip)

                    # Determine which stage this node is
                    node_stage = stage_mapping.get(ip)

                    # Calculate time from ray cluster start ONLY for the pipeline this node belongs to
                    time_from_ray_cluster = {}
                    if node_pipeline and node_pipeline in ray_cluster_starts:
                        ray_start = ray_cluster_starts[node_pipeline]
                        if ts_data.get('loading_end') and ray_start:
                            time_from_ray_cluster[f'time_from_ray_start'] = ts_data['loading_end'] - ray_start

                    result = {
                        'experiment': experiment_name,
                        'run': run_idx,
                        'pipeline': node_pipeline,
                        'stage': node_stage,
                        'node_type': node_type,
                        'node_num': node_num,
                        'ip': ip,
                        'gpu': gpu,
                        'module': 'tensorstore',
                        'start_time': ts_data.get('start', None),
                        'end_time': ts_data.get('loading_end', None),
                        'loading_time': ts_data.get('loading_time', None),
                        **time_from_ray_cluster
                    }
                    results.append(result)

        # Parse API Server logs
        for api_log in remote_path.glob("apiserver_*.log"):
            # Extract IP and run from filename
            # Format: apiserver_IP_RUN.log
            parts = api_log.stem.split('_')
            if len(parts) >= 3:
                ip = parts[1]
                log_run = int(parts[2])

                if log_run != run_idx:
                    continue

                api_data = parse_apiserver_log(api_log)

                if ip in node_mapping:
                    node_type, node_num = node_mapping[ip]

                    # Determine which pipeline this node belongs to
                    node_pipeline = pipeline_mapping.get(ip)

                    # Determine which stage this node is
                    node_stage = stage_mapping.get(ip)

                    # Calculate time from ray cluster start ONLY for the pipeline this node belongs to
                    time_from_ray_cluster = {}
                    if node_pipeline and node_pipeline in ray_cluster_starts:
                        ray_start = ray_cluster_starts[node_pipeline]
                        if api_data.get('end') and ray_start:
                            time_from_ray_cluster[f'time_from_ray_start'] = api_data['end'] - ray_start

                    result = {
                        'experiment': experiment_name,
                        'run': run_idx,
                        'pipeline': node_pipeline,
                        'stage': node_stage,
                        'node_type': node_type,
                        'node_num': node_num,
                        'ip': ip,
                        'gpu': 'all',
                        'module': 'apiserver',
                        'start_time': api_data.get('start', None),
                        'end_time': api_data.get('end', None),
                        'loading_time': api_data.get('total_time', None),
                        **time_from_ray_cluster
                    }
                    results.append(result)

        # Add cluster timing info
        for pipeline, data in cluster_data.items():
            result = {
                'experiment': experiment_name,
                'run': run_idx,
                'pipeline': pipeline,
                'ray_cluster_start': data.get('ray_cluster_start'),
                'ray_cluster_end': data.get('ray_cluster_end'),
                'ray_cluster_time': data.get('ray_cluster_end', 0) - data.get('ray_cluster_start', 0),
                'tensorstore_start_from_cluster': data.get('tensorstore_start'),
                'apiserver_start_from_cluster': data.get('apiserver_start'),
            }
            # Calculate time from ray cluster start to module completion
            if data.get('tensorstore_start') and data.get('ray_cluster_start'):
                result['tensorstore_time_from_ray_start'] = data['tensorstore_start'] - data['ray_cluster_start']
            if data.get('apiserver_start') and data.get('ray_cluster_start'):
                result['apiserver_time_from_ray_start'] = data['apiserver_start'] - data['ray_cluster_start']

            cluster_timing_results.append(result)

    return results, cluster_timing_results


def main():
    """Main analysis function."""
    base_dir = Path("logs")

    # Analyze withfsr
    print("=" * 80)
    print("Analyzing WITH FSR logs...")
    print("=" * 80)
    withfsr_modules, withfsr_cluster = analyze_logs(
        base_dir / "withfsr",
        WITHFSR_NODE_MAPPING,
        WITHFSR_PIPELINE_MAPPING,
        WITHFSR_STAGE_MAPPING,
        "withfsr"
    )

    # Analyze withoutfsr
    print("\n" + "=" * 80)
    print("Analyzing WITHOUT FSR logs...")
    print("=" * 80)
    withoutfsr_modules, withoutfsr_cluster = analyze_logs(
        base_dir / "withoutfsr",
        WITHOUTFSR_NODE_MAPPING,
        WITHOUTFSR_PIPELINE_MAPPING,
        WITHOUTFSR_STAGE_MAPPING,
        "withoutfsr"
    )

    # Analyze 10_iterations
    print("\n" + "=" * 80)
    print("Analyzing 10 ITERATIONS logs...")
    print("=" * 80)
    iterations_modules, iterations_cluster = analyze_logs(
        base_dir / "10_iterations",
        ITERATIONS_10_NODE_MAPPING,
        ITERATIONS_10_PIPELINE_MAPPING,
        ITERATIONS_10_STAGE_MAPPING,
        "10_iterations"
    )

    # Analyze new_10_iterations
    print("\n" + "=" * 80)
    print("Analyzing new 10 ITERATIONS logs...")
    print("=" * 80)
    new_iterations_modules, new_iterations_cluster = analyze_logs(
        base_dir / "new_10iterations",
        NEW_ITERATIONS_10_NODE_MAPPING,
        NEW_ITERATIONS_10_PIPELINE_MAPPING,
        NEW_ITERATIONS_10_STAGE_MAPPING,
        "new_10_iterations"
    )

    # Combine results
    module_results = withfsr_modules + withoutfsr_modules + iterations_modules + new_iterations_modules
    cluster_results = withfsr_cluster + withoutfsr_cluster + iterations_cluster + new_iterations_cluster

    # Create DataFrames
    if module_results:
        df_modules = pd.DataFrame(module_results)
        print("\n" + "=" * 80)
        print("MODULE LOADING TIMES")
        print("=" * 80)
        print(df_modules.to_string(index=False))

        # Summary by experiment, module, and node type
        print("\n" + "=" * 80)
        print("SUMMARY: Average Loading Time by Experiment, Module, and Node Type")
        print("=" * 80)
        summary = df_modules.groupby(['experiment', 'module', 'node_type'])['loading_time'].agg(['mean', 'std', 'count'])
        print(summary)

        # Summary by experiment and module only
        print("\n" + "=" * 80)
        print("SUMMARY: Average Loading Time by Experiment and Module")
        print("=" * 80)
        summary2 = df_modules.groupby(['experiment', 'module'])['loading_time'].agg(['mean', 'std', 'count'])
        print(summary2)

        # Summary: Time from Ray Cluster start
        if 'time_from_ray_start' in df_modules.columns:
            print("\n" + "=" * 80)
            print("SUMMARY: Time from Ray Cluster Start to Module Completion (by Pipeline)")
            print("=" * 80)

            # Filter data with valid time_from_ray_start
            df_with_time = df_modules[df_modules['time_from_ray_start'].notna()]

            if len(df_with_time) > 0:
                for pipeline in [1, 2]:
                    pipeline_data = df_with_time[df_with_time['pipeline'] == pipeline]
                    if len(pipeline_data) > 0:
                        print(f"\n{'='*80}")
                        print(f"Pipeline {pipeline}:")
                        print('='*80)
                        time_summary = pipeline_data.groupby(['experiment', 'module', 'node_type'])['time_from_ray_start'].agg(['mean', 'std', 'min', 'max', 'count'])
                        print(time_summary)

                        # Overall summary by experiment and module for this pipeline
                        print(f"\nOverall average for Pipeline {pipeline}:")
                        overall = pipeline_data.groupby(['experiment', 'module'])['time_from_ray_start'].agg(['mean', 'std', 'count'])
                        print(overall)

    if cluster_results:
        df_cluster = pd.DataFrame(cluster_results)
        print("\n" + "=" * 80)
        print("CLUSTER TIMING INFORMATION")
        print("=" * 80)
        print(df_cluster.to_string(index=False))

    # Save to CSV with sorting
    if module_results:
        # Sort by experiment, pipeline, run, stage, module, gpu
        df_modules_sorted = df_modules.sort_values(
            by=['experiment', 'pipeline', 'run', 'stage', 'module', 'gpu'],
            na_position='last'
        )
        df_modules_sorted.to_csv('module_loading_times.csv', index=False)
        print("\n✓ Module results saved to module_loading_times.csv (sorted by experiment, pipeline, run, stage, module, gpu)")

    if cluster_results:
        # Sort by experiment, run, pipeline
        df_cluster_sorted = df_cluster.sort_values(
            by=['experiment', 'run', 'pipeline']
        )
        df_cluster_sorted.to_csv('cluster_timing.csv', index=False)
        print("✓ Cluster results saved to cluster_timing.csv (sorted)")


if __name__ == "__main__":
    main()
