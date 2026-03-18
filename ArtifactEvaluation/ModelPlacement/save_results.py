"""Utility to save BenchmarkMetrics to JSON file."""
import json
import os


def save_benchmark_results(metrics, output_path: str, extra: dict = None):
    """Save BenchmarkMetrics to a JSON file.
    
    Args:
        metrics: BenchmarkMetrics object from benchmark_utils.
        output_path: Path to save JSON file.
        extra: Optional dict of additional fields to include (e.g., pipeline config).
    """
    result = {
        "completed": metrics.completed,
        "total_input": metrics.total_input,
        "total_output": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "mean_e2el_ms": metrics.mean_e2el_ms,
        "median_e2el_ms": metrics.median_e2el_ms,
        "benchmark_duration": metrics.benchmark_duration,
    }

    for attr in ["percentiles_ttft_ms", "percentiles_tpot_ms",
                 "percentiles_itl_ms", "percentiles_e2el_ms"]:
        val = getattr(metrics, attr, None)
        if val:
            result[attr] = {str(int(p)): v for p, v in val}

    if extra:
        result.update(extra)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")
