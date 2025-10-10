"""
Benchmark utilities for GlobalServer testing.
This file re-exports functions from the parent benchmark_utils module for backwards compatibility.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and re-export all functions and classes from parent benchmark_utils
from benchmark_utils import (
    BenchmarkMetrics,
    calculate_benchmark_metrics,
    print_benchmark_results,
    run_benchmark_requests
)

__all__ = [
    'BenchmarkMetrics',
    'calculate_benchmark_metrics',
    'print_benchmark_results',
    'run_benchmark_requests'
]
