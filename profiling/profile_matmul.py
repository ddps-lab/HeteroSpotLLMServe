# Run matmul and predict latency using flops and memory bandwidth

import torch
import time


if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()

    DTYPE=torch.float16
    DEVICE="cuda:0"

    # L4 is listed as 242 TFLOPS on the data sheet, but that applies to sparse matrices.
    # For structures like Transformers, the matrices are not sparse, so 121 TFLOPS is used in practice.
    test_FLOPS = 121 * 10**12
    test_memory_bandwidth = 300 * 10**9 # Unit: bytes/s

    ridge_point = test_FLOPS / test_memory_bandwidth
    
    print(f"DTYPE: {DTYPE}")
    print(f"Device FLOPS: {test_FLOPS // (10**12)} TFLOPS")
    print(f"Device Memory Bandwidth: {test_memory_bandwidth // (10**9)} GB/s")
    print(f"Ridge point (Machine Balance Point): {ridge_point:.2f}")

    # Measure both computation-bound workload and memory-bound workload
    # Use the simplest possible matrix multiplication workload.

    # 1. Computation-bound workload
    print("\n--- Computation-Bound Workload ---")
    K = 8192
    M = 8192
    N = 8192

    A = torch.rand(K, M, device=DEVICE, dtype=DTYPE) # Adjusted for matmul, dtype float16
    B = torch.rand(M, N, device=DEVICE, dtype=DTYPE) # Adjusted for matmul, dtype float16

    # Number of matrix multiplication operations: 2 * K * M * N
    matmul_flops = 2 * K * M * N
    # Memory access cost: read A + read B (writing C typically overlaps with computation)
    matmul_memory_bytes = (K * M + M * N) * A.element_size() # float16 is 2 bytes

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up
    for _ in range(5):
        C_warmup = torch.matmul(A, B)
    torch.cuda.synchronize()

    iterations = 1000
    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()

    real_time = (start_event.elapsed_time(end_event) / 1000) / iterations  # Average time (in seconds)
    real_flops = matmul_flops / real_time
    real_memory_bandwidth = matmul_memory_bytes / real_time
    
    # In computation-bound workloads, compute time is dominant
    estimated_time_flops = matmul_flops / test_FLOPS
    estimated_time_memory = matmul_memory_bytes / test_memory_bandwidth
    arithmetic_intensity = matmul_flops / matmul_memory_bytes

    print(f"Matrix multiplication workload: K={K}, M={M}, N={N}")
    print(f"Real Latency: {real_time*1000:.3f} ms")
    print(f"Estimated Latency (Compute): {estimated_time_flops*1000:.3f} ms")
    print(f"Estimated Latency (Memory Access): {estimated_time_memory*1000:.3f} ms")
    print(f"Arithmetic Intensity: {arithmetic_intensity:.2f}")
    if arithmetic_intensity > ridge_point:
        print("This workload is computation-bound")
        print(f"Real FLOPS: {real_flops / (1000**4):.2f} TFLOPS")
    else:
        print("This workload is memory-bound")
        print(f"Real Memory Bandwidth: {real_memory_bandwidth / (1000**3):.2f} GB/s")


    # 2. Memory-bound workload
    print("\n--- Memory-Bound Workload (Element-wise Sum) ---")
    # Create large tensors and perform element-wise addition (high memory access, low computation)
    size_mem = 1024 * 1024 * 1024  # 1Gi elements
    
    A_mem = torch.rand(size_mem, device=DEVICE, dtype=DTYPE)
    B_mem = torch.rand(size_mem, device=DEVICE, dtype=DTYPE)

    # Number of element-wise addition operations: size_mem (one addition per element)
    # In practice, SIMD and other optimizations may apply, but a simplified model is used here
    add_ops_mem = size_mem 
    # Memory access cost: read A + read B + write C
    # Each tensor is size_mem * element_size() bytes
    element_wise_memory_bytes_mem = (size_mem * A_mem.element_size()) * 3 # A read, B read, C write

    start_event_mem = torch.cuda.Event(enable_timing=True)
    end_event_mem = torch.cuda.Event(enable_timing=True)

    # Warm-up
    for _ in range(5):
        C_mem_warmup = A_mem + B_mem
    torch.cuda.synchronize()

    del C_mem_warmup
    torch.cuda.empty_cache()

    start_event_mem.record()
    for _ in range(iterations):
        A_mem += B_mem
    end_event_mem.record()
    torch.cuda.synchronize()

    real_time_mem = start_event_mem.elapsed_time(end_event_mem) / 1000 / iterations # In seconds
    real_flops_mem = add_ops_mem / real_time_mem
    real_memory_bandwidth_mem = element_wise_memory_bytes_mem / real_time_mem

    # In memory-bound workloads, memory transfer time is dominant
    estimated_time_mem_ops = add_ops_mem / test_FLOPS # Compute time is expected to be very small
    estimated_time_mem_memory = element_wise_memory_bytes_mem / test_memory_bandwidth
    arithmetic_intensity_mem = add_ops_mem / element_wise_memory_bytes_mem

    mem_size_per_tensor = size_mem * A_mem.element_size() / (1024**3)
    print(f"Element-wise addition workload: A ({mem_size_per_tensor:.2f} GB) + B ({mem_size_per_tensor:.2f} GB) = C ({mem_size_per_tensor:.2f} GB)")
    print(f"Real Latency: {real_time_mem*1000:.3f} ms")
    print(f"Estimated Latency (Compute): {estimated_time_mem_ops*1000:.3f} ms")
    print(f"Estimated Latency (Memory Access): {estimated_time_mem_memory*1000:.3f} ms")
    print(f"Arithmetic Intensity: {arithmetic_intensity_mem:.2f}")
    if arithmetic_intensity_mem > ridge_point:
        print("This workload is computation-bound")
        print(f"Real FLOPS: {real_flops_mem / (1000**4):.2f} TFLOPS")
    else:
        print("This workload is memory-bound")
        print(f"Real Memory Bandwidth: {real_memory_bandwidth_mem / (1000**3):.2f} GB/s")