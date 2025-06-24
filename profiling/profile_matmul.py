# matmul 을 실행하고 flops 와 memory bandwidth 를 통해 latency 를 예측

import torch
import time


if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()

    DTYPE=torch.float16
    DEVICE="cuda:0"

    # L4 는 Data Sheet 상으로는 242 TFLOPS 이지만, 이는 Sparsity Matrix 일 경우이며
    # Transformer 와 같은 구조에서는 Sparsity matrix 가 아니므로 실제로는 121 TFLOPS 로 적용된다.
    test_FLOPS = 121 * 10**12
    test_memory_bandwidth = 300 * 10**9 # 단위 : bytes/s

    ridge_point = test_FLOPS / test_memory_bandwidth
    
    print(f"DTYPE: {DTYPE}")
    print(f"Device FLOPS: {test_FLOPS // (10**12)} TFLOPS")
    print(f"Device Memory Bandwidth: {test_memory_bandwidth // (10**9)} GB/s")
    print(f"Ridge point (Machine Balance Point): {ridge_point:.2f}")

    # Computation-bound workload 와 Memory-bound workload 를 모두 측정
    # 최대한 단순한 행렬곱 워크로드로 설정한다.

    # 1. Computation-bound workload
    print("\n--- Computation-Bound Workload ---")
    K = 8192
    M = 8192
    N = 8192

    A = torch.rand(K, M, device=DEVICE, dtype=DTYPE) # Adjusted for matmul, dtype float16
    B = torch.rand(M, N, device=DEVICE, dtype=DTYPE) # Adjusted for matmul, dtype float16

    # 행렬곱 연산 수: 2 * K * M * N
    matmul_flops = 2 * K * M * N
    # 메모리 접근 비용: A 읽기 + B 읽기 (C 쓰기는 일반적으로 연산과 겹침)
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

    real_time = (start_event.elapsed_time(end_event) / 1000) / iterations  # 평균 시간 (초 단위)
    real_flops = matmul_flops / real_time
    real_memory_bandwidth = matmul_memory_bytes / real_time
    
    # Computation-bound workload에서는 연산 시간이 지배적
    estimated_time_flops = matmul_flops / test_FLOPS
    estimated_time_memory = matmul_memory_bytes / test_memory_bandwidth
    arithmetic_intensity = matmul_flops / matmul_memory_bytes

    print(f"행렬곱 워크로드 : K={K}, M={M}, N={N}")
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
    # 큰 텐트를 만들어서 요소별 합산을 수행 (메모리 접근이 많고, 연산은 적음)
    size_mem = 1024 * 1024 * 1024  # 1Gi elements
    
    A_mem = torch.rand(size_mem, device=DEVICE, dtype=DTYPE)
    B_mem = torch.rand(size_mem, device=DEVICE, dtype=DTYPE)

    # 요소별 덧셈 연산 수: size_mem (각 요소당 1번의 덧셈)
    # 실제로는 SIMD 등으로 더 최적화될 수 있지만, 단순화된 모델 사용
    add_ops_mem = size_mem 
    # 메모리 접근 비용: A 읽기 + B 읽기 + C 쓰기
    # 각 텐서는 size_mem * element_size() 바이트
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

    real_time_mem = start_event_mem.elapsed_time(end_event_mem) / 1000 / iterations # 초 단위
    real_flops_mem = add_ops_mem / real_time_mem
    real_memory_bandwidth_mem = element_wise_memory_bytes_mem / real_time_mem

    # Memory-bound workload에서는 메모리 전송 시간이 지배적
    estimated_time_mem_ops = add_ops_mem / test_FLOPS # 연산 시간은 매우 작을 것으로 예상
    estimated_time_mem_memory = element_wise_memory_bytes_mem / test_memory_bandwidth
    arithmetic_intensity_mem = add_ops_mem / element_wise_memory_bytes_mem

    mem_size_per_tensor = size_mem * A_mem.element_size() / (1024**3)
    print(f"행렬합 워크로드 : A ({mem_size_per_tensor:.2f} GB) + B ({mem_size_per_tensor:.2f} GB) = C ({mem_size_per_tensor:.2f} GB)")
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