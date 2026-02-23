#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h> // Required for __half type
#include <iomanip>

// Helper macro for checking CUDA API call errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Helper macro for checking cuBLAS API call errors
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: Status %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Function to initialize a matrix with random values (for FP16)
void initialize_matrix_fp16(std::vector<__half>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) {
        float rand_val = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f;
        mat[i] = __float2half(rand_val);
    }
}

// Function to verify if two matrices are approximately equal
void verify_result(const std::vector<float>& ref, const std::vector<float>& res, float tolerance) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(ref[i] - res[i]));
    }
    std::cout << "Max Difference: " << max_diff << std::endl;
    if (max_diff > tolerance) {
        std::cout << "Verification failed! Error exceeds tolerance (" << tolerance << ")." << std::endl;
    } else {
        std::cout << "Verification passed!" << std::endl;
    }
}

int main() {
    // Set matrix dimensions (multiples of 8 for Tensor Core efficiency)
    int M = 8192;
    int N = 8192;
    int K = 8192;
    int A_size = M * K;
    int B_size = K * N;
    int C_size = M * N;

    std::cout << "Matrix size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Print GPU information
    int deviceId;
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    CHECK_CUDA(cudaGetDeviceProperties(&props, deviceId));
    std::cout << "GPU in use: " << props.name << " (Compute Capability: " << props.major << "." << props.minor << ")" << std::endl;

    int64_t hardware_FLOPS = 121LL * 1000LL * 1000LL * 1000LL * 1000LL; // 121 TFLOPS
    int64_t hardware_mem_bandwidth = 300LL * 1000LL * 1000LL * 1000LL; // 300 GB/s

    std::cout << "GPU FLOPS: " << hardware_FLOPS / (1000LL * 1000LL * 1000LL * 1000LL) << " TFLOPS" << std::endl;
    std::cout << "GPU Memory Bandwidth: " << hardware_mem_bandwidth / (1000LL * 1000LL * 1000LL) << " GB/s" << std::endl;

    int64_t mag_ops = 2LL * M * N * K;
    int64_t estimated_time_ms = mag_ops / (hardware_FLOPS / 1000LL);

    std::cout << "Estimated Time: " << estimated_time_ms << " ms" << std::endl;
    
    // ----------------------------------------------------------------------
    // Measure initialization time (data preparation + memory allocation/copy)
    // ----------------------------------------------------------------------
    auto start_init = std::chrono::high_resolution_clock::now();

    // Allocate and initialize host (CPU) memory
    std::vector<__half> h_A(A_size);
    std::vector<__half> h_B(B_size);
    std::vector<__half> h_C_gpu_fp16(C_size);

    initialize_matrix_fp16(h_A);
    initialize_matrix_fp16(h_B);

    // Allocate device (GPU) memory (for FP16)
    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc((void**)&d_A_fp16, A_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_B_fp16, B_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_C_fp16, C_size * sizeof(__half)));

    // Copy FP16 data from host to device
    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A.data(), A_size * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B.data(), B_size * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end_init = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> init_dur = end_init - start_init;
    std::cout << "\n--- Initialization Time ---\n";
    std::cout << "Data preparation + memory allocation/copy: " << init_dur.count() << " ms" << std::endl;

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Set iteration counts and warmup
    const int WARMUP_ITERS = 10;
    const int BENCH_ITERS = 100;

    // ======================================================================
    // 1. CUDA Core (FP16) benchmark
    // ======================================================================
    std::cout << "\n--- 1. CUDA Core (FP16) Execution ---" << std::endl;
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    
    __half alpha_fp16 = __float2half(1.0f);
    __half beta_fp16 = __float2half(0.0f);

    // Measure warmup time
    auto start_warmup_cuda = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha_fp16, d_B_fp16, N, d_A_fp16, K,
                                 &beta_fp16, d_C_fp16, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_warmup_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> warmup_cuda_dur = end_warmup_cuda - start_warmup_cuda;
    std::cout << "Warmup (" << WARMUP_ITERS << " iters) time: " << warmup_cuda_dur.count() << " ms" << std::endl;

    // Benchmark iterations
    auto start_gpu_fp16_cuda = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha_fp16, d_B_fp16, N, d_A_fp16, K,
                                 &beta_fp16, d_C_fp16, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_gpu_fp16_cuda = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> gpu_fp16_cuda_total = end_gpu_fp16_cuda - start_gpu_fp16_cuda;
    double gpu_fp16_cuda_avg = gpu_fp16_cuda_total.count() / BENCH_ITERS;
    std::cout << "CUDA Core (FP16) avg time (" << BENCH_ITERS << " iters): " << gpu_fp16_cuda_avg << " ms" << std::endl;

    // ======================================================================
    // 2. Tensor Core (FP16) benchmark
    // ======================================================================
    if (props.major < 7) {
        std::cout << "\n--- 2. Tensor Core (FP16) Skipped ---" << std::endl;
        std::cout << "This GPU (Compute Capability " << props.major << "." << props.minor << ") does not support FP16 Tensor Cores (requires Volta architecture or later)." << std::endl;
    } else {
        std::cout << "\n--- 2. Tensor Core (FP16) Execution ---" << std::endl;
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        
        __half alpha_fp16 = __float2half(1.0f);
        __half beta_fp16 = __float2half(0.0f);
        
        // Measure warmup time
        auto start_warmup_tensor = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha_fp16, d_B_fp16, N, d_A_fp16, K,
                                     &beta_fp16, d_C_fp16, N));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_warmup_tensor = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> warmup_tensor_dur = end_warmup_tensor - start_warmup_tensor;
        std::cout << "Warmup (" << WARMUP_ITERS << " iters) time: " << warmup_tensor_dur.count() << " ms" << std::endl;

        // Benchmark iterations
        auto start_gpu_fp16 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITERS; ++i) {
            CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha_fp16, d_B_fp16, N, d_A_fp16, K,
                                     &beta_fp16, d_C_fp16, N));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_gpu_fp16 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> gpu_fp16_total = end_gpu_fp16 - start_gpu_fp16;
        double gpu_fp16_avg = gpu_fp16_total.count() / BENCH_ITERS;
        std::cout << "Tensor Core (FP16) avg time (" << BENCH_ITERS << " iters): " << gpu_fp16_avg << " ms" << std::endl;

        std::cout << "\n--- Performance Comparison (Tensor Core / CUDA Core) ---" << std::endl;
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) << gpu_fp16_cuda_avg / gpu_fp16_avg << "x" << std::endl;
    }

    // ======================================================================
    // Cleanup
    // ======================================================================
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));

    return 0;
}
