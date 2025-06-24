#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h> // __half 타입을 위해 필수
#include <iomanip>

// CUDA API 호출의 에러를 확인하기 위한 헬퍼 매크로
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// cuBLAS API 호출의 에러를 확인하기 위한 헬퍼 매크로
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: Status %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 행렬을 랜덤 값으로 초기화하는 함수 (FP16용)
void initialize_matrix_fp16(std::vector<__half>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) {
        float rand_val = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f;
        mat[i] = __float2half(rand_val);
    }
}

// 두 행렬이 거의 같은지 확인하는 함수
void verify_result(const std::vector<float>& ref, const std::vector<float>& res, float tolerance) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(ref[i] - res[i]));
    }
    std::cout << "최대 오차 (Max Difference): " << max_diff << std::endl;
    if (max_diff > tolerance) {
        std::cout << "결과 검증 실패! 오차가 허용치(" << tolerance << ")보다 큽니다." << std::endl;
    } else {
        std::cout << "결과 검증 성공!" << std::endl;
    }
}

int main() {
    // 행렬 차원 설정 (Tensor Core 효율을 위해 8의 배수로 설정)
    int M = 8192;
    int N = 8192;
    int K = 8192;
    int A_size = M * K;
    int B_size = K * N;
    int C_size = M * N;

    std::cout << "행렬 크기: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // GPU 정보 출력
    int deviceId;
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    CHECK_CUDA(cudaGetDeviceProperties(&props, deviceId));
    std::cout << "사용 중인 GPU: " << props.name << " (Compute Capability: " << props.major << "." << props.minor << ")" << std::endl;

    int64_t hardware_FLOPS = 121LL * 1000LL * 1000LL * 1000LL * 1000LL; // 121 TFLOPS
    int64_t hardware_mem_bandwidth = 300LL * 1000LL * 1000LL * 1000LL; // 300 GB/s

    std::cout << "GPU FLOPS: " << hardware_FLOPS / (1000LL * 1000LL * 1000LL * 1000LL) << " TFLOPS" << std::endl;
    std::cout << "GPU Memory Bandwidth: " << hardware_mem_bandwidth / (1000LL * 1000LL * 1000LL) << " GB/s" << std::endl;

    int64_t mag_ops = 2LL * M * N * K;
    int64_t estimated_time_ms = mag_ops / (hardware_FLOPS / 1000LL);

    std::cout << "Estimated Time: " << estimated_time_ms << " ms" << std::endl;
    
    // ----------------------------------------------------------------------
    // 초기화(데이터 준비 + 메모리 할당/복사) 시간 측정
    // ----------------------------------------------------------------------
    auto start_init = std::chrono::high_resolution_clock::now();

    // 호스트(CPU) 메모리 할당 및 초기화
    std::vector<__half> h_A(A_size);
    std::vector<__half> h_B(B_size);
    std::vector<__half> h_C_gpu_fp16(C_size);

    initialize_matrix_fp16(h_A);
    initialize_matrix_fp16(h_B);

    // 디바이스(GPU) 메모리 할당 (FP16용)
    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc((void**)&d_A_fp16, A_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_B_fp16, B_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_C_fp16, C_size * sizeof(__half)));

    // 호스트에서 디바이스로 FP16 데이터 복사
    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A.data(), A_size * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B.data(), B_size * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end_init = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> init_dur = end_init - start_init;
    std::cout << "\n--- 초기화 시간 ---\n";
    std::cout << "데이터 준비 + 메모리 할당/복사: " << init_dur.count() << " ms" << std::endl;

    // cuBLAS 핸들 생성
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // 반복 횟수 및 워밍업 설정
    const int WARMUP_ITERS = 10;
    const int BENCH_ITERS = 100;

    // ======================================================================
    // 1. 일반 CUDA Core (FP16) 측정
    // ======================================================================
    std::cout << "\n--- 1. 일반 CUDA Core (FP16) 실행 ---" << std::endl;
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    
    __half alpha_fp16 = __float2half(1.0f);
    __half beta_fp16 = __float2half(0.0f);

    // 워밍업 시간 측정
    auto start_warmup_cuda = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha_fp16, d_B_fp16, N, d_A_fp16, K,
                                 &beta_fp16, d_C_fp16, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_warmup_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> warmup_cuda_dur = end_warmup_cuda - start_warmup_cuda;
    std::cout << "워밍업(" << WARMUP_ITERS << "회) 시간: " << warmup_cuda_dur.count() << " ms" << std::endl;

    // 벤치마크 반복
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
    std::cout << "CUDA Core (FP16) 평균 실행 시간 (" << BENCH_ITERS << "회): " << gpu_fp16_cuda_avg << " ms" << std::endl;

    // ======================================================================
    // 2. Tensor Core (FP16) 측정
    // ======================================================================
    if (props.major < 7) {
        std::cout << "\n--- 2. Tensor Core (FP16) 건너뜀 ---" << std::endl;
        std::cout << "이 GPU(Compute Capability " << props.major << "." << props.minor << ")는 FP16 Tensor Core를 지원하지 않습니다 (Volta 아키텍처 이상 필요)." << std::endl;
    } else {
        std::cout << "\n--- 2. Tensor Core (FP16) 실행 ---" << std::endl;
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        
        __half alpha_fp16 = __float2half(1.0f);
        __half beta_fp16 = __float2half(0.0f);
        
        // 워밍업 시간 측정
        auto start_warmup_tensor = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha_fp16, d_B_fp16, N, d_A_fp16, K,
                                     &beta_fp16, d_C_fp16, N));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_warmup_tensor = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> warmup_tensor_dur = end_warmup_tensor - start_warmup_tensor;
        std::cout << "워밍업(" << WARMUP_ITERS << "회) 시간: " << warmup_tensor_dur.count() << " ms" << std::endl;
        
        // 벤치마크 반복
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
        std::cout << "Tensor Core (FP16) 평균 실행 시간 (" << BENCH_ITERS << "회): " << gpu_fp16_avg << " ms" << std::endl;

        std::cout << "\n--- 성능 비교 (Tensor Core / CUDA Core) ---" << std::endl;
        std::cout << "배속: " << std::fixed << std::setprecision(2) << gpu_fp16_cuda_avg / gpu_fp16_avg << "x" << std::endl;
    }

    // ======================================================================
    // 정리
    // ======================================================================
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));

    return 0;
}
