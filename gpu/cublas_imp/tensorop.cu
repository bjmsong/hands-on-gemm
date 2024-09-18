#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../helper.h" 

int main(int argc, char** argv){
    int version;
    cublasStatus_t status = cublasGetVersion(cublasHandle_t(), &version);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS version: " << version << std::endl;
    } else {
        std::cerr << "Failed to get cuBLAS version" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    size_t bytes_a = M * N * sizeof(half);
    size_t bytes_b = N * K * sizeof(half);
    size_t bytes_c = M * K * sizeof(float);

    half* h_a = (half*)malloc(bytes_a);
    half* h_b = (half*)malloc(bytes_b);
    float* h_c = (float*)malloc(bytes_c);

    matrix_init(h_a, M, N);
    matrix_init(h_b, N, K);

    half *d_a, *d_b;
    float *d_c;
    checkCuda(cudaMalloc(&d_a, bytes_a));
    checkCuda(cudaMalloc(&d_b, bytes_b));
    checkCuda(cudaMalloc(&d_c, bytes_c));

    checkCuda(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
	cublasCreate(&handle);
    float alpha = 1.0f;
	float beta = 0.0f;
    int WARMUP_TIMES = 100;
    for (int n_count=0;n_count<WARMUP_TIMES;n_count++){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, 
        &alpha, d_b, CUDA_R_16F, K, d_a, CUDA_R_16F, N, &beta, d_c, 
        CUDA_R_32F, K, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
    }

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&end));
    checkCuda(cudaEventRecord(start));
    
    cudaDeviceSynchronize();
    int EXECUTE_TIMES = 100;
    for (int n_count=0; n_count<EXECUTE_TIMES; n_count++){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, 
        &alpha, d_b, CUDA_R_16F, K, d_a, CUDA_R_16F, N, &beta, d_c, 
        CUDA_R_32F, K, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
    }
    cudaDeviceSynchronize();

    checkCuda(cudaEventRecord(end));
    checkCuda(cudaEventSynchronize(start));
    checkCuda(cudaEventSynchronize(end));

    float msec;
    cudaEventElapsedTime(&msec, start, end);
    printf("spend %f ms with size of (%d, %d, %d)\n", msec/EXECUTE_TIMES, M, N, K);
    printf("Computational Throughput: %f TFLOPS\n", (float)2*M*N*K*1e-9*EXECUTE_TIMES/msec);

    checkCuda(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));
    checkResult(d_a, d_b, h_c, bytes_c, M, N, K);

    free(h_a);
    free(h_b);
    free(h_c);

    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    checkcuBLAS(cublasDestroy(handle));
}