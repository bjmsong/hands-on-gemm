#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper.h" 

int main(int argc, char** argv){
    int version;
    cublasStatus_t status = cublasGetVersion(cublasHandle_t(), &version);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS version: " << version << std::endl;
    } else {
        std::cerr << "Failed to get cuBLAS version" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int M = N;
    int K = N;

    size_t bytes_a = M * N * sizeof(half);
    size_t bytes_b = N * K * sizeof(half);
    size_t bytes_c = M * K * sizeof(half);

    half* h_a = (half*)malloc(bytes_a);
    half* h_b = (half*)malloc(bytes_b);
    half* h_c = (half*)malloc(bytes_c);

    matrix_init(h_a, M, N);
    matrix_init(h_b, N, K);

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 32;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cublasHandle_t handle;
	cublasCreate(&handle);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

	__half alpha = 1.0f;
	__half beta = 0.0f;
    cudaDeviceSynchronize();
    int EXECUTE_TIMES = 100;
    for (int n_count=0; n_count<EXECUTE_TIMES; n_count++){
        // c = (alpha*a) * b + (beta*c)
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
        &alpha, d_b, K, d_a, N, &beta, d_c, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);
    printf("spend %f ms with size of (%d, %d, %d)\n", msec/EXECUTE_TIMES, M, N, K);
    printf("Computational Throughput: %f TFLOPS\n", (float)2*M*N*K*1e-9*EXECUTE_TIMES/msec);

    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);
    checkResult(d_a, d_b, h_c, bytes_c, M, N, K);

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}