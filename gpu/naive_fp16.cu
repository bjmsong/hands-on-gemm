#include <cuda_runtime.h>
#include "helper.h" 

__global__ void matrixMultipy(half* a, half* b, half* c, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    if (row < M && col < K){
        for (int i=0; i<N; i++){
            // 转换会带来误差
            temp += __half2float(a[row * N + i]) * __half2float(b[i * K + col]);
        }
        c[row * K + col] = __float2half(temp);
    }
}

int main(int argc, char** argv){
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

    int BLOCK_SIZE = 16;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    cudaDeviceSynchronize();
    int EXECUTE_TIMES = 100;
    for (int n_count=0;n_count<EXECUTE_TIMES;n_count++){
        matrixMultipy<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);
    checkResult(d_a, d_b, h_c, bytes_c, M, N, K);

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("spend %f ms with size of (%d, %d, %d)\n", msec/EXECUTE_TIMES, M, N, K);
    printf("Computational Throughput: %f TFLOPS\n", (float)2*M*N*K*1e-9*EXECUTE_TIMES/msec);
}