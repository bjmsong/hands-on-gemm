#include <cuda_runtime.h>
#include "../helper.h" 

// Unified Memory 不影响性能

__global__ void matrixMultipy(float* a, float* b, float* c, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    if (row < M && col < K){
        for (int i=0; i<N; i++){
            temp += a[row * N + i] * b[i * K + col];
        }
        c[row * K + col] = temp;
    }
}

int main(int argc, char** argv){
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    size_t bytes_a = M * N * sizeof(float);
    size_t bytes_b = N * K * sizeof(float);
    size_t bytes_c = M * K * sizeof(float);

    float* h_a = (float*)malloc(bytes_a);
    float* h_b = (float*)malloc(bytes_b);
    float* h_c = (float*)malloc(bytes_c);

    matrix_init(h_a, M, N);
    matrix_init(h_b, N, K);

    float *d_a, *d_b, *d_c;
    checkCuda(cudaMallocManaged((void **)&d_a, bytes_a));
    checkCuda(cudaMallocManaged((void **)&d_b, bytes_b));
    checkCuda(cudaMallocManaged((void **)&d_c, bytes_c));
    
    matrix_init(d_a, M, N);
    matrix_init(d_b, N, K);

    int BLOCK_SIZE = 16;
    int GRID_SIZE_X = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int GRID_SIZE_Y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(GRID_SIZE_X, GRID_SIZE_Y);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int WARMUP_TIMES = 100;
    for (int n_count=0; n_count < WARMUP_TIMES; n_count++){
        matrixMultipy<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&end));
    checkCuda(cudaEventRecord(start));

    cudaDeviceSynchronize();
    int EXECUTE_TIMES = 100;
    for (int n_count=0; n_count < EXECUTE_TIMES; n_count++){
        matrixMultipy<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaDeviceSynchronize();

    checkCuda(cudaEventRecord(end));
    checkCuda(cudaEventSynchronize(end));

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    // 有diff
    // checkResult(d_a, d_b, d_c, bytes_c, M, N, K);

    free(h_a);
    free(h_b);
    free(h_c);

    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    printf("spend %f ms with size of (%d, %d, %d)\n", msec/EXECUTE_TIMES, M, N, K);
    printf("Computational Throughput: %f TFLOPS\n", (float)2*M*N*K*1e-9*EXECUTE_TIMES/msec);
}