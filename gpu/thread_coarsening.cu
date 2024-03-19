#include <cuda_runtime.h>
#include "helper.h" 

#define TILE_WIDTH 16 
#define CORSE_FATOR 4
__global__ void matrixMultipy(float*  a, float* b, float* c, int M, int N, int K){

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // each thread calculate #CORSE_FATOR points of Matrix C
    // (row, colStart), (row, colStart + TILE_WIDTH), ... (row, colStart + (CORSE_FATOR-1)*TILE_WIDTH)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * blockDim.x * CORSE_FATOR + threadIdx.x;

    float temp[CORSE_FATOR];
    for (int f = 0; f < CORSE_FATOR; f++){
        temp[f] = 0.0f;
    }
    if(row < M && colStart < K){
        for(int ph=0; ph<N/TILE_WIDTH; ph++){
            // load by row
            Mds[threadIdx.y][threadIdx.x] = a[row * N + ph * TILE_WIDTH + threadIdx.x];
            // load by col
            for (int f = 0; f < CORSE_FATOR; f++){
                Nds[threadIdx.y][threadIdx.x] = b[(ph*TILE_WIDTH+threadIdx.y)*K + colStart + f * TILE_WIDTH];
                __syncthreads();

            for(int i = 0; i < TILE_WIDTH; i++)
                temp[f] += Mds[threadIdx.y][i] * Nds[i][threadIdx.x];
            __syncthreads();
        }
        for (int f = 0; f < CORSE_FATOR; f++)
            c[row*K + colStart + f * TILE_WIDTH] = temp[f];
    }
    }
}

int main(int argc, char** argv){
    int N = std::atoi(argv[1]);
    int M = N;
    int K = N;

    size_t bytes_a = M * N * sizeof(float);
    size_t bytes_b = N * K * sizeof(float);
    size_t bytes_c = M * K * sizeof(float);

    float* h_a = (float*)malloc(bytes_a);
    float* h_b = (float*)malloc(bytes_b);
    float* h_c = (float*)malloc(bytes_c);

    matrix_init(h_a, M, N);
    matrix_init(h_b, N, K);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = TILE_WIDTH;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(GRID_SIZE/CORSE_FATOR, GRID_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    matrixMultipy<<<grid, block>>>(d_a, d_b, d_c, N, N, N);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);
    // checkResult(d_a, d_b, h_c, bytes_c, M, N, K);

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("spend %f ms with size of (%d, %d, %d)\n", msec, M, N, K);
    printf("Computational Throughput: %f TFLOPS\n", (float)2*M*N*K*1e-9/msec);
}