#include <chrono>
#include "helper.h"

int main(int argc, char **argv){
    int N = std::atoi(argv[1]);
    int M = N;
    int K = N;

    size_t bytes_a = M * N * sizeof(float);
    size_t bytes_b = N * K * sizeof(float);
    size_t bytes_c = M * K * sizeof(float);

    float *A = (float*)malloc(bytes_a);
    float *B = (float*)malloc(bytes_b);
    float *C = (float*)malloc(bytes_c);
    matrix_init(A, M, N);
    matrix_init(B, N, K);

    auto st = std::chrono::steady_clock::now();
    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            C[i*K+j] = 0;
            for(int k=0; k<N; k++){
                C[i*K+j] += A[i*N + k] *  B[k*K + j];
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto dt = end - st;
    float msec = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();

    printf("spend %f ms with size of (%d, %d, %d)\n", msec, M, N, K);
    printf("Computational Throughput: %f GFLOPS\n", (float)2*M*N*K*1e-6/msec);

    // TODO: 有diff
    // checkResult(A, B, C, M, N, K);

    free(A);
    free(B);
    free(C);
}