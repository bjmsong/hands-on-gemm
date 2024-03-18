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
    float* Bt = (float*)malloc(bytes_b);
    // B(i,j) == Bt(j,i)
    for (int i = 0; i < N; i++){
        for (int j = 0; j < K; j++)
            Bt[j*N+i] = B[i*K+j];
    }

    #pragma omp parallel for collapse(2)
    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            float tmp = 0;
            for(int k=0; k<N; k++){
                tmp += A[i*N + k] *  Bt[j*N + k];
            }
            C[i*K+j] = tmp;
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto dt = end - st;
    float msec = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();

    printf("spend %f ms with size of (%d, %d, %d)\n", msec, M, N, K);

    // TODO: 有diff
    // checkResult(A, B, C, M, N, K);
}