#include <chrono>
#include <immintrin.h>
#include "helper.h"

float Floatsum(const __m256 a) {
    __m128 res = _mm256_extractf128_ps(a, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(a));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

void gemm(float* A, float* Bt, float* C, int M, int N, int K){
    #pragma omp parallel for collapse(2)
    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            float tmp = 0;
            tmp = 0;
            int l = 0;
            __m256 vsum = _mm256_setzero_ps();
            for (; l + 7 < N; l += 8) {
                __m256 vi = _mm256_loadu_ps(A + i * N + l);
                __m256 vw = _mm256_loadu_ps(Bt + j * N + l);
                vsum = _mm256_fmadd_ps(vi, vw, vsum);
            }
            tmp += Floatsum(vsum);
            for (; l < N; l++) {
                tmp += A[i * N + l] * Bt[j * N + l];
            }
            C[i * K + j] = tmp;
        }
    }
}

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
    auto end = std::chrono::steady_clock::now();
    auto dt = end - st;
    float msec = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
    printf("spend %f ms with transpose\n", msec);

    st = std::chrono::steady_clock::now();
    int EXECUTE_TIMES = 100;
    for (int n_count=0; n_count < EXECUTE_TIMES; n_count++){
        gemm(A, B, Bt, M, N, K);
    }
    end = std::chrono::steady_clock::now();
    dt = end - st;
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();

    printf("spend %f ms with size of (%d, %d, %d)\n", msec/EXECUTE_TIMES, M, N, K);
    printf("Computational Throughput: %f GFLOPS\n", (float)2*M*N*K*1e-6*EXECUTE_TIMES/msec);

    // TODO: 有diff
    // checkResult(A, B, C, M, N, K);

    free(A);
    free(B);
    free(C);
    free(Bt);
}