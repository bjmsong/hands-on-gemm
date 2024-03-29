#include <Eigen/Dense>
#include <chrono>
#include "helper.h"

using Eigen::MatrixXf;
 
int main(int argc, char** argv){
    int N = std::atoi(argv[1]);
    int M = N;
    int K = N;

    size_t bytes_a = M * N * sizeof(float);
    size_t bytes_b = N * K * sizeof(float);
    size_t bytes_c = M * K * sizeof(float);

    float *A = (float*)malloc(bytes_a);
    float *B = (float*)malloc(bytes_b);
    matrix_init(A, M, N);
    matrix_init(B, N, K);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e, B_e, C_e;
    A_e.resize(M, N);
    B_e.resize(N, K);
    C_e.resize(M, K);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A_e(i, j) = A[i * N + j];
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            B_e(i, j) = B[i * K + j];
        }
    }

    Eigen::setNbThreads(10);
    auto st = std::chrono::steady_clock::now();
    int EXECUTE_TIMES = 100;
    for (int n_count=0; n_count < EXECUTE_TIMES; n_count++){
        C_e = A_e * B_e;
    }
    auto end = std::chrono::steady_clock::now();
    auto dt = end - st;
    float msec = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();

    printf("spend %f ms with size of (%d, %d, %d)\n", msec/EXECUTE_TIMES, M, N, K);
    printf("Computational Throughput: %f GFLOPS\n", (float)2*M*N*K*1e-6*EXECUTE_TIMES/msec);
    checkResult(A, B, C_e.data(), M, N, K);

    free(A);
    free(B);
}