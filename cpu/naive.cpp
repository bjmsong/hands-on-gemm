#include <chrono>
#include <vector>
#include "helper.h"

int main(int argc, char **argv){
    int N = std::atoi(argv[1]);
    int M = N;
    int K = N;

    std::vector<float> A(M*N, 1);
    std::vector<float> B(N*K, 1);
    std::vector<float> C(M*K, 0);
    vector_init(A, M*N);
    vector_init(B, N*K);

    auto st = std::chrono::steady_clock::now();
    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
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

    checkResult(A.data(), B.data(), C.data(), M, N, K);
}