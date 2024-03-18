#include <chrono>
#include "helper.h"
#include "threadpool.h"


void gemm(float* A, float* B, float* C, int M, int N, int K, int col_start, int col_end){
    for(int i=0; i<M; i++){
        for(int j=col_start; j<col_end; j++){
            float tmp = 0;
            for(int k=0; k<K; k++){
                tmp += A[i*N + k] *  B[k*K + j];
            }
            C[i*K+j] = tmp;
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
    int threadNum = std::atoi(argv[2]);
    int per = N / threadNum;    // 一个线程负责per列的计算
    int cur = 0;
    auto pool = new ThreadPool(threadNum);
    std::vector<std::future<void> > futures;
    for (int i = 0; i < threadNum - 1; i++) {
        int end = cur + per + (cur + per * (threadNum - i) < K);
        futures.push_back(pool->enqueue(gemm, A, B, C, M, N, K, cur, end));
        cur = end;
    }

    gemm(A, B, C, M, N, K, cur, N);  // 如果k不能被threadNum整除
    for (int i = 0; i < futures.size(); i++) {
        futures[i].get();
    }

    auto end = std::chrono::steady_clock::now();
    auto dt = end - st;
    float msec = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();

    printf("spend %f ms with size of (%d, %d, %d)\n", msec, M, N, K);

    // TODO: 有diff
    // checkResult(A, B, C, M, N, K);

    free(A);
    free(B);
    free(C);
}