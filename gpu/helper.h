#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void matrix_init(float* a, int M, int N){
    // A(M,N)
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = (float)(rand()) / (float)(rand());
        }
    }
}

bool floatEqual(float a, float b, float epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

static void PrintAssert(bool condition, float a, float b) {
    if (!condition) {
        printf("%f is not equal to %f\n", a, b);
    }
}

void checkResult(float* a, float* b, float* c, size_t bytes, int M, int N, int K){

    float* c_check;
    cudaMalloc(&c_check, bytes);

    cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
    // Calculate: c = (alpha*a) * b + (beta*c)
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
    &alpha, b, K, a, N, &beta, c_check, K);

    float* h_c_check = (float*)malloc(bytes);
    cudaMemcpy(h_c_check, c_check, bytes, cudaMemcpyDeviceToHost);

    for (int i=0; i < M; i++){
        for (int j=0; j < K; j++)
            PrintAssert(floatEqual(c[i*K + j],h_c_check[i*K + j]), c[i*K + j], h_c_check[i*K + j]);
    }

    free(h_c_check);
    cudaFree(c_check);
}
