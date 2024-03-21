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

void matrix_init(half* a, int M, int N){
    // A(M,N)
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = __float2half((float)(rand()) / (float)(rand()));
        }
    }
}

bool Equal(float a, float b, float epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// bool Equal(half a, half b, float epsilon = 1e-5) {
//     return std::abs(a - b) < epsilon;
// }

static void PrintAssert(bool condition, float a, float b) {
    if (!condition) {
        printf("%f is not equal to %f\n", a, b);
    }
}

static void PrintAssert(bool condition, half a, half b) {
    if (!condition) {
        printf("%f is not equal to %f\n", __half2float(a), __half2float(b));
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
            PrintAssert(Equal(c[i*K + j],h_c_check[i*K + j]), c[i*K + j], h_c_check[i*K + j]);
    }

    free(h_c_check);
    cudaFree(c_check);
}

void checkResult(half* a, half* b, half* c, size_t bytes, int M, int N, int K){

    half* c_check;
    cudaMalloc(&c_check, bytes);

    cublasHandle_t handle;
	cublasCreate(&handle);
	__half alpha = __float2half_rn(1.0f);
	__half beta = __float2half_rn(0.0f);
    // Calculate: c = (alpha*a) * b + (beta*c)
	cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
    &alpha, b, K, a, N, &beta, c_check, K);

    half* h_c_check = (half*)malloc(bytes);
    cudaMemcpy(h_c_check, c_check, bytes, cudaMemcpyDeviceToHost);

    for (int i=0; i < M; i++){
        for (int j=0; j < K; j++)
            PrintAssert(Equal(c[i*K + j],h_c_check[i*K + j]),
          c[i*K + j], h_c_check[i*K + j]);
    }

    free(h_c_check);
    cudaFree(c_check);
}