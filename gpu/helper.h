#include <iostream>
#include <random>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::default_random_engine random_engine(0);
std::uniform_real_distribution<float> uniform_dist(-256, 256);

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void matrix_init(float* a, int M, int N){
    // A(M,N)
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = uniform_dist(random_engine);
        }
    }
}

void matrix_init(half* a, int M, int N){
    // A(M,N)
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = __float2half(uniform_dist(random_engine));
        }
    }
}

bool FloatEqual(float a, float b, float epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

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
    checkCuda(cudaMalloc(&c_check, bytes));

    cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
    // Calculate: c = (alpha*a) * b + (beta*c)
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
    &alpha, b, K, a, N, &beta, c_check, K);

    float* h_c_check = (float*)malloc(bytes);
    checkCuda(cudaMemcpy(h_c_check, c_check, bytes, cudaMemcpyDeviceToHost));

    for (int i=0; i < M; i++){
        for (int j=0; j < K; j++)
            PrintAssert(FloatEqual(c[i*K + j],h_c_check[i*K + j]), c[i*K + j], h_c_check[i*K + j]);
    }

    free(h_c_check);
    checkCuda(cudaFree(c_check));
}

void checkResult(half* a, half* b, float* c, size_t bytes, int M, int N, int K){

    float* c_check;
    checkCuda(cudaMalloc(&c_check, bytes));

    cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
    &alpha, b, CUDA_R_16F, K, a, CUDA_R_16F, N, &beta, c_check, CUDA_R_32F, K, CUDA_R_32F,
    static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

    float* h_c_check = (float*)malloc(bytes);
    checkCuda(cudaMemcpy(h_c_check, c_check, bytes, cudaMemcpyDeviceToHost));

    for (int i=0; i < M; i++){
        for (int j=0; j < K; j++)
            PrintAssert(FloatEqual(c[i*K + j],h_c_check[i*K + j]),
          c[i*K + j], h_c_check[i*K + j]);
    }

    free(h_c_check);
    checkCuda(cudaFree(c_check));
}