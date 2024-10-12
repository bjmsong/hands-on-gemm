/*
nvcc -O3 --use_fast_math -lcublas -lcublasLt -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward

OMP_NUM_THREADS=32 ./matmul_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <float.h>
#include <utility>
#include "../helper.h" 

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp
    // dL/dinp(B, T, C) = dout(B, T, OC) * weight(OC, C)
    // Batch GEMM, no transpose
    // parallelize over B,T
    // utilize cache locality of weight, dinp
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;   // pointer to the t th row
            float* dinp_bt = dinp + b * T * C + t * C;     // pointer to the t th row
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;    // pointer to the o th row
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }

    // backward into weight/bias
    // dL/dweight(OC, C) = dout^T(OC, BT) * inp(BT, C)
    // GEMM, dout need to be transposed
    // dL/dbias(OC) = sum()
    // parallelize over output channels OC
    // utilize cache locality
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != NULL){dbias[o] = sum;}
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel to backpropagate only the bias, it's just a sum :'(
__global__ void matmul_backward_bias_kernel_naive(float* dbias, const float* dout, int B, int T, int OC) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < OC) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                sum += dout[b * T * OC + t * OC + o];
            }
        }
        dbias[o] = sum;
    }
}

// use shared memory and coarsening + reductions
__global__ void matmul_backward_bias_kernel_faster(float* dbias, const float* dout, int B, int T, int OC) {
    extern __shared__ float shared[];
    int o = blockIdx.x; // range [0, OC)
    int tid = threadIdx.x; // range [0, block_size)
    int block_size = blockDim.x;
    const float* x = dout + o;
    // thread coarsening
    double sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    shared[tid] = (float) sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = shared[0];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC) {
    float alpha = 1.0f;
    float beta = 1.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // for reference the API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)

    // recall the forward pass was calculated with alpha = 1.0f, beta = 0.0f as:
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);

    cublasHandle_t cublas_handle;
    // backward to input
    checkcuBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &alpha, weight, C, dout, OC, &beta, dinp, C));
    // backward to weight
    checkcuBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &alpha, inp, C, dout, OC, &beta, dweight, C));
    // backward to bias, if given
    if (dbias != NULL) {

        // sum over B,T using matrix vector multiplication with cuBLAS
        // for reference this API is:
        // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        //                    int m, int n,
        //                    const float           *alpha,
        //                    const float           *A, int lda,
        //                    const float           *x, int incx,
        //                    const float           *beta,
        //                    float           *y, int incy)
        // dout is (B,T,OC), or in 2D terms (B*T, OC)
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_N, B*T, OC, &alpha, dout, B*T, ones, 1, &beta, dbias, 1));
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, OC, B*T, &alpha, dout, OC, ones, 1, &beta, dbias, 1));

        // ugh the above isn't working...
        // let's just do naive calculation for now, fix later
        // const int block_size=128;
        // const int grid_size=(OC + block_size - 1) / block_size;
        // matmul_backward_bias_kernel<<<grid_size, block_size>>>(dbias, dout, B, T, OC);

        // bit faster
        const int block_size=512;
        dim3 block_dim(block_size);
        dim3 grid_dim(OC);
        size_t shared_mem_size = block_size * sizeof(float);
        matmul_backward_bias_kernel_faster<<<grid_dim, block_dim, shared_mem_size>>>(dbias, dout, B, T, OC);
    }
}


float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
        // arr[i] = 1;  // for debug
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    checkCuda(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    checkCuda(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    checkCuda(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        checkCuda(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        checkCuda(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        checkCuda(cudaEventRecord(stop, nullptr));
        checkCuda(cudaEventSynchronize(start));
        checkCuda(cudaEventSynchronize(stop));
        float single_call;
        checkCuda(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    checkCuda(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    checkCuda(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and its mathmodes, ensure fp32
    int enable_tf32 = 0; // use fp32 to get accurate results for checking w.r.t. CPU
    cublasHandle_t cublas_handle;
    checkcuBLAS(cublasCreate(&cublas_handle));
    printf("enable_tf32: %d\n", enable_tf32);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    checkcuBLAS(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // out(B, T, OC) = inp(B, T, C) * weight^T (C, OC) + bias(OC)
    // create host memory of random numbers
    float* dout = make_random_float(B * T * OC);    // dL/dout, L is loss value which is a scalar
    float* dinp = make_zeros_float(B * T * C); 
    float* dweight = make_zeros_float(OC * C);      
    float* dbias = make_zeros_float(OC);            
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* ones = make_ones_float(OC);

    // move to GPU
    float* d_dinp;
    float* d_dweight;
    float* d_dbias;
    float* d_dout;
    float* d_inp;
    float* d_weight;
    float* d_ones;
    checkCuda(cudaMalloc(&d_dinp, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_dweight, OC * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_dbias, OC * sizeof(float)));
    checkCuda(cudaMalloc(&d_dout, B * T * OC * sizeof(float)));
    checkCuda(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_weight, OC * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_ones, OC * sizeof(float)));
    checkCuda(cudaMemcpy(d_dinp, dinp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_dweight, dweight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_dbias, dbias, OC * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_dout, dout, B * T * OC * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weight, weight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_ones, ones, OC * sizeof(float), cudaMemcpyHostToDevice));

    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);

    // calculate the GPU version
    matmul_backward(d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);

    // compare
    printf("Checking correctness...\n");
    printf("dinp:\n");
    validate_result(d_dinp, dinp, "dinp", B * T * C, 1e-3f);
    printf("dweight:\n");
    validate_result(d_dweight, dweight, "dweight", OC * C, 1e-3f);
    printf("dbias:\n");
    validate_result(d_dbias, dbias, "dbias", OC, 1e-3f);
    printf("All results match.\n\n");

    // now benchmark the kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_backward, kernel_num,
                                          d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                          B, T, C, OC);
    printf("time %.4f ms\n", elapsed_time);

    // cleanups
    free(dinp);
    free(dweight);
    free(dbias);
    free(dout);
    free(inp);
    free(weight);
    free(ones);
    checkCuda(cudaFree(d_dinp));
    checkCuda(cudaFree(d_dweight));
    checkCuda(cudaFree(d_dbias));
    checkCuda(cudaFree(d_dout));
    checkCuda(cudaFree(d_inp));
    checkCuda(cudaFree(d_weight));
    checkCuda(cudaFree(d_ones));
    checkcuBLAS(cublasDestroy(cublas_handle));

    return 0;
}