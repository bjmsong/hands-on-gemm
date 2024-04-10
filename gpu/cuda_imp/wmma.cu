#include <cuda_runtime.h>
#include <mma.h>
#include "../helper.h" 

template <int WMMA_M, int WMMA_N, int WMMA_K, typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm(
    half const* A, half const* B, float* C, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    // Tile using a 2D grid.
    // Determine the warp 2D index.
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpK{blockIdx.y * blockDim.y + threadIdx.y};

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_M, WMMA_N, half, WMMA_FRAG_LAYOUT_A> a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, WMMA_FRAG_LAYOUT_B> b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_M, WMMA_K, float> acc_frag{};

    // Make sure the accumulator starts from 0.
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<float>(0));

    // Loop over N.
    for (uint32_t ni{0}; ni < n; ni += WMMA_N)
    {
        // Determine the first element of the mma matrices on the linear memory.
        // Matrix A mma matrix
        uint32_t const matrix_mma_a_col_idx{ni};
        uint32_t const matrix_mma_a_row_idx{warpK * WMMA_K};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_col_idx{warpM * WMMA_M};
        uint32_t const matrix_mma_b_row_idx{ni};

        // Determine the memory address of the first element of the mma
        half const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx * lda + matrix_mma_a_col_idx};
        half const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx * ldb + matrix_mma_b_col_idx};
        // Load the mma matrix inputs.
        nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
        nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // write result to HBM
    uint32_t const matrix_mma_c_col_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_row_idx{warpK * WMMA_K};

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < k)
    {
        float* matrix_mma_c_mptr{C + matrix_mma_c_row_idx * ldc + matrix_mma_c_col_idx};
        // Store the output
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, acc_frag, ldc, nvcuda::wmma::mem_row_major);
    }
}

void launch_wmma_mm(half* a, half* b, float* c, uint32_t M, uint32_t N, uint32_t K){
    uint32_t const lda{N};
    uint32_t const ldb{K};
    uint32_t const ldc{K};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // Block size of 128x4 means we have 16 (4x4) warps,
    // each warp computes a 16x16 output tile,
    // and a block computes a 64x64 output tile.
    // Each block has 4x4 warps, totalling 4x4x32 threads.
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    gridDim.x = (M + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (K + WMMA_K * num_warps_y - 1) / (WMMA_K * num_warps_y);

    wmma_gemm<WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::row_major, nvcuda::wmma::row_major>
    <<<gridDim, blockDim>>>(a, b, c, M, N, K, lda, ldb, ldc);
}

int main(int argc, char** argv){
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    size_t bytes_a = M * N * sizeof(half);
    size_t bytes_b = N * K * sizeof(half);
    size_t bytes_c = M * K * sizeof(float);

    half* h_a = (half*)malloc(bytes_a);
    half* h_b = (half*)malloc(bytes_b);
    float* h_c = (float*)malloc(bytes_c);

    matrix_init(h_a, M, N);
    matrix_init(h_b, N, K);

    half *d_a, *d_b;
    float *d_c;
    checkCuda(cudaMalloc(&d_a, bytes_a));
    checkCuda(cudaMalloc(&d_b, bytes_b));
    checkCuda(cudaMalloc(&d_c, bytes_c));

    checkCuda(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

    int WARMUP_TIMES = 100;
    for (int n_count=0; n_count < WARMUP_TIMES; n_count++){
        launch_wmma_mm(d_a, d_b, d_c, M, N, K);
    }
    
    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&end));
    checkCuda(cudaEventRecord(start));
    cudaDeviceSynchronize();
    int EXECUTE_TIMES = 100;
    for (int n_count=0; n_count < EXECUTE_TIMES; n_count++){
        launch_wmma_mm(d_a, d_b, d_c, M, N, K);
    }
    cudaDeviceSynchronize();
    checkCuda(cudaEventRecord(end));
    checkCuda(cudaEventSynchronize(start));
    checkCuda(cudaEventSynchronize(end));

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    checkCuda(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));
    // checkResult(d_a, d_b, h_c, bytes_c, M, N, K);

    free(h_a);
    free(h_b);
    free(h_c);

    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    printf("spend %f ms with size of (%d, %d, %d)\n", msec/EXECUTE_TIMES, M, N, K);
    printf("Computational Throughput: %f TFLOPS\n", (float)2*M*N*K*1e-9*EXECUTE_TIMES/msec);
}