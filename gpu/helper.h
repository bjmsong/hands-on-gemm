#include <Eigen/Dense>
using Eigen::MatrixXd;

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

// TODO: replace with cublas
void checkResult(float* a, float* b, float* c, int M, int N, int K){
    MatrixXd a_e(M,N);
    MatrixXd b_e(N,K);
    MatrixXd c_e(M,K);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            a_e(i, j) = a[i * N + j];
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            b_e(i, j) = b[i * K + j];
        }
    }

    c_e = a_e * b_e;

    for (int i=0; i < M; i++){
        for (int j=0; j < K; j++)
            PrintAssert(floatEqual(c[i*K + j],c_e(i, j)), c[i*K + j], c_e(i, j));
    }
}
