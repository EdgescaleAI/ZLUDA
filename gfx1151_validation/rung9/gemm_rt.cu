// Rung 9a (step 2) — a real nvcc-compiled CUDA *runtime* binary calling cuBLAS sgemm, run through
// ZLUDA on gfx1151. Tests the cudart -> cuBLAS -> ZLUDA libcublas -> rocBLAS path as a native ELF
// (rung 2 proved the same math via Python ctypes; this proves the compiled-binary linkage).
//   nvcc -gencode arch=compute_70,code=compute_70 gemm_rt.cu -lcublas -o gemm_rt
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    // C(MxN) = A(MxK) * B(KxN), row-major emulated via column-major swap (compute B^T*A^T style).
    const int M = 64, K = 48, N = 32;
    float *hA = (float*)malloc(sizeof(float)*M*K);
    float *hB = (float*)malloc(sizeof(float)*K*N);
    float *hC = (float*)malloc(sizeof(float)*M*N);
    for (int i = 0; i < M*K; i++) hA[i] = ((i*7) % 13 - 6) * 0.1f;
    for (int i = 0; i < K*N; i++) hB[i] = ((i*5) % 11 - 5) * 0.1f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(float)*M*K); cudaMalloc(&dB, sizeof(float)*K*N); cudaMalloc(&dC, sizeof(float)*M*N);
    cudaMemcpy(dA, hA, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*K*N, cudaMemcpyHostToDevice);

    cublasHandle_t h; cublasCreate(&h);
    const float alpha = 1.0f, beta = 0.0f;
    // Column-major: treat row-major A(MxK),B(KxN) as col-major A'(KxM),B'(NxK).
    // C_rowmajor(MxN) = A*B  ==>  C'_colmajor(NxM) = B'_col(NxK) * A'_col(KxM).
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    // Host reference (row-major C[m*N+n])
    double max_abs = 0.0; int n_bad = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double acc = 0.0;
        for (int k = 0; k < K; k++) acc += (double)hA[m*K+k] * (double)hB[k*N+n];
        double got = hC[m*N+n];          // dC is col-major C'(NxM): C'[n + m*N] == C[m*N+n]
        double d = fabs(got - acc);
        if (d > max_abs) max_abs = d;
        if (d > 1e-3) n_bad++;
    }
    printf("GEMM_RT M=%d K=%d N=%d max_abs=%.3e n_bad=%d\n", M, K, N, max_abs, n_bad);
    printf("%s\n", (n_bad == 0 && max_abs < 1e-2) ? "RUNG9A_GEMM_VERDICT PASS" : "RUNG9A_GEMM_VERDICT FAIL");
    cublasDestroy(h); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
