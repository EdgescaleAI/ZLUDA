// Rung 9a — a REAL nvcc-compiled CUDA binary (CUDA runtime API) run through ZLUDA on gfx1151.
// All prior rungs drove ZLUDA via Python ctypes->libcuda + nvrtc PTX strings + cuBLAS redirect.
// This is the canonical ZLUDA path that none of them exercised: nvcc emits a fatbin embedded in
// a normal ELF; cudart's __cudaRegisterFatBinary + cudaLaunchKernel call into the driver API,
// which ZLUDA intercepts (cuModuleLoadData of nvcc's PTX -> PTX->LLVM->gfx1151 JIT -> cuLaunchKernel).
//
// MUST compile PTX-bearing (virtual arch, code=compute_XX) — ZLUDA translates PTX, not SASS.
//   nvcc -gencode arch=compute_70,code=compute_70 vecadd.cu -o vecadd
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    printf("CUDA_ERR %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); return 2; } } while(0)

int main() {
    const int n = 1 << 16;            // 65536 elements
    const size_t sz = n * sizeof(float);
    float *ha = (float*)malloc(sz), *hb = (float*)malloc(sz), *hc = (float*)malloc(sz);
    for (int i = 0; i < n; i++) { ha[i] = i * 0.5f; hb[i] = i * 2.0f - 3.0f; }

    float *da, *db, *dc;
    CK(cudaMalloc(&da, sz)); CK(cudaMalloc(&db, sz)); CK(cudaMalloc(&dc, sz));
    CK(cudaMemcpy(da, ha, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb, sz, cudaMemcpyHostToDevice));

    vadd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { printf("LAUNCH_ERR %s\n", cudaGetErrorString(le)); return 3; }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hc, dc, sz, cudaMemcpyDeviceToHost));

    // Verify against the host reference — a hollow pass (wrong output) is the cardinal sin.
    double max_abs = 0.0; int n_bad = 0;
    for (int i = 0; i < n; i++) {
        float ref = ha[i] + hb[i];
        double d = fabs((double)hc[i] - (double)ref);
        if (d > max_abs) max_abs = d;
        if (d > 1e-5) n_bad++;
    }
    printf("VECADD n=%d max_abs=%.3e n_bad=%d sample c[7]=%.4f (ref %.4f)\n",
           n, max_abs, n_bad, hc[7], ha[7] + hb[7]);
    printf("%s\n", (n_bad == 0 && max_abs < 1e-4) ? "RUNG9A_VERDICT PASS" : "RUNG9A_VERDICT FAIL");
    cudaFree(da); cudaFree(db); cudaFree(dc); free(ha); free(hb); free(hc);
    return 0;
}
