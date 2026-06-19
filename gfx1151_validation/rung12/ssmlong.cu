#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

// =============================================================================
// RUNG 12 reproducer #2 — isolate ggml ssm_conv_long_token_f32 (the n_t>32 path)
// which is what test-backend-ops SSM_CONV's FAILING cases hit (reproducer #1
// mod_iso proved the SHORT n_t<=32 path is correct to ~1e-7). The long path is a
// DIFFERENT kernel: extern __shared__ smem + __syncthreads() + a cooperative load
// loop with signed int div/mod by a compile-time const. We replicate it VERBATIM
// (+ a simpler shared-mem self-test) so a failure localizes to ONE construct.
// =============================================================================

#define D_INNER 128      // == threads (one bidy block); real nr is a multiple of 128
#define SPLIT_NT 32

// --- (A) verbatim replica of ssm_conv_long_token_f32 (split_d_inner=128) -----
template <size_t split_d_inner, size_t d_conv, int64_t split_n_t>
__global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                        const int src0_nb1, const int src1_nb1,
                                        float * __restrict__ dst, const int dst_nb1, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidz * split_n_t * sizeof(float));
    const float * w_block = src1;
    float *       y_block = (float *) ((char *) dst + bidz * split_n_t * dst_nb1);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }
        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) w[j] = w_block[tid * stride_w + j];

    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) sumf += smem[tid * n_cols + i + j] * w[j];
        y_block[i * stride_y + tid] = sumf;
    }
}

// --- (B) minimal shared-mem + __syncthreads sanity: each thread writes its slot,
// syncs, reads its NEIGHBOR's slot. If extern-smem or __syncthreads is broken,
// this reads stale/garbage. -----------------------------------------------------
__global__ void smem_sanity(float * out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    s[tid] = (float)(tid * 2 + 1);
    __syncthreads();
    int nb = (tid + 1) % blockDim.x;
    out[blockIdx.x * blockDim.x + tid] = s[nb];
}

template <size_t d_conv>
void run_long(const char* tag) {
    const int64_t n_t = 64;                  // > 32 -> long path, 2 z-blocks
    const int xn_cols = d_conv - 1 + n_t;    // src0 first dim
    const int d_inner = D_INNER;
    // src0: [xn_cols, d_inner] row-major per ggml (nb1 = xn_cols*4); src1(w): [d_conv, d_inner]
    float *hx = new float[(size_t)xn_cols * d_inner];
    float *hw = new float[(size_t)d_conv * d_inner];
    for (int r = 0; r < d_inner; r++)
        for (int c = 0; c < xn_cols; c++) hx[r*xn_cols + c] = 0.3f + 0.017f*c + 0.001f*r;
    for (int r = 0; r < d_inner; r++)
        for (size_t j = 0; j < d_conv; j++) hw[r*d_conv + j] = 0.1f*(j+1) - 0.15f + 0.0001f*r;
    float *hy = new float[(size_t)n_t * d_inner];

    float *dx,*dw,*dy;
    cudaMalloc(&dx, (size_t)xn_cols*d_inner*sizeof(float));
    cudaMalloc(&dw, (size_t)d_conv*d_inner*sizeof(float));
    cudaMalloc(&dy, (size_t)n_t*d_inner*sizeof(float));
    cudaMemcpy(dx, hx, (size_t)xn_cols*d_inner*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw, (size_t)d_conv*d_inner*sizeof(float), cudaMemcpyHostToDevice);

    const int threads = D_INNER;
    const int64_t split_n_t = SPLIT_NT;
    dim3 blocks(1, 1, (n_t + split_n_t - 1)/split_n_t);
    size_t smem = threads * (d_conv - 1 + split_n_t) * sizeof(float);
    const int src0_nb1 = xn_cols * sizeof(float);
    const int src1_nb1 = d_conv  * sizeof(float);
    const int dst_nb1  = d_inner * sizeof(float);   // dst layout: [d_inner, n_t], nb1 stride over tokens
    ssm_conv_long_token_f32<D_INNER, d_conv, SPLIT_NT><<<blocks, threads, smem>>>(
        dx, dw, src0_nb1, src1_nb1, dy, dst_nb1, n_t);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(hy, dy, (size_t)n_t*d_inner*sizeof(float), cudaMemcpyDeviceToHost);

    int bad=0; float worst=0; int worst_i=-1,worst_ch=-1;
    for (int ch = 0; ch < d_inner; ch++) {
        for (int i = 0; i < n_t; i++) {
            float ref = 0.f;
            for (size_t j = 0; j < d_conv; j++) ref += hx[ch*xn_cols + i + j] * hw[ch*d_conv + j];
            float got = hy[i*d_inner + ch];
            float d = fabsf(got - ref);
            if (d > worst) { worst=d; worst_i=i; worst_ch=ch; }
            if (d > 1e-4f) { if (bad<6) printf("LONGBAD[%s] ch=%d i=%d gpu=%.6f ref=%.6f\n", tag, ch, i, got, ref); bad++; }
        }
    }
    printf("SSMLONG[%s d_conv=%zu] bad=%d/%d worst=%.6e (i=%d ch=%d) cudaerr=%d(%s)\n",
           tag, d_conv, bad, (int)(n_t*d_inner), worst, worst_i, worst_ch, (int)e, cudaGetErrorString(e));
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    delete[] hx; delete[] hw; delete[] hy;
}

int main(){
    // (B) shared-mem + __syncthreads sanity
    const int T=128;
    float *dso, hso[T];
    cudaMalloc(&dso, T*sizeof(float));
    smem_sanity<<<1,T,T*sizeof(float)>>>(dso, T);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(hso, dso, T*sizeof(float), cudaMemcpyDeviceToHost);
    int sbad=0; for(int t=0;t<T;t++){ float ref=(float)(((t+1)%T)*2+1); if(fabsf(hso[t]-ref)>1e-4f){ if(sbad<6) printf("SMEMBAD t=%d gpu=%.1f ref=%.1f\n",t,hso[t],ref); sbad++; } }
    printf("SMEM_SANITY sbad=%d/%d cudaerr=%d(%s)\n", sbad, T, (int)e, cudaGetErrorString(e));
    cudaFree(dso);

    // (A) verbatim long-token kernel for the 3 failing d_conv values
    run_long<3>("d3");
    run_long<4>("d4");
    run_long<9>("d9");
    return 0;
}
