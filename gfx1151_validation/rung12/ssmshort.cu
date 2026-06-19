#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// =============================================================================
// RUNG 12 reproducer #3 — VERBATIM ggml ssm_conv_f32 SHORT kernel at the EXACT
// failing test-backend-ops config: d_conv=3, d_inner=1024, n_t=4, n_s=1 (the ONLY
// SSM_CONV shape that FAILs through ZLUDA, NMSE ~0.09). Reproducers #1/#2 (smooth
// ramp, single channel / long kernel) passed — this drives the real multi-block /
// 128-thread / random-input path and dumps per-(channel,token) GPU-vs-CPU.
// =============================================================================

#define GGML_CUDA_RESTRICT __restrict__   // compute_70: PDL off -> __restrict__ (matches common.cuh)

// verbatim ssm_conv_f32 (pdl_lc/pdl_sync are no-ops on non-Hopper, omitted)
template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    (void) src0_nb0;
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }
    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;
        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) x[j] = x_block[tid * stride_x + j];
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) sumf += x[(i + j) % d_conv] * w[j];
        sumf += b;
        y_block[i * stride_y + tid] = sumf;
    }
}

template <size_t D>
void run(const char* tag, int d_inner, int n_t) {
    const int threads = 128;
    const int xcols = n_t + D - 1;                 // ne_a[0]
    // src0 layout [xcols, d_inner]: nb0=4, nb1=xcols*4, nb2=xcols*d_inner*4
    // src1 (w)    [D, d_inner]:     nb1=D*4
    // dst         [d_inner, n_t]:   nb0=4, nb1=d_inner*4, nb2=d_inner*n_t*4
    size_t xn = (size_t)xcols * d_inner, wn = (size_t)D * d_inner, yn = (size_t)d_inner * n_t;
    float *hx = new float[xn], *hw = new float[wn], *hy = new float[yn];
    srand(1234);
    for (size_t k=0;k<xn;k++) hx[k] = 2.0f*((float)rand()/RAND_MAX) - 1.0f;   // uniform[-1,1] like test
    for (size_t k=0;k<wn;k++) hw[k] = 2.0f*((float)rand()/RAND_MAX) - 1.0f;
    float *dx,*dw,*dy;
    cudaMalloc(&dx, xn*sizeof(float)); cudaMalloc(&dw, wn*sizeof(float)); cudaMalloc(&dy, yn*sizeof(float));
    cudaMemcpy(dx,hx,xn*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dw,hw,wn*sizeof(float),cudaMemcpyHostToDevice);

    dim3 blocks(1, (d_inner + threads - 1)/threads, 1);
    int src0_nb1 = xcols*sizeof(float), src0_nb2 = xcols*d_inner*sizeof(float);
    int src1_nb1 = D*sizeof(float);
    int dst_nb0 = sizeof(float), dst_nb1 = d_inner*sizeof(float), dst_nb2 = d_inner*n_t*sizeof(float);
    ssm_conv_f32<false, 128, D><<<blocks, threads>>>(dx, dw, nullptr, sizeof(float), src0_nb1, src0_nb2, src1_nb1,
                                                     dy, dst_nb0, dst_nb1, dst_nb2, n_t);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(hy, dy, yn*sizeof(float), cudaMemcpyDeviceToHost);

    // reference: channel ch, token i -> sum_j x[ch*xcols + i + j]*w[ch*D + j]
    int bad=0; double se=0, sref=0; float worst=0; int wi=-1,wch=-1;
    for (int ch=0; ch<d_inner; ch++) {
        for (int i=0;i<n_t;i++) {
            float ref=0; for (size_t j=0;j<D;j++) ref += hx[ch*xcols + i + j]*hw[ch*D + j];
            float got = hy[i*d_inner + ch];
            float d = fabsf(got-ref); se += (double)d*d; sref += (double)ref*ref;
            if (d>worst){worst=d;wi=i;wch=ch;}
            if (d>1e-4f){ if(bad<10) printf("BAD[%s] ch=%d i=%d gpu=%.6f ref=%.6f diff=%.4f\n",tag,ch,i,got,ref,d); bad++; }
        }
    }
    double nmse = se/(sref>0?sref:1);
    printf("SSMSHORT[%s D=%zu d_inner=%d n_t=%d] bad=%d/%zu worst=%.4e(i=%d ch=%d) NMSE=%.6f cudaerr=%d(%s)\n",
           tag, D, d_inner, n_t, bad, yn, worst, wi, wch, nmse, (int)e, cudaGetErrorString(e));
    cudaFree(dx);cudaFree(dw);cudaFree(dy); delete[] hx; delete[] hw; delete[] hy;
}

// control: same kernel, single thread/block (channel 0 only) vs 128-thread launch
template <size_t D>
void run_threads(const char* tag, int d_inner, int n_t, int threads) {
    const int xcols = n_t + D - 1;
    size_t xn=(size_t)xcols*d_inner, wn=(size_t)D*d_inner, yn=(size_t)d_inner*n_t;
    float *hx=new float[xn],*hw=new float[wn],*hy=new float[yn];
    srand(1234);
    for(size_t k=0;k<xn;k++) hx[k]=2.0f*((float)rand()/RAND_MAX)-1.0f;
    for(size_t k=0;k<wn;k++) hw[k]=2.0f*((float)rand()/RAND_MAX)-1.0f;
    float *dx,*dw,*dy; cudaMalloc(&dx,xn*4);cudaMalloc(&dw,wn*4);cudaMalloc(&dy,yn*4);
    cudaMemset(dy,0,yn*4);
    cudaMemcpy(dx,hx,xn*4,cudaMemcpyHostToDevice); cudaMemcpy(dw,hw,wn*4,cudaMemcpyHostToDevice);
    int src0_nb1=xcols*4,src0_nb2=xcols*d_inner*4,src1_nb1=D*4,dst_nb0=4,dst_nb1=d_inner*4,dst_nb2=d_inner*n_t*4;
    // template split_d_inner stays 128 for addressing; launch with `threads` actual threads
    dim3 blocks(1,1,1);
    ssm_conv_f32<false,128,D><<<blocks,threads>>>(dx,dw,nullptr,4,src0_nb1,src0_nb2,src1_nb1,dy,dst_nb0,dst_nb1,dst_nb2,n_t);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(hy,dy,yn*4,cudaMemcpyDeviceToHost);
    int bad=0; float worst=0;
    int chk = threads<d_inner?threads:d_inner;   // only channels [0,threads) were computed
    for(int ch=0;ch<chk;ch++) for(int i=0;i<n_t;i++){
        float ref=0; for(size_t j=0;j<D;j++) ref+=hx[ch*xcols+i+j]*hw[ch*D+j];
        float d=fabsf(hy[i*d_inner+ch]-ref); if(d>worst)worst=d;
        if(d>1e-4f){ if(bad<4)printf("  T%d BAD ch=%d i=%d gpu=%.6f ref=%.6f\n",threads,ch,i,hy[i*d_inner+ch],ref); bad++; }
    }
    printf("SSMSHORT_THREADS[%s threads=%d D=%zu n_t=%d] bad=%d/%d worst=%.4e cudaerr=%d\n",tag,threads,D,n_t,bad,chk*n_t,worst,(int)e);
    cudaFree(dx);cudaFree(dw);cudaFree(dy);delete[]hx;delete[]hw;delete[]hy;
}

int main(){
    // control: thread-count sweep on the SAME kernel/data (channel 0 in all)
    run_threads<3>("ctl", 1024, 4, 1);
    run_threads<3>("ctl", 1024, 4, 2);
    run_threads<3>("ctl", 1024, 4, 32);
    run_threads<3>("ctl", 1024, 4, 128);
    run<3>("FAILcfg_1024", 1024, 4);   // the exact failing case
    run<3>("d3_ix512",      512, 4);   // smaller d_inner, same n_t
    run<3>("d3_nt12",      1024, 12);  // same d_inner, larger n_t (my passing replica's n_t)
    run<3>("d3_nt1",       1024, 1);   // n_t=1 (passing case)
    run<4>("d4_nt4",       1024, 4);   // d_conv=4 n_t=4 (passes in real test)
    return 0;
}
