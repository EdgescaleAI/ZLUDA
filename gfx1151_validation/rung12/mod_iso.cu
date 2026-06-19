#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

// =============================================================================
// RUNG 12 — isolate the ggml ssm_conv numeric FAIL (ERR ~0.045-0.10) through ZLUDA
// on gfx1151, away from test-backend-ops noise. Three escalating kernels, each
// self-checks GPU-vs-CPU so a failure localizes to ONE arithmetic construct.
//
// Real kernel (ggml-cuda/ssm-conv.cu, ssm_conv_f32<apply_silu,split_d_inner,d_conv>):
//   float x[d_conv];                       // PER-THREAD REGISTER/LOCAL array
//   for (i=0..n_t) {
//     if (i==0) for j: x[j] = x_block[tid*stride_x + j];
//     else      x[(i-1)%d_conv] = x_block[tid*stride_x + i + d_conv - 1];
//     #pragma unroll for j: sumf += x[(i+j)%d_conv] * w[j];   // DYNAMIC modulo index into LOCAL array
//     sumf += b;  y = silu?silu(sumf):sumf;
//   }
// Net: y[i] = sum_j x_in[i+j]*w[j] + b  (a length-d_conv sliding window).
// =============================================================================

// (a) the u64 magic-division modulo (i+j)%D that the compile-time-const d_conv emits
template<size_t D>
__global__ void modk(uint64_t* out, int n) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < n) {
        uint64_t i = (uint64_t)t;
        out[t] = (i + 0) % D + 10*((i + 1) % D) + 100*((i + 2) % D);
    }
}

// (b) conv inner loop but reading a GLOBAL array (simpler than the real kernel)
template<size_t D>
__global__ void miniconv(const float* x, const float* w, float* out, int n) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < n) {
        float s = 0.f;
        for (size_t j = 0; j < D; j++) s += x[(t + j) % D] * w[j];
        out[t] = s;
    }
}

// (c) EXACT replica of ssm_conv: per-thread LOCAL register array x[D] with the
// sliding-window update + DYNAMIC modulo index into the local array. This is the
// construct that lowers to .local memory / selp chains and is the prime suspect.
template<size_t D>
__global__ void slidewin(const float* x_in, const float* w_in, float b, float* y, int n_t) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;  // one thread does the whole sequence
    if (tid != 0) return;
    float x[D];
    float w[D];
    for (size_t j = 0; j < D; j++) w[j] = w_in[j];
    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.f;
        if (i == 0) {
            for (size_t j = 0; j < D; j++) x[j] = x_in[j];
        } else {
            x[(i - 1) % D] = x_in[i + D - 1];
        }
#pragma unroll
        for (size_t j = 0; j < D; j++) sumf += x[(i + j) % D] * w[j];
        sumf += b;
        y[i] = sumf;
    }
}

template<size_t D>
void run_slidewin(const char* tag) {
    const int n_t = 12;
    const int xn = n_t + D - 1;
    float *hx = new float[xn], hw[D], *hy = new float[n_t];
    for (int k = 0; k < xn; k++) hx[k] = 0.5f + 0.37f*k;          // deterministic, non-trivial
    for (size_t j = 0; j < D; j++) hw[j] = 0.1f*(j+1) - 0.15f;    // mixed signs
    float bb = 0.25f;
    float *dx,*dw,*dy;
    cudaMalloc(&dx, xn*sizeof(float)); cudaMalloc(&dw, D*sizeof(float)); cudaMalloc(&dy, n_t*sizeof(float));
    cudaMemcpy(dx, hx, xn*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw, D*sizeof(float), cudaMemcpyHostToDevice);
    slidewin<D><<<1,32>>>(dx, dw, bb, dy, n_t);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(hy, dy, n_t*sizeof(float), cudaMemcpyDeviceToHost);
    int bad=0; float worst=0;
    for (int i = 0; i < n_t; i++) {
        float ref = bb; for (size_t j = 0; j < D; j++) ref += hx[i+j]*hw[j];
        float d = fabsf(hy[i]-ref); if (d>worst) worst=d;
        if (d > 1e-4f) { if (bad<8) printf("SLIDEBAD[%s] i=%d gpu=%.6f ref=%.6f\n", tag, i, hy[i], ref); bad++; }
    }
    printf("SLIDEWIN[%s D=%zu] bad=%d/%d worst=%.6e cudaerr=%d(%s)\n", tag, D, bad, n_t, worst, (int)e, cudaGetErrorString(e));
    cudaFree(dx); cudaFree(dw); cudaFree(dy); delete[] hx; delete[] hy;
}

int main(){
    const int N=64; const size_t D=3;
    uint64_t *dout, hout[N];
    cudaMalloc(&dout, N*sizeof(uint64_t));
    modk<D><<<(N+31)/32,32>>>(dout, N);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(hout, dout, N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    int badmod=0;
    for(int t=0;t<N;t++){
        uint64_t ref=(t%3)+10*((t+1)%3)+100*((t+2)%3);
        if(hout[t]!=ref){ if(badmod<8) printf("MODBAD t=%d gpu=%llu ref=%llu\n",t,(unsigned long long)hout[t],(unsigned long long)ref); badmod++; }
    }
    printf("MOD_ISO badmod=%d/%d cudaerr=%d(%s)\n",badmod,N,(int)e,cudaGetErrorString(e));

    // miniconv (global-array read)
    float hx[D]={1.5f,2.5f,3.5f}, hw[D]={0.25f,0.5f,0.75f}, *dx,*dw,*dco, hco[N];
    cudaMalloc(&dx,D*sizeof(float)); cudaMalloc(&dw,D*sizeof(float)); cudaMalloc(&dco,N*sizeof(float));
    cudaMemcpy(dx,hx,D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dw,hw,D*sizeof(float),cudaMemcpyHostToDevice);
    miniconv<D><<<(N+31)/32,32>>>(dx,dw,dco,N);
    e=cudaDeviceSynchronize();
    cudaMemcpy(hco,dco,N*sizeof(float),cudaMemcpyDeviceToHost);
    int badc=0; float worst=0;
    for(int t=0;t<N;t++){
        float ref=0; for(size_t j=0;j<D;j++) ref+=hx[(t+j)%D]*hw[j];
        float d=fabsf(hco[t]-ref); if(d>worst)worst=d;
        if(d>1e-4f){ if(badc<8) printf("CONVBAD t=%d gpu=%.6f ref=%.6f\n",t,hco[t],ref); badc++; }
    }
    printf("MINICONV badc=%d/%d worst=%.6e cudaerr=%d(%s)\n",badc,N,worst,(int)e,cudaGetErrorString(e));

    // slidewin (EXACT ssm_conv structure: local register array + dynamic modulo index)
    run_slidewin<3>("d3");
    run_slidewin<4>("d4");
    return 0;
}
