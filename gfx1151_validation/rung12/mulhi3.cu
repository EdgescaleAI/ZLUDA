#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// RUNG 12 surgical test: the two halves of the localized ssm_conv d_conv=3 bug.
// (1) mul.hi.u64 n, 0xAAAAAAAAAAAAAAAB ; shr 1  == n/3  (the divide-by-3 magic).
// (2) the WALKING-BASE .local circular-index pattern nvcc actually emits:
//     addr = base + 4*idx - 12*(idx/3)  (== base + 4*(idx%3)), via a per-iter
//     base that advances 8 bytes over a 12-byte depot. Forced into .local with a
//     volatile sink so nvcc can't keep it in registers/const-fold (mirrors the
//     128-thread spill that the single-thread replica avoided).

__global__ void mulhi3(uint64_t* out, int n) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < n) {
        uint64_t q;
        uint64_t v = (uint64_t)t;
        asm("{ .reg .u64 hi; mul.hi.u64 hi, %1, 0xAAAAAAAAAAAAAAAB; shr.u64 %0, hi, 1; }"
            : "=l"(q) : "l"(v));
        out[t] = q;          // expect t/3
    }
}

// circular 3-slot buffer indexed by a runtime modulo, forced to .local
__global__ void circ3(const float* xin, const float* w, float* yout, int n_t) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= 1) return;
    volatile float vsink = 0.f;
    float x[3];
    // i==0 fill
    x[0]=xin[0]; x[1]=xin[1]; x[2]=xin[2];
    vsink += x[0];                       // force spill
    for (int64_t i = 0; i < n_t; i++) {
        if (i != 0) x[(i-1)%3] = xin[i+2];
        float s = 0.f;
        for (int j=0;j<3;j++) s += x[(i+j)%3]*w[j];
        yout[i] = s;
        vsink += s;
    }
    (void)vsink;
}

int main(){
    const int N=24;
    uint64_t *dq, hq[N];
    cudaMalloc(&dq, N*sizeof(uint64_t));
    mulhi3<<<1,32>>>(dq, N);
    cudaDeviceSynchronize();
    cudaMemcpy(hq, dq, N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    int mbad=0;
    for(int t=0;t<N;t++){ if(hq[t]!=(uint64_t)(t/3)){ if(mbad<8) printf("MULHI3BAD t=%d gpu=%llu exp=%d\n",t,(unsigned long long)hq[t],t/3); mbad++; } }
    printf("MULHI3 mbad=%d/%d\n", mbad, N);

    const int n_t=4;
    float hx[6]={ 0.7f,-0.3f, 0.5f, 0.9f,-0.8f, 0.2f}, hw[3]={0.25f,-0.5f,0.75f}, hy[8];
    float *dx,*dw,*dy; cudaMalloc(&dx,6*4); cudaMalloc(&dw,3*4); cudaMalloc(&dy,n_t*4);
    cudaMemcpy(dx,hx,6*4,cudaMemcpyHostToDevice); cudaMemcpy(dw,hw,3*4,cudaMemcpyHostToDevice);
    circ3<<<1,32>>>(dx,dw,dy,n_t);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(hy,dy,n_t*4,cudaMemcpyDeviceToHost);
    int cbad=0;
    for(int i=0;i<n_t;i++){ float ref=0; for(int j=0;j<3;j++) ref+=hx[i+j]*hw[j];
        if(fabsf(hy[i]-ref)>1e-4f){ printf("CIRC3BAD i=%d gpu=%.6f ref=%.6f\n",i,hy[i],ref); cbad++; }
        else printf("CIRC3 ok i=%d gpu=%.6f ref=%.6f\n",i,hy[i],ref); }
    printf("CIRC3 cbad=%d/%d cudaerr=%d(%s)\n", cbad, n_t, (int)e, cudaGetErrorString(e));
    return 0;
}
