// Phase-2 differential: cuBLAS sgemm through ZLUDA (libcuda + libcublas) on gfx1151.
// Loads driver API + cuBLAS via dlsym (no CUDA toolkit/headers needed).
// Computes C(MxN) = A(MxK) @ B(KxN), all row-major fp32, writes C.bin.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int CUresult;
typedef int CUdevice;
typedef uintptr_t CUdeviceptr;
typedef void* CUcontext;
typedef void* cublasHandle_t;

#define M 8
#define K 16
#define N 8

static void* must(void* h, const char* s){ void* p=dlsym(h,s); if(!p){fprintf(stderr,"missing sym %s: %s\n",s,dlerror()); exit(3);} return p; }
#define CK(call) do{ CUresult _r=(call); if(_r!=0){fprintf(stderr,"%s -> %d\n",#call,_r); exit(4);} }while(0)

static float* readbin(const char* p,int n){ FILE* f=fopen(p,"rb"); if(!f){perror(p);exit(2);} float* b=malloc(n*4); fread(b,4,n,f); fclose(f); return b; }

int main(){
    void* cuda = dlopen("libcuda.so",RTLD_NOW|RTLD_GLOBAL);
    if(!cuda){ fprintf(stderr,"dlopen libcuda.so: %s\n",dlerror()); return 1; }
    void* blas = dlopen("libcublas.so",RTLD_NOW|RTLD_GLOBAL);
    if(!blas){ fprintf(stderr,"dlopen libcublas.so: %s\n",dlerror()); return 1; }

    CUresult (*cuInit)(unsigned)=must(cuda,"cuInit");
    CUresult (*cuDeviceGet)(CUdevice*,int)=must(cuda,"cuDeviceGet");
    CUresult (*cuCtxCreate)(CUcontext*,unsigned,CUdevice)=must(cuda,"cuCtxCreate_v2");
    CUresult (*cuMemAlloc)(CUdeviceptr*,size_t)=must(cuda,"cuMemAlloc_v2");
    CUresult (*cuMemcpyHtoD)(CUdeviceptr,const void*,size_t)=must(cuda,"cuMemcpyHtoD_v2");
    CUresult (*cuMemcpyDtoH)(void*,CUdeviceptr,size_t)=must(cuda,"cuMemcpyDtoH_v2");
    CUresult (*cuCtxSynchronize)(void)=must(cuda,"cuCtxSynchronize");

    int (*cublasCreate)(cublasHandle_t*)=must(blas,"cublasCreate_v2");
    int (*cublasSgemm)(cublasHandle_t,int,int,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int)=must(blas,"cublasSgemm_v2");

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev,0));
    CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev));

    float* A=readbin("A.bin",M*K);   // MxK row-major
    float* B=readbin("B.bin",K*N);   // KxN row-major
    float* C=calloc(M*N,4);

    CUdeviceptr dA,dB,dC;
    CK(cuMemAlloc(&dA,M*K*4)); CK(cuMemAlloc(&dB,K*N*4)); CK(cuMemAlloc(&dC,M*N*4));
    CK(cuMemcpyHtoD(dA,A,M*K*4)); CK(cuMemcpyHtoD(dB,B,K*N*4));

    cublasHandle_t h; if(cublasCreate(&h)){fprintf(stderr,"cublasCreate failed\n");return 5;}
    float alpha=1.f, beta=0.f;
    // row-major C=A*B via column-major: C^T = B^T*A^T -> sgemm(N,N, N,M,K, B,N, A,K, C,N)
    int st=cublasSgemm(h, 0,0, N,M,K, &alpha, (const float*)dB,N, (const float*)dA,K, &beta, (float*)dC,N);
    if(st){fprintf(stderr,"cublasSgemm -> %d\n",st);return 6;}
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(C,dC,M*N*4));

    FILE* o=fopen("C.bin","wb"); fwrite(C,4,M*N,o); fclose(o);
    fprintf(stderr,"OK wrote C.bin (%dx%d)\n",M,N);
    return 0;
}
