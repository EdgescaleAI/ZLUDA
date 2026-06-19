// Parameterized cuBLAS sgemm through ZLUDA. argv: M K N Afile Bfile Cfile
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>
typedef int CUresult; typedef int CUdevice; typedef uintptr_t CUdeviceptr;
typedef void* CUcontext; typedef void* cublasHandle_t;
static void* must(void* h,const char* s){void* p=dlsym(h,s);if(!p){fprintf(stderr,"missing %s\n",s);exit(3);}return p;}
#define CK(c) do{CUresult r=(c);if(r){fprintf(stderr,"%s->%d\n",#c,r);exit(4);}}while(0)
static float* rd(const char* p,int n){FILE* f=fopen(p,"rb");if(!f){perror(p);exit(2);}float* b=malloc(n*4);fread(b,4,n,f);fclose(f);return b;}
int main(int argc,char**argv){
  if(argc!=7){fprintf(stderr,"usage: M K N A B C\n");return 1;}
  int M=atoi(argv[1]),K=atoi(argv[2]),N=atoi(argv[3]);
  void* cuda=dlopen("libcuda.so",RTLD_NOW|RTLD_GLOBAL); if(!cuda){fprintf(stderr,"%s\n",dlerror());return 1;}
  void* blas=dlopen("libcublas.so",RTLD_NOW|RTLD_GLOBAL); if(!blas){fprintf(stderr,"%s\n",dlerror());return 1;}
  CUresult(*cuInit)(unsigned)=must(cuda,"cuInit");
  CUresult(*cuDeviceGet)(CUdevice*,int)=must(cuda,"cuDeviceGet");
  CUresult(*cuCtxCreate)(CUcontext*,unsigned,CUdevice)=must(cuda,"cuCtxCreate_v2");
  CUresult(*cuMemAlloc)(CUdeviceptr*,size_t)=must(cuda,"cuMemAlloc_v2");
  CUresult(*cuMemcpyHtoD)(CUdeviceptr,const void*,size_t)=must(cuda,"cuMemcpyHtoD_v2");
  CUresult(*cuMemcpyDtoH)(void*,CUdeviceptr,size_t)=must(cuda,"cuMemcpyDtoH_v2");
  CUresult(*cuCtxSynchronize)(void)=must(cuda,"cuCtxSynchronize");
  int(*cublasCreate)(cublasHandle_t*)=must(blas,"cublasCreate_v2");
  int(*cublasSgemm)(cublasHandle_t,int,int,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int)=must(blas,"cublasSgemm_v2");
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext c; CK(cuCtxCreate(&c,0,d));
  float* A=rd(argv[4],M*K); float* B=rd(argv[5],K*N); float* C=calloc(M*N,4);
  CUdeviceptr dA,dB,dC; CK(cuMemAlloc(&dA,M*K*4));CK(cuMemAlloc(&dB,K*N*4));CK(cuMemAlloc(&dC,M*N*4));
  CK(cuMemcpyHtoD(dA,A,M*K*4));CK(cuMemcpyHtoD(dB,B,K*N*4));
  cublasHandle_t h; if(cublasCreate(&h)){fprintf(stderr,"create fail\n");return 5;}
  float al=1.f,be=0.f;
  int st=cublasSgemm(h,0,0,N,M,K,&al,(const float*)dB,N,(const float*)dA,K,&be,(float*)dC,N);
  if(st){fprintf(stderr,"sgemm->%d\n",st);return 6;}
  CK(cuCtxSynchronize()); CK(cuMemcpyDtoH(C,dC,M*N*4));
  FILE* o=fopen(argv[6],"wb");fwrite(C,4,M*N,o);fclose(o);
  fprintf(stderr,"OK %dx%d\n",M,N); return 0;
}
