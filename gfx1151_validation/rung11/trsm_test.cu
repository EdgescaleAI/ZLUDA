// Rung 11 verification: cuBLAS trsm (triangular solve) through ZLUDA -> rocBLAS.
// Solves op(A) X = alpha B (side=LEFT) for several (uplo,trans,diag) configs and
// a double-precision case, checking the recovered X against a known X (column-major).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)
#define BK(x) do{ cublasStatus_t s=(x); if(s){printf("CUBLAS ERR %s @%d: status=%d\n",#x,__LINE__,(int)s); exit(3);} }while(0)

// column-major index
static inline int IDX(int i,int j,int ld){ return i + j*ld; }

template<typename T>
static double run_case(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans,
                       cublasDiagType_t diag, int m, int n, const char* tag, bool dbl){
  std::vector<T> A(m*m, (T)0), X(m*n), B(m*m? m*n:0), Bcheck(m*n);
  // Build a well-conditioned triangular A: diagonal ~ 2..3, off-tri small.
  for(int j=0;j<m;j++) for(int i=0;i<m;i++){
    bool keep = (uplo==CUBLAS_FILL_MODE_LOWER) ? (i>=j) : (i<=j);
    if(!keep) continue;
    T v = (i==j) ? (T)(2.0 + 0.1*i) : (T)(0.15*((i*7+j*3)%5 - 2));
    if(diag==CUBLAS_DIAG_UNIT && i==j) v=(T)1.0; // unit diag: cublas ignores stored diag
    A[IDX(i,j,m)] = v;
  }
  // Known X
  for(int j=0;j<n;j++) for(int i=0;i<m;i++) X[IDX(i,j,m)] = (T)(0.5 + 0.3*((i*5+j*2)%7) - 0.9*(j%3));
  // Effective A used by solver: if UNIT diag, treat diagonal as 1 for the host product.
  auto Aeff=[&](int i,int j)->T{
    if(diag==CUBLAS_DIAG_UNIT && i==j) return (T)1.0;
    return A[IDX(i,j,m)];
  };
  // op(A): trans?  We test trans=N only here for the host product simplicity, plus a T case
  // computed by transposing the operator.
  std::vector<T> Bm(m*n,(T)0);
  for(int j=0;j<n;j++) for(int i=0;i<m;i++){
    T acc=(T)0;
    for(int k=0;k<m;k++){
      T a = (trans==CUBLAS_OP_N) ? Aeff(i,k) : Aeff(k,i);
      acc += a * X[IDX(k,j,m)];
    }
    Bm[IDX(i,j,m)] = acc; // op(A) X = B
  }
  T *dA,*dB; CK(cudaMalloc(&dA,sizeof(T)*m*m)); CK(cudaMalloc(&dB,sizeof(T)*m*n));
  CK(cudaMemcpy(dA,A.data(),sizeof(T)*m*m,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,Bm.data(),sizeof(T)*m*n,cudaMemcpyHostToDevice));
  T alpha=(T)1.0;
  if(dbl) BK(cublasDtrsm_v2(h,CUBLAS_SIDE_LEFT,uplo,trans,diag,m,n,(const double*)&alpha,(const double*)dA,m,(double*)dB,m));
  else    BK(cublasStrsm_v2(h,CUBLAS_SIDE_LEFT,uplo,trans,diag,m,n,(const float*)&alpha,(const float*)dA,m,(float*)dB,m));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(Bcheck.data(),dB,sizeof(T)*m*n,cudaMemcpyDeviceToHost));
  double maxabs=0;
  for(int t=0;t<m*n;t++){ double d=fabs((double)Bcheck[t]-(double)X[t]); if(d>maxabs)maxabs=d; }
  printf("CASE %-22s m=%d n=%d -> max_abs_err=%.3e\n", tag, m, n, maxabs);
  cudaFree(dA); cudaFree(dB);
  return maxabs;
}

int main(){
  cublasHandle_t h; BK(cublasCreate_v2(&h));
  double tol_s=1e-3, tol_d=1e-9, worst=0; int bad=0;
  worst=fmax(worst, run_case<float>(h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT, 64, 8,"S lower N nonunit",false)); bad += (worst>tol_s);
  {double e=run_case<float>(h,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT, 64, 8,"S upper N nonunit",false); worst=fmax(worst,e); bad += (e>tol_s);}
  {double e=run_case<float>(h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT, 48,16,"S lower T nonunit",false); worst=fmax(worst,e); bad += (e>tol_s);}
  {double e=run_case<float>(h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_UNIT,     32, 4,"S lower N unit",   false); worst=fmax(worst,e); bad += (e>tol_s);}
  {double e=run_case<double>(h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,64, 8,"D lower N nonunit",true ); worst=fmax(worst,e); bad += (e>tol_d);}
  cublasDestroy_v2(h);
  printf("RUNG11_TRSM worst_err=%.3e bad_cases=%d -> %s\n", worst, bad, bad? "FAIL":"PASS");
  return bad?1:0;
}
