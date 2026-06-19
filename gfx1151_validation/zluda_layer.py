#!/usr/bin/env python3
"""Rung-2.5 stepping stone: a full Qwen3-style decoder layer through ZLUDA on gfx1151.

Chains ZLUDA cuBLAS->rocBLAS GEMMs with nvrtc->PTX->ZLUDA kernels (RMSNorm, per-head
QK-norm, RoPE, causal GQA attention, SwiGLU, residual), keeping intermediates on the
device between steps. Graded against an independent pure-Python fp64 reference (the
oracle), tight fp32 tolerance. Proves the op set COMPOSES correctly in one context —
the rung between single-op (rung 1, 20/20) and whole-model (rung 3).

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib python3 zluda_layer.py
"""
import ctypes as C, os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_diff import Cuda, compile_ptx, CUDA, CUBLAS, _ck

# ---- dims (small but model-realistic ratios; GQA 4:1, hd=H/nq) ----
S, H = 6, 64
NQ, NKV, HD = 8, 2, 8         # nq*hd = 64 = H ; gqa 4:1
I = 128
THETA, EPS = 10000.0, 1e-6
import sys as _s; random.seed(int(_s.argv[1]) if len(_s.argv)>1 else 20260618)

def rnd(n): return [random.uniform(-0.5, 0.5) for _ in range(n)]
def mat(r, c): return [rnd(c) for _ in range(r)]

# weights (row-major, [in, out] so x@W works): Wq/Wk/Wv/Wo/Wg/Wu/Wd
Wq = mat(H, NQ*HD); Wk = mat(H, NKV*HD); Wv = mat(H, NKV*HD)
Wo = mat(NQ*HD, H)
Wg = mat(H, I); Wu = mat(H, I); Wd = mat(I, H)
gN1 = rnd(H); gN2 = rnd(H)            # rmsnorm weights
gQ = rnd(HD); gK = rnd(HD)            # qk-norm weights
X = mat(S, H)

# ================= pure-python fp64 reference (oracle) =================
def mm(A, B):  # A:(m,k) B:(k,n)
    m, k, n = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][t]*B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
def rmsnorm(x, g):
    out=[]
    for row in x:
        ss=sum(v*v for v in row)/len(row)
        inv=1.0/math.sqrt(ss+EPS)
        out.append([row[j]*inv*g[j] for j in range(len(row))])
    return out
def add(a,b): return [[a[i][j]+b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
def rope_heads(mat_, nh):  # apply rope within each head block of width HD
    out=[row[:] for row in mat_]
    half=HD//2
    for h in range(nh):
        base=h*HD
        for p in range(len(mat_)):
            for i in range(half):
                ang=p*(THETA**(-2.0*i/HD)); c=math.cos(ang); s=math.sin(ang)
                x1=mat_[p][base+i]; x2=mat_[p][base+half+i]
                out[p][base+i]=x1*c-x2*s; out[p][base+half+i]=x1*s+x2*c
    return out
def qknorm(mat_, nh, g):  # per-head rmsnorm over HD
    out=[row[:] for row in mat_]
    for h in range(nh):
        base=h*HD
        for p in range(len(mat_)):
            ss=sum(mat_[p][base+t]**2 for t in range(HD))/HD
            inv=1.0/math.sqrt(ss+EPS)
            for t in range(HD): out[p][base+t]=mat_[p][base+t]*inv*g[t]
    return out
def gqa_attn(Q,K,V):  # Q:(S,NQ*HD) K/V:(S,NKV*HD) causal
    scale=1.0/math.sqrt(HD); out=[[0.0]*(NQ*HD) for _ in range(S)]
    for h in range(NQ):
        kvh=h//(NQ//NKV); qb=h*HD; kb=kvh*HD
        for i in range(S):
            sc=[];
            for j in range(S):
                if j<=i: sc.append(sum(Q[i][qb+t]*K[j][kb+t] for t in range(HD))*scale)
                else: sc.append(-1e30)
            m=max(sc); e=[math.exp(v-m) if v>-1e29 else 0.0 for v in sc]; ssum=sum(e)
            for t in range(HD):
                out[i][qb+t]=sum((e[j]/ssum)*V[j][kb+t] for j in range(S))
    return out
def swiglu(g,u): return [[ (g[i][j]/(1.0+math.exp(-g[i][j])))*u[i][j] for j in range(len(g[0]))] for i in range(len(g))]

def reference():
    xn=rmsnorm(X,gN1)
    Q=mm(xn,Wq); K=mm(xn,Wk); V=mm(xn,Wv)
    Q=qknorm(Q,NQ,gQ); K=qknorm(K,NKV,gK)
    Q=rope_heads(Q,NQ); K=rope_heads(K,NKV)
    ao=gqa_attn(Q,K,V)
    o=mm(ao,Wo)
    x1=add(X,o)
    xn2=rmsnorm(x1,gN2)
    g=mm(xn2,Wg); u=mm(xn2,Wu)
    sw=swiglu(g,u)
    d=mm(sw,Wd)
    return add(x1,d)

# ================= ZLUDA path (cuBLAS GEMM + PTX kernels, device-resident) =================
KSRC=r'''
extern "C" __global__ void rmsnorm(const float* x,const float* g,float* o,int rows,int cols,float eps){
  int r=blockIdx.x; if(r>=rows) return; float ss=0; for(int j=0;j<cols;j++){float v=x[r*cols+j]; ss+=v*v;}
  float inv=rsqrtf(ss/cols+eps); for(int j=0;j<cols;j++) o[r*cols+j]=x[r*cols+j]*inv*g[j]; }
extern "C" __global__ void qknorm(const float* x,const float* g,float* o,int rows,int width,int nh,int hd,float eps){
  int r=blockIdx.x; if(r>=rows) return;
  for(int h=0;h<nh;h++){int b=h*hd; float ss=0; for(int t=0;t<hd;t++){float v=x[r*width+b+t]; ss+=v*v;}
    float inv=rsqrtf(ss/hd+eps); for(int t=0;t<hd;t++) o[r*width+b+t]=x[r*width+b+t]*inv*g[t]; } }
extern "C" __global__ void rope(const float* x,float* o,int rows,int width,int nh,int hd,float theta){
  int r=blockIdx.x; if(r>=rows) return; int half=hd/2; for(int k=0;k<width;k++) o[r*width+k]=x[r*width+k];
  for(int h=0;h<nh;h++){int b=h*hd; for(int i=0;i<half;i++){float ang=r*powf(theta,-2.f*i/hd);float c=cosf(ang),s=sinf(ang);
    float x1=x[r*width+b+i],x2=x[r*width+b+half+i]; o[r*width+b+i]=x1*c-x2*s; o[r*width+b+half+i]=x1*s+x2*c; } } }
extern "C" __global__ void gqa(const float* Q,const float* K,const float* V,float* O,int S,int nq,int nkv,int hd){
  float scale=rsqrtf((float)hd);
  for(int h=0;h<nq;h++){int kvh=h/(nq/nkv); int qb=h*hd; int kb=kvh*hd; int qc=nq*hd; int kc=nkv*hd;
    for(int i=0;i<S;i++){ float sc[64]; float m=-1e30f;
      for(int j=0;j<S;j++){ if(j<=i){float d=0;for(int t=0;t<hd;t++)d+=Q[i*qc+qb+t]*K[j*kc+kb+t]; sc[j]=d*scale; if(sc[j]>m)m=sc[j];} else sc[j]=-1e30f; }
      float ss=0; for(int j=0;j<S;j++){float e=(sc[j]<=-1e29f)?0.f:expf(sc[j]-m); sc[j]=e; ss+=e;}
      for(int t=0;t<hd;t++){float acc=0; for(int j=0;j<S;j++) acc+=(sc[j]/ss)*V[j*kc+kb+t]; O[i*qc+qb+t]=acc;} } } }
extern "C" __global__ void swiglu(const float* g,const float* u,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float x=g[i]; o[i]=(x/(1.f+expf(-x)))*u[i];} }
extern "C" __global__ void addk(const float* a,const float* b,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }
'''

def flat(m): return [v for row in m for v in row]

class Z:
    def __init__(self):
        self.cu=Cuda(); self.m=self.cu.module(compile_ptx(KSRC))
        self.h=C.c_void_p(); _ck(CUBLAS.cublasCreate_v2(C.byref(self.h)),"cublasCreate")
        CUBLAS.cublasSgemm_v2.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int]
    def fn(self,name): return self.cu.func(self.m,name)
    def dev(self,floats): return self.cu.to_dev(floats)
    def alloc(self,n): return self.cu.alloc(n)
    def gemm(self,dA,dB,M,K,N):  # row-major C=A*B, returns device ptr
        dC=self.alloc(M*N); a=C.c_float(1.0); b=C.c_float(0.0)
        pB=C.c_void_p(dB.value); pA=C.c_void_p(dA.value); pC=C.c_void_p(dC.value)
        _ck(CUBLAS.cublasSgemm_v2(self.h,0,0,N,M,K,C.byref(a),pB,N,pA,K,C.byref(b),pC,N),"sgemm")
        _ck(CUDA.cuCtxSynchronize(),"sync"); return dC
    def launch(self,name,grid,block,params): self.cu.launch(self.fn(name),grid,block,params)
    def get(self,dptr,n): return self.cu.from_dev(dptr,n)

def run_zluda():
    z=Z()
    dX=z.dev(flat(X))
    dWq=z.dev(flat(Wq)); dWk=z.dev(flat(Wk)); dWv=z.dev(flat(Wv)); dWo=z.dev(flat(Wo))
    dWg=z.dev(flat(Wg)); dWu=z.dev(flat(Wu)); dWd=z.dev(flat(Wd))
    dgN1=z.dev(gN1); dgN2=z.dev(gN2); dgQ=z.dev(gQ); dgK=z.dev(gK)
    # rmsnorm1
    dxn=z.alloc(S*H); z.launch("rmsnorm",(S,1,1),(1,1,1),[C.c_void_p(dX.value),C.c_void_p(dgN1.value),C.c_void_p(dxn.value),C.c_int(S),C.c_int(H),C.c_float(EPS)])
    # qkv proj
    dQ=z.gemm(dxn,dWq,S,H,NQ*HD); dK=z.gemm(dxn,dWk,S,H,NKV*HD); dV=z.gemm(dxn,dWv,S,H,NKV*HD)
    # qk-norm
    dQn=z.alloc(S*NQ*HD); z.launch("qknorm",(S,1,1),(1,1,1),[C.c_void_p(dQ.value),C.c_void_p(dgQ.value),C.c_void_p(dQn.value),C.c_int(S),C.c_int(NQ*HD),C.c_int(NQ),C.c_int(HD),C.c_float(EPS)])
    dKn=z.alloc(S*NKV*HD); z.launch("qknorm",(S,1,1),(1,1,1),[C.c_void_p(dK.value),C.c_void_p(dgK.value),C.c_void_p(dKn.value),C.c_int(S),C.c_int(NKV*HD),C.c_int(NKV),C.c_int(HD),C.c_float(EPS)])
    # rope
    dQr=z.alloc(S*NQ*HD); z.launch("rope",(S,1,1),(1,1,1),[C.c_void_p(dQn.value),C.c_void_p(dQr.value),C.c_int(S),C.c_int(NQ*HD),C.c_int(NQ),C.c_int(HD),C.c_float(THETA)])
    dKr=z.alloc(S*NKV*HD); z.launch("rope",(S,1,1),(1,1,1),[C.c_void_p(dKn.value),C.c_void_p(dKr.value),C.c_int(S),C.c_int(NKV*HD),C.c_int(NKV),C.c_int(HD),C.c_float(THETA)])
    # attention
    dao=z.alloc(S*NQ*HD); z.launch("gqa",(1,1,1),(1,1,1),[C.c_void_p(dQr.value),C.c_void_p(dKr.value),C.c_void_p(dV.value),C.c_void_p(dao.value),C.c_int(S),C.c_int(NQ),C.c_int(NKV),C.c_int(HD)])
    # o proj + residual
    do=z.gemm(dao,dWo,S,NQ*HD,H)
    dx1=z.alloc(S*H); z.launch("addk",((S*H+63)//64,1,1),(64,1,1),[C.c_void_p(dX.value),C.c_void_p(do.value),C.c_void_p(dx1.value),C.c_int(S*H)])
    # rmsnorm2 + mlp
    dxn2=z.alloc(S*H); z.launch("rmsnorm",(S,1,1),(1,1,1),[C.c_void_p(dx1.value),C.c_void_p(dgN2.value),C.c_void_p(dxn2.value),C.c_int(S),C.c_int(H),C.c_float(EPS)])
    dg=z.gemm(dxn2,dWg,S,H,I); du=z.gemm(dxn2,dWu,S,H,I)
    dsw=z.alloc(S*I); z.launch("swiglu",((S*I+63)//64,1,1),(64,1,1),[C.c_void_p(dg.value),C.c_void_p(du.value),C.c_void_p(dsw.value),C.c_int(S*I)])
    dd=z.gemm(dsw,dWd,S,I,H)
    dout=z.alloc(S*H); z.launch("addk",((S*H+63)//64,1,1),(64,1,1),[C.c_void_p(dx1.value),C.c_void_p(dd.value),C.c_void_p(dout.value),C.c_int(S*H)])
    flat_out=z.get(dout,S*H)
    return [flat_out[i*H:(i+1)*H] for i in range(S)]

def main():
    ref=reference(); got=run_zluda()
    mxa=mxr=0.0; nf=0; rtol,atol=1e-4,1e-5
    for i in range(S):
        for j in range(H):
            a=abs(got[i][j]-ref[i][j]); r=a/(abs(ref[i][j])+1e-12)
            mxa=max(mxa,a); mxr=max(mxr,r)
            if a>atol+rtol*abs(ref[i][j]): nf+=1
    print(f"Qwen3-style decoder layer through ZLUDA: max_abs={mxa:.3e} max_rel={mxr:.3e} n_fail={nf}/{S*H}")
    print("VERDICT:", "PASS" if nf==0 else "FAIL")
    return 0 if nf==0 else 1

if __name__=="__main__":
    sys.exit(main())
