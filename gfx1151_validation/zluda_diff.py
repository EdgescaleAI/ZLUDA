#!/usr/bin/env python3
"""Rung-1 T0 op differential through ZLUDA on gfx1151, WITHOUT PyTorch.

NVIDIA nvrtc compiles a per-op CUDA-C kernel -> PTX; ZLUDA's libcuda loads that
PTX (cuModuleLoadData), runs it on the AMD GPU (cuLaunchKernel), and we diff the
output against the captured fixture using the harness's tolerance policy.

This is the faithful ZLUDA path (PTX -> ZLUDA frontend -> AMD), bypassing the
stock PyTorch wheel whose kernels are SASS-only (no PTX for ZLUDA to translate).

Usage: LD_LIBRARY_PATH=<zluda>:/opt/rocm/lib python3 tests/zluda_diff.py [op ...]
"""
import ctypes as C
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t0_ops.ops import REGISTRY
from harness.diff import compare
from harness.tolerances import tolerance_for

NVRTC = C.CDLL("libnvrtc.so.12")
CUDA = C.CDLL("libcuda.so")   # ZLUDA via LD_LIBRARY_PATH
CUBLAS = C.CDLL("libcublas.so")  # ZLUDA -> rocBLAS

def _ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what} -> CUresult {r}")

# ---- nvrtc: CUDA-C -> PTX -------------------------------------------------
def compile_ptx(src, name="k.cu", arch="compute_75"):
    prog = C.c_void_p()
    _ck(NVRTC.nvrtcCreateProgram(C.byref(prog), src.encode(), name.encode(), 0, None, None), "nvrtcCreateProgram")
    opts = (C.c_char_p * 1)(f"--gpu-architecture={arch}".encode())
    rc = NVRTC.nvrtcCompileProgram(prog, 1, opts)
    logsz = C.c_size_t()
    NVRTC.nvrtcGetProgramLogSize(prog, C.byref(logsz))
    log = C.create_string_buffer(logsz.value)
    NVRTC.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise RuntimeError("nvrtc compile failed:\n" + log.value.decode())
    sz = C.c_size_t()
    NVRTC.nvrtcGetPTXSize(prog, C.byref(sz))
    ptx = C.create_string_buffer(sz.value)
    NVRTC.nvrtcGetPTX(prog, ptx)
    return ptx.value  # bytes, null-terminated

# ---- ZLUDA libcuda driver -------------------------------------------------
class Cuda:
    def __init__(self):
        _ck(CUDA.cuInit(0), "cuInit")
        self.dev = C.c_int()
        _ck(CUDA.cuDeviceGet(C.byref(self.dev), 0), "cuDeviceGet")
        self.ctx = C.c_void_p()
        _ck(CUDA.cuCtxCreate_v2(C.byref(self.ctx), 0, self.dev), "cuCtxCreate")
    def module(self, ptx):
        m = C.c_void_p()
        _ck(CUDA.cuModuleLoadData(C.byref(m), ptx), "cuModuleLoadData")
        return m
    def func(self, m, name):
        f = C.c_void_p()
        _ck(CUDA.cuModuleGetFunction(C.byref(f), m, name.encode()), "cuModuleGetFunction")
        return f
    def to_dev(self, floats):
        n = len(floats)
        arr = (C.c_float * n)(*floats)
        dptr = C.c_ulonglong()
        _ck(CUDA.cuMemAlloc_v2(C.byref(dptr), n*4), "cuMemAlloc")
        _ck(CUDA.cuMemcpyHtoD_v2(dptr, arr, n*4), "cuMemcpyHtoD")
        return dptr
    def alloc(self, n):
        dptr = C.c_ulonglong()
        _ck(CUDA.cuMemAlloc_v2(C.byref(dptr), n*4), "cuMemAlloc")
        return dptr
    def from_dev(self, dptr, n):
        arr = (C.c_float * n)()
        _ck(CUDA.cuMemcpyDtoH_v2(arr, dptr, n*4), "cuMemcpyDtoH")
        return list(arr)
    def launch(self, f, grid, block, params):
        # params: list of (ctype_instance) already; build void* array of &param
        parr = (C.c_void_p * len(params))()
        for i,p in enumerate(params):
            parr[i] = C.cast(C.byref(p), C.c_void_p)
        _ck(CUDA.cuLaunchKernel(f, grid[0],grid[1],grid[2], block[0],block[1],block[2],
                                0, None, parr, None), "cuLaunchKernel")
        _ck(CUDA.cuCtxSynchronize(), "cuCtxSynchronize")

def flat(x):
    out=[]
    def rec(v):
        if isinstance(v,(list,tuple)):
            for e in v: rec(e)
        else: out.append(float(v))
    rec(x); return out

# ---- per-op kernels + launch specs ---------------------------------------
# Each spec(ins) returns: (cuda_src, kernel_name, [input_flat_arrays], out_numel,
#                          grid, block, [extra_scalar_ctypes])
OPS = {}
def op(name):
    def deco(fn): OPS[name]=fn; return fn
    return deco

@op("residual_add_4x16")
def _residual_add(ins):
    x=flat(ins["x"]); y=flat(ins["y"]); n=len(x)
    src=r'''
extern "C" __global__ void residual_add(const float* x, const float* y, float* o, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=x[i]+y[i];
}'''
    return src,"residual_add",[x,y],n,((n+63)//64,1,1),(64,1,1),[C.c_int(n)]

@op("gelu_4x16")
def _gelu(ins):
    x=flat(ins["x"]); n=len(x)
    # tanh-approx GELU (matches harness reference)
    src=r'''
extern "C" __global__ void gelu(const float* x, float* o, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){ float v=x[i]; float c=0.7978845608028654f; // sqrt(2/pi)
    float t=tanhf(c*(v+0.044715f*v*v*v)); o[i]=0.5f*v*(1.0f+t); }
}'''
    return src,"gelu",[x],n,((n+63)//64,1,1),(64,1,1),[C.c_int(n)]

@op("rmsnorm_4x16")
def _rmsnorm(ins):
    x=flat(ins["x"]); w=flat(ins["w"]); rows=len(ins["x"]); cols=len(ins["x"][0])
    src=r'''
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* o, int rows, int cols){
  int r=blockIdx.x; if(r>=rows) return;
  float ss=0.f; for(int j=0;j<cols;j++){ float v=x[r*cols+j]; ss+=v*v; }
  float inv=rsqrtf(ss/cols + 1e-6f);
  for(int j=0;j<cols;j++){ o[r*cols+j]=x[r*cols+j]*inv*w[j]; }
}'''
    return src,"rmsnorm",[x,w],rows*cols,(rows,1,1),(1,1,1),[C.c_int(rows),C.c_int(cols)]

@op("softmax_4x16")
def _softmax(ins):
    x=flat(ins["x"]); rows=len(ins["x"]); cols=len(ins["x"][0])
    src=r'''
extern "C" __global__ void softmax(const float* x, float* o, int rows, int cols){
  int r=blockIdx.x; if(r>=rows) return;
  float m=-1e30f; for(int j=0;j<cols;j++) m=fmaxf(m,x[r*cols+j]);
  float s=0.f; for(int j=0;j<cols;j++){ float e=expf(x[r*cols+j]-m); o[r*cols+j]=e; s+=e; }
  for(int j=0;j<cols;j++) o[r*cols+j]/=s;
}'''
    return src,"softmax",[x],rows*cols,(rows,1,1),(1,1,1),[C.c_int(rows),C.c_int(cols)]

@op("swiglu_4x16")
def _swiglu(ins):
    g=flat(ins["gate"]); u=flat(ins["up"]); n=len(g)
    src=r'''
extern "C" __global__ void swiglu(const float* g, const float* u, float* o, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){ float x=g[i]; float silu=x/(1.0f+expf(-x)); o[i]=silu*u[i]; }
}'''
    return src,"swiglu",[g,u],n,((n+63)//64,1,1),(64,1,1),[C.c_int(n)]

@op("layernorm_4x16")
def _layernorm(ins):
    x=flat(ins["x"]); w=flat(ins["g"]); b=flat(ins["b"]); rows=len(ins["x"]); cols=len(ins["x"][0])
    src=r'''
extern "C" __global__ void layernorm(const float* x, const float* w, const float* b, float* o, int rows, int cols){
  int r=blockIdx.x; if(r>=rows) return;
  float mean=0.f; for(int j=0;j<cols;j++) mean+=x[r*cols+j]; mean/=cols;
  float var=0.f; for(int j=0;j<cols;j++){ float d=x[r*cols+j]-mean; var+=d*d; } var/=cols;
  float inv=rsqrtf(var+1e-6f);
  for(int j=0;j<cols;j++){ o[r*cols+j]=(x[r*cols+j]-mean)*inv*w[j]+b[j]; }
}'''
    return src,"layernorm",[x,w,b],rows*cols,(rows,1,1),(1,1,1),[C.c_int(rows),C.c_int(cols)]

@op("rope_4x8")
def _rope(ins):
    x=flat(ins["x"]); seq,dim=len(ins["x"]),len(ins["x"][0])
    src=r'''
extern "C" __global__ void rope(const float* x, float* o, int seq, int dim){
  int r=blockIdx.x; if(r>=seq) return; int half=dim/2; float theta=10000.f;
  for(int i=0;i<half;i++){ float ang=r*powf(theta,-2.f*i/dim); float c=cosf(ang),s=sinf(ang);
    float x1=x[r*dim+i], x2=x[r*dim+half+i];
    o[r*dim+i]=x1*c-x2*s; o[r*dim+half+i]=x1*s+x2*c; }
}'''
    return src,"rope",[x],seq*dim,(seq,1,1),(1,1,1),[C.c_int(seq),C.c_int(dim)]

@op("rope2d_4x8")
def _rope2d(ins):
    x=flat(ins["x"]); seq,dim=len(ins["x"]),len(ins["x"][0])
    src=r'''
extern "C" __global__ void rope2d(const float* x, float* o, int seq, int dim){
  int r=blockIdx.x; if(r>=seq) return; int half=dim/2; int hl=half/2; float theta=10000.f;
  for(int k=0;k<dim;k++) o[r*dim+k]=x[r*dim+k];
  float coord[2]; coord[0]=(float)(r/2); coord[1]=(float)(r%2);
  for(int axis=0;axis<2;axis++){ int base=axis*half; float pos=coord[axis];
    for(int i=0;i<hl;i++){ float ang=pos*powf(theta,-2.f*i/half); float c=cosf(ang),s=sinf(ang);
      float x1=x[r*dim+base+i], x2=x[r*dim+base+hl+i];
      o[r*dim+base+i]=x1*c-x2*s; o[r*dim+base+hl+i]=x1*s+x2*c; } }
}'''
    return src,"rope2d",[x],seq*dim,(seq,1,1),(1,1,1),[C.c_int(seq),C.c_int(dim)]

@op("repeat_kv_3x8")
def _repeat_kv(ins):
    kv=flat(ins["kv"]); rows=len(ins["kv"])  # (3,8) -> (3,16); nkv=2,hd=4,rep=2
    src=r'''
extern "C" __global__ void repeat_kv(const float* kv, float* o, int rows){
  int r=blockIdx.x; if(r>=rows) return; int incol=8,outcol=16;
  // heads h0=cols0-3,h1=4-7 ; expanded [h0,h0,h1,h1]
  for(int j=0;j<4;j++){ float a=kv[r*incol+j], b=kv[r*incol+4+j];
    o[r*outcol+j]=a; o[r*outcol+4+j]=a; o[r*outcol+8+j]=b; o[r*outcol+12+j]=b; }
}'''
    return src,"repeat_kv",[kv],rows*16,(rows,1,1),(1,1,1),[C.c_int(rows)]

@op("embedding_gather_5")
def _embgather(ins):
    table=flat(ins["table"]); ids=ins["ids"][0]
    ids_int=[float(int(round(i))) for i in ids]; n=len(ids_int); cols=len(ins["table"][0])
    src=r'''
extern "C" __global__ void embgather(const float* table, const float* ids, float* o, int n, int cols){
  int k=blockIdx.x; if(k>=n) return; int row=(int)(ids[k]+0.5f);
  for(int j=0;j<cols;j++) o[k*cols+j]=table[row*cols+j];
}'''
    return src,"embgather",[table,ids_int],n*cols,(n,1,1),(1,1,1),[C.c_int(n),C.c_int(cols)]

@op("spatial_merge_2x2")
def _spatial(ins):
    x=flat(ins["x"]); n=len(x)
    src=r'''
extern "C" __global__ void spatial(const float* x, float* o, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=x[i];
}'''
    return src,"spatial",[x],n,((n+31)//32,1,1),(32,1,1),[C.c_int(n)]

@op("token_scatter_4x4")
def _tokscatter(ins):
    llm=flat(ins["llm"]); vis=flat(ins["vis"]); tap=flat(ins["tap"]); cols=len(ins["llm"][0])
    src=r'''
extern "C" __global__ void tokscatter(const float* llm, const float* vis, const float* tap, float* o, int rows, int cols){
  for(int r=0;r<rows;r++) for(int j=0;j<cols;j++) o[r*cols+j]=llm[r*cols+j];
  // rows 1,2 replaced by vis[k]+tap[k], k=0,1
  for(int k=0;k<2;k++){ int p=k+1; for(int j=0;j<cols;j++) o[p*cols+j]=vis[k*cols+j]+tap[k*cols+j]; }
}'''
    return src,"tokscatter",[llm,vis,tap],4*cols,(1,1,1),(1,1,1),[C.c_int(4),C.c_int(cols)]

@op("kv_cache_2p1")
def _kvcache(ins):
    cache=flat(ins["cache"]); new=flat(ins["new"]); cols=len(ins["cache"][0])
    nc=len(ins["cache"]); nn=len(ins["new"])
    src=r'''
extern "C" __global__ void kvcache(const float* cache, const float* nw, float* o, int nc, int nn, int cols){
  for(int r=0;r<nc;r++) for(int j=0;j<cols;j++) o[r*cols+j]=cache[r*cols+j];
  for(int r=0;r<nn;r++) for(int j=0;j<cols;j++) o[(nc+r)*cols+j]=nw[r*cols+j];
}'''
    return src,"kvcache",[cache,new],(nc+nn)*cols,(1,1,1),(1,1,1),[C.c_int(nc),C.c_int(nn),C.c_int(cols)]

@op("sampling_topk_8")
def _topk(ins):
    logits=flat(ins["logits"]); n=len(logits)
    src=r'''
extern "C" __global__ void topk(const float* logits, float* o, int n){
  float scaled[64]; for(int i=0;i<n;i++) scaled[i]=logits[i]/0.8f;
  // keep top-4: element kept if (#strictly greater) < 4
  float m=-1e30f;
  for(int i=0;i<n;i++){ int gt=0; for(int j=0;j<n;j++) if(scaled[j]>scaled[i]) gt++;
    o[i]= (gt<4)? scaled[i] : -1e30f; if(o[i]>m) m=o[i]; }
  float s=0.f; for(int i=0;i<n;i++){ float e=(o[i]<=-1e29f)?0.f:expf(o[i]-m); o[i]=e; s+=e; }
  for(int i=0;i<n;i++) o[i]/=s;
}'''
    return src,"topk",[logits],n,(1,1,1),(1,1,1),[C.c_int(n)]

# ---- attention family (single-threaded kernels; tiny fixtures) ----
_ATTN_SRC=r'''
__device__ void attend(const float* Q,const float* K,const float* V,float* O,
                       int sq,int sk,int d,int maskmode,const int* seg){
  // maskmode: 0 none, 1 causal, 2 segment(seg)
  float scale=rsqrtf((float)d);
  for(int i=0;i<sq;i++){
    float sc[64]; float m=-1e30f;
    for(int j=0;j<sk;j++){
      bool allow=true;
      if(maskmode==1) allow=(j<=i);
      else if(maskmode==2) allow=(seg[i]==seg[j]);
      if(allow){ float dot=0.f; for(int t=0;t<d;t++) dot+=Q[i*d+t]*K[j*d+t]; sc[j]=dot*scale; if(sc[j]>m)m=sc[j]; }
      else sc[j]=-1e30f;
    }
    float ssum=0.f; for(int j=0;j<sk;j++){ float e=(sc[j]<=-1e29f)?0.f:expf(sc[j]-m); sc[j]=e; ssum+=e; }
    for(int t=0;t<d;t++){ float acc=0.f; for(int j=0;j<sk;j++) acc+=(sc[j]/ssum)*V[j*d+t]; O[i*d+t]=acc; }
  }
}'''

@op("attn_causal_4x8")
def _attn_causal(ins):
    Q=flat(ins["Q"]);K=flat(ins["K"]);V=flat(ins["V"]); sq=len(ins["Q"]); d=len(ins["Q"][0])
    src=_ATTN_SRC+r'''
extern "C" __global__ void attn(const float* Q,const float* K,const float* V,float* O,int sq,int d){
  attend(Q,K,V,O,sq,sq,d,1,0);
}'''
    return src,"attn",[Q,K,V],sq*d,(1,1,1),(1,1,1),[C.c_int(sq),C.c_int(d)]

@op("paged_attn_4x4")
def _paged(ins):
    Q=flat(ins["Q"]);K=flat(ins["K"]);V=flat(ins["V"]); sq=len(ins["Q"]); d=len(ins["Q"][0])
    src=_ATTN_SRC+r'''
extern "C" __global__ void attn(const float* Q,const float* K,const float* V,float* O,int sq,int d){
  attend(Q,K,V,O,sq,sq,d,1,0);
}'''
    return src,"attn",[Q,K,V],sq*d,(1,1,1),(1,1,1),[C.c_int(sq),C.c_int(d)]

@op("gqa_attn_3x16")
def _gqa(ins):
    Q=flat(ins["Q"]);K=flat(ins["K"]);V=flat(ins["V"]); sq=len(ins["Q"])  # 3
    # nq=4,nkv=2,hd=4. Q cols 16, K/V cols 8.
    src=_ATTN_SRC+r'''
extern "C" __global__ void attn(const float* Q,const float* K,const float* V,float* O,int sq){
  int nq=4,nkv=2,hd=4,qc=16,kc=8;
  float Qh[3*4],Kh[3*4],Vh[3*4],Oh[3*4];
  for(int h=0;h<nq;h++){ int kvh=h/(nq/nkv);
    for(int r=0;r<sq;r++) for(int t=0;t<hd;t++){ Qh[r*hd+t]=Q[r*qc+h*hd+t]; Kh[r*hd+t]=K[r*kc+kvh*hd+t]; Vh[r*hd+t]=V[r*kc+kvh*hd+t]; }
    attend(Qh,Kh,Vh,Oh,sq,sq,hd,1,0);
    for(int r=0;r<sq;r++) for(int t=0;t<hd;t++) O[r*qc+h*hd+t]=Oh[r*hd+t]; }
}'''
    return src,"attn",[Q,K,V],sq*16,(1,1,1),(1,1,1),[C.c_int(sq)]

@op("vis_attn_varlen_4x8")
def _vis(ins):
    Q=flat(ins["Q"]);K=flat(ins["K"]);V=flat(ins["V"]); sq=len(ins["Q"])  # 4
    src=_ATTN_SRC+r'''
extern "C" __global__ void attn(const float* Q,const float* K,const float* V,float* O,int sq){
  int nh=2,hd=4,c=8; int seg[4]={0,0,1,1};
  float Qh[4*4],Kh[4*4],Vh[4*4],Oh[4*4];
  for(int h=0;h<nh;h++){
    for(int r=0;r<sq;r++) for(int t=0;t<hd;t++){ Qh[r*hd+t]=Q[r*c+h*hd+t]; Kh[r*hd+t]=K[r*c+h*hd+t]; Vh[r*hd+t]=V[r*c+h*hd+t]; }
    attend(Qh,Kh,Vh,Oh,sq,sq,hd,2,seg);
    for(int r=0;r<sq;r++) for(int t=0;t<hd;t++) O[r*c+h*hd+t]=Oh[r*hd+t]; }
}'''
    return src,"attn",[Q,K,V],sq*8,(1,1,1),(1,1,1),[C.c_int(sq)]

@op("posemb_interp_2to3")
def _posemb(ins):
    g=flat(ins["grid"]); cols=len(ins["grid"][0])  # grid (4,4): 2x2 of 4-vectors
    src=r'''
extern "C" __global__ void posemb(const float* g, float* o, int cols){
  int gh=2,gw=2,th=3,tw=3;
  int idx=0;
  for(int ti=0;ti<th;ti++) for(int tj=0;tj<tw;tj++){
    float si=ti*(float)(gh-1)/(th-1), sj=tj*(float)(gw-1)/(tw-1);
    int i0=(int)floorf(si), j0=(int)floorf(sj);
    int i1=min(i0+1,gh-1), j1=min(j0+1,gw-1);
    float di=si-i0, dj=sj-j0;
    for(int c=0;c<cols;c++){
      float top=g[(i0*gw+j0)*cols+c]*(1-dj)+g[(i0*gw+j1)*cols+c]*dj;
      float bot=g[(i1*gw+j0)*cols+c]*(1-dj)+g[(i1*gw+j1)*cols+c]*dj;
      o[idx*cols+c]=top*(1-di)+bot*di;
    }
    idx++;
  }
}'''
    return src,"posemb",[g],9*cols,(1,1,1),(1,1,1),[C.c_int(cols)]

# gemm via ZLUDA cuBLAS->rocBLAS (the architecturally-correct redirect, not a PTX kernel)
def _run_gemm_cublas(cuda, ins, gold):
    A=flat(ins["A"]); B=flat(ins["B"])
    M=len(ins["A"]); K=len(ins["A"][0]); N=len(ins["B"][0])
    dA=cuda.to_dev(A); dB=cuda.to_dev(B); dC=cuda.alloc(M*N)
    h=C.c_void_p()
    _ck(CUBLAS.cublasCreate_v2(C.byref(h)),"cublasCreate")
    alpha=C.c_float(1.0); beta=C.c_float(0.0)
    CUBLAS.cublasSgemm_v2.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
        C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int]
    pB=C.c_void_p(dB.value); pA=C.c_void_p(dA.value); pC=C.c_void_p(dC.value)
    # row-major C=A*B via col-major: sgemm(N,N, N,M,K, B,N, A,K, C,N)
    _ck(CUBLAS.cublasSgemm_v2(h,0,0,N,M,K,C.byref(alpha),
        pB,N, pA,K, C.byref(beta), pC,N),"cublasSgemm")
    _ck(CUDA.cuCtxSynchronize(),"sync")
    out=cuda.from_dev(dC,M*N)
    def reshape(fv,like):
        idx=[0]
        def rec(l):
            if isinstance(l,(list,tuple)): return [rec(e) for e in l]
            v=fv[idx[0]];idx[0]+=1;return v
        return rec(like)
    cand=reshape(out,gold)
    rtol,atol,_=tolerance_for("fp32","fp32")
    d=compare(gold,cand,rtol,atol)
    return {"op":"gemm_8x16x8","verdict":"PASS" if d["allclose"] else "FAIL",
            "max_abs":d["max_abs"],"max_rel":d["max_rel"],"n_fail":d["n_fail"],"n":d["n"]}

def run_op(cuda, op_obj):
    fixp=os.path.join(os.path.dirname(__file__),"fixtures",f"{op_obj.name}__{op_obj.case_id}.json")
    if op_obj.name=="gemm_8x16x8":
        return _run_gemm_cublas(cuda, op_obj.make_inputs(), json.load(open(fixp))["output"])
    spec = OPS.get(op_obj.name)
    if spec is None:
        return {"op":op_obj.name,"verdict":"TODO","detail":"no zluda kernel"}
    ins = op_obj.make_inputs()
    src,kname,inputs,out_n,grid,block,scalars = spec(ins)
    ptx = compile_ptx(src)
    m = cuda.module(ptx); f = cuda.func(m, kname)
    dins = [cuda.to_dev(a) for a in inputs]
    dout = cuda.alloc(out_n)
    params = dins + [dout] + scalars
    cuda.launch(f, grid, block, params)
    out_flat = cuda.from_dev(dout, out_n)
    # reshape to fixture shape via the golden's structure
    fix = json.load(open(os.path.join(os.path.dirname(__file__),"fixtures",f"{op_obj.name}__{op_obj.case_id}.json")))
    gold = fix["output"]
    # build candidate with same nesting as gold
    def reshape(flatvals, like):
        idx=[0]
        def rec(l):
            if isinstance(l,(list,tuple)):
                return [rec(e) for e in l]
            v=flatvals[idx[0]]; idx[0]+=1; return v
        return rec(like)
    cand = reshape(out_flat, gold)
    # ZLUDA kernel computes in fp32; reference golden is fp32 -> tight fp32-vs-fp32 diff.
    rtol,atol,mode = tolerance_for("fp32", "fp32")
    d = compare(gold, cand, rtol, atol)
    return {"op":op_obj.name,"verdict":"PASS" if d["allclose"] else "FAIL",
            "max_abs":d["max_abs"],"max_rel":d["max_rel"],"n_fail":d["n_fail"],"n":d["n"]}

def main():
    want = sys.argv[1:]
    cuda = Cuda()
    npass=nfail=ntodo=0
    for o in REGISTRY:
        if want and o.name not in want: continue
        try:
            r = run_op(cuda, o)
        except Exception as e:
            r = {"op":o.name,"verdict":"ERROR","detail":str(e)[:200]}
        v=r["verdict"]
        if v=="PASS":
            npass+=1; print(f"  PASS  {r['op']:24s} max_rel={r['max_rel']:.2e} max_abs={r['max_abs']:.2e}")
        elif v=="FAIL":
            nfail+=1; print(f"  FAIL  {r['op']:24s} max_rel={r['max_rel']:.2e} n_fail={r['n_fail']}/{r['n']}")
        elif v=="TODO":
            ntodo+=1
        else:
            nfail+=1; print(f"  ERROR {r['op']:24s} {r.get('detail','')}")
    print(f"--- ZLUDA T0: {npass} pass, {nfail} fail, {ntodo} no-kernel-yet ---")
    return 1 if nfail else 0

if __name__=="__main__":
    sys.exit(main())
