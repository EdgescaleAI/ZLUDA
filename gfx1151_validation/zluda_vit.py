#!/usr/bin/env python3
"""Rung-6.5: a composed Qwen3-VL VISION-tower (ViT) block + patch-merger through ZLUDA on gfx1151.

The whole-night text ladder (rungs 0->6) climbed the LLM half of the coverage target
(Cosmos-Reason2-8B = Qwen3-VL). This is the VISION half — the analog of rung 2.5
(zluda_layer.py composed one decoder layer) but for the ViT. It exercises a genuinely
different op mix from the text decoder:
  * LayerNorm (mean+variance+bias)  -- NOT RMSNorm
  * GELU (tanh approximation)        -- NOT SwiGLU
  * FULL bidirectional MHA           -- NOT causal GQA
  * 2D-RoPE (vision)                 -- row/col positional, NOT 1D
  * spatial 2x2 patch merge          -- merger reshape/gather

Chains ZLUDA cuBLAS->rocBLAS GEMMs with nvrtc->PTX->ZLUDA elementwise/attention kernels,
device-resident between steps, graded against an independent pure-Python fp64 oracle at
tight fp32 tolerance. Both paths implement the SAME spec, so the diff tests whether the
op set COMPOSES correctly through ZLUDA — not an interpretation of HF internals.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 zluda_vit.py [seed]
"""
import ctypes as C, os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_diff import Cuda, compile_ptx, CUDA, CUBLAS, _ck

# ---- vision dims (small, but Qwen3-VL-shaped ratios; grid GH x GW patches) ----
GH, GW = 4, 4
S = GH * GW                 # 16 patches
HV = 48                     # vision hidden
NH, HD = 4, 12              # heads, head_dim (NH*HD == HV); HD%4==0 for 2D-rope split
IV = 96                     # vision MLP intermediate
MS = 2                      # spatial merge size (2x2)
SM = (GH // MS) * (GW // MS) # merged tokens = 4
MDIM = MS * MS * HV         # merged dim = 192
IM, OUT = 128, 64           # merger MLP intermediate, output dim
EPS = 1e-6
THETA = 10000.0
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 20260619)

def rnd(n): return [random.uniform(-0.5, 0.5) for _ in range(n)]
def mat(r, c): return [rnd(c) for _ in range(r)]

# ---- weights (row-major [in,out] so x@W) ----
Wq = mat(HV, NH*HD); Wk = mat(HV, NH*HD); Wv = mat(HV, NH*HD); Wo = mat(NH*HD, HV)
Wfc1 = mat(HV, IV); Wfc2 = mat(IV, HV)
ln1_g = rnd(HV); ln1_b = rnd(HV); ln2_g = rnd(HV); ln2_b = rnd(HV)   # block LayerNorms
mln_g = rnd(MDIM); mln_b = rnd(MDIM)                                  # merger LayerNorm
Wm1 = mat(MDIM, IM); Wm2 = mat(IM, OUT)
X = mat(S, HV)

# ================= pure-python fp64 reference (oracle) =================
def mm(A, B):
    m, k, n = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][t]*B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
def layernorm(x, g, b):
    out = []
    for row in x:
        n = len(row); mean = sum(row)/n
        var = sum((v-mean)**2 for v in row)/n
        inv = 1.0/math.sqrt(var+EPS)
        out.append([(row[j]-mean)*inv*g[j] + b[j] for j in range(n)])
    return out
def add(a, b): return [[a[i][j]+b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
def gelu_tanh(x):
    c = math.sqrt(2.0/math.pi)
    return [[0.5*v*(1.0+math.tanh(c*(v+0.044715*v**3))) for v in row] for row in x]
def rope2d(mat_, nh):
    # head_dim split: first HD/2 rotary pairs use ROW index, next HD/2 use COL index.
    out = [row[:] for row in mat_]
    half = HD // 2          # rotary pairs per head
    qrt = half // 2         # pairs assigned to each spatial axis
    for p in range(len(mat_)):
        r = p // GW; col = p % GW
        for h in range(nh):
            base = h*HD
            for i in range(half):
                pos = r if i < qrt else col
                ang = pos * (THETA ** (-2.0*i/half))
                cs, sn = math.cos(ang), math.sin(ang)
                x1 = mat_[p][base+i]; x2 = mat_[p][base+half+i]
                out[p][base+i] = x1*cs - x2*sn
                out[p][base+half+i] = x1*sn + x2*cs
    return out
def full_attn(Q, K, V):   # full (non-causal) MHA, NH heads
    scale = 1.0/math.sqrt(HD); out = [[0.0]*(NH*HD) for _ in range(S)]
    for h in range(NH):
        b = h*HD
        for i in range(S):
            sc = [sum(Q[i][b+t]*K[j][b+t] for t in range(HD))*scale for j in range(S)]
            m = max(sc); e = [math.exp(v-m) for v in sc]; ssum = sum(e)
            for t in range(HD):
                out[i][b+t] = sum((e[j]/ssum)*V[j][b+t] for j in range(S))
    return out
def spatial_merge(x):
    # group MSxMS spatial neighbours of the GHxGW grid -> (GH/MS*GW/MS, MS*MS*HV)
    out = []
    for br in range(0, GH, MS):
        for bc in range(0, GW, MS):
            row = []
            for dr in range(MS):
                for dc in range(MS):
                    p = (br+dr)*GW + (bc+dc)
                    row.extend(x[p])
            out.append(row)
    return out

def reference():
    xn = layernorm(X, ln1_g, ln1_b)
    Q = mm(xn, Wq); K = mm(xn, Wk); V = mm(xn, Wv)
    Q = rope2d(Q, NH); K = rope2d(K, NH)
    ao = full_attn(Q, K, V)
    o = mm(ao, Wo)
    x1 = add(X, o)
    xn2 = layernorm(x1, ln2_g, ln2_b)
    h = gelu_tanh(mm(xn2, Wfc1))
    mlp = mm(h, Wfc2)
    blk = add(x1, mlp)
    # patch merger
    mg = spatial_merge(blk)
    mgn = layernorm(mg, mln_g, mln_b)
    m1 = gelu_tanh(mm(mgn, Wm1))
    return mm(m1, Wm2)

# ================= ZLUDA path (cuBLAS GEMM + PTX kernels, device-resident) =================
KSRC = r'''
extern "C" __global__ void layernorm(const float* x,const float* g,const float* b,float* o,int rows,int cols,float eps){
  int r=blockIdx.x; if(r>=rows) return;
  float mean=0; for(int j=0;j<cols;j++) mean+=x[r*cols+j]; mean/=cols;
  float var=0; for(int j=0;j<cols;j++){float d=x[r*cols+j]-mean; var+=d*d;} var/=cols;
  float inv=rsqrtf(var+eps);
  for(int j=0;j<cols;j++) o[r*cols+j]=(x[r*cols+j]-mean)*inv*g[j]+b[j]; }
extern "C" __global__ void rope2d(const float* x,float* o,int rows,int width,int nh,int hd,int gw,float theta){
  int p=blockIdx.x; if(p>=rows) return; int half=hd/2; int qrt=half/2;
  for(int k=0;k<width;k++) o[p*width+k]=x[p*width+k];
  int rr=p/gw; int cc=p%gw;
  for(int h=0;h<nh;h++){int base=h*hd;
    for(int i=0;i<half;i++){ float pos=(i<qrt)?(float)rr:(float)cc;
      float ang=pos*powf(theta,-2.f*i/half); float cs=cosf(ang),sn=sinf(ang);
      float x1=x[p*width+base+i],x2=x[p*width+base+half+i];
      o[p*width+base+i]=x1*cs-x2*sn; o[p*width+base+half+i]=x1*sn+x2*cs; } } }
extern "C" __global__ void mha(const float* Q,const float* K,const float* V,float* O,int S,int nh,int hd){
  float scale=rsqrtf((float)hd);
  for(int h=0;h<nh;h++){int b=h*hd; int c=nh*hd;
    for(int i=0;i<S;i++){ float sc[128]; float m=-1e30f;
      for(int j=0;j<S;j++){ float d=0; for(int t=0;t<hd;t++) d+=Q[i*c+b+t]*K[j*c+b+t]; sc[j]=d*scale; if(sc[j]>m)m=sc[j]; }
      float ss=0; for(int j=0;j<S;j++){float e=expf(sc[j]-m); sc[j]=e; ss+=e;}
      for(int t=0;t<hd;t++){float acc=0; for(int j=0;j<S;j++) acc+=(sc[j]/ss)*V[j*c+b+t]; O[i*c+b+t]=acc;} } } }
extern "C" __global__ void gelu(const float* x,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float v=x[i];
    float c=0.7978845608028654f; o[i]=0.5f*v*(1.f+tanhf(c*(v+0.044715f*v*v*v)));} }
extern "C" __global__ void addk(const float* a,const float* b,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }
extern "C" __global__ void smerge(const float* x,float* o,int gh,int gw,int hv,int ms){
  // o[(br/ms*gwm+bc/ms)] row = concat of ms*ms patches; one thread per merged token
  int gwm=gw/ms; int idx=blockIdx.x; int tot=(gh/ms)*gwm; if(idx>=tot) return;
  int mbr=idx/gwm, mbc=idx%gwm; int br=mbr*ms, bc=mbc*ms; int mdim=ms*ms*hv; int w=0;
  for(int dr=0;dr<ms;dr++) for(int dc=0;dc<ms;dc++){ int p=(br+dr)*gw+(bc+dc);
    for(int t=0;t<hv;t++) o[idx*mdim+(w*hv)+t]=x[p*hv+t]; w++; } }
'''

def flat(m): return [v for row in m for v in row]

class Z:
    def __init__(self):
        self.cu = Cuda(); self.m = self.cu.module(compile_ptx(KSRC))
        self.h = C.c_void_p(); _ck(CUBLAS.cublasCreate_v2(C.byref(self.h)), "cublasCreate")
        CUBLAS.cublasSgemm_v2.argtypes = [C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int]
    def fn(self, name): return self.cu.func(self.m, name)
    def dev(self, floats): return self.cu.to_dev(floats)
    def alloc(self, n): return self.cu.alloc(n)
    def gemm(self, dA, dB, M, K, N):
        dC = self.alloc(M*N); a = C.c_float(1.0); b = C.c_float(0.0)
        pB = C.c_void_p(dB.value); pA = C.c_void_p(dA.value); pC = C.c_void_p(dC.value)
        _ck(CUBLAS.cublasSgemm_v2(self.h,0,0,N,M,K,C.byref(a),pB,N,pA,K,C.byref(b),pC,N), "sgemm")
        _ck(CUDA.cuCtxSynchronize(), "sync"); return dC
    def launch(self, name, grid, block, params): self.cu.launch(self.fn(name), grid, block, params)
    def get(self, dptr, n): return self.cu.from_dev(dptr, n)

def vp(p): return C.c_void_p(p.value)

def run_zluda():
    z = Z()
    dX = z.dev(flat(X))
    dWq=z.dev(flat(Wq)); dWk=z.dev(flat(Wk)); dWv=z.dev(flat(Wv)); dWo=z.dev(flat(Wo))
    dWfc1=z.dev(flat(Wfc1)); dWfc2=z.dev(flat(Wfc2))
    dl1g=z.dev(ln1_g); dl1b=z.dev(ln1_b); dl2g=z.dev(ln2_g); dl2b=z.dev(ln2_b)
    dmg=z.dev(mln_g); dmb=z.dev(mln_b); dWm1=z.dev(flat(Wm1)); dWm2=z.dev(flat(Wm2))
    # LayerNorm1
    dxn = z.alloc(S*HV); z.launch("layernorm",(S,1,1),(1,1,1),[vp(dX),vp(dl1g),vp(dl1b),vp(dxn),C.c_int(S),C.c_int(HV),C.c_float(EPS)])
    # QKV proj
    dQ=z.gemm(dxn,dWq,S,HV,NH*HD); dK=z.gemm(dxn,dWk,S,HV,NH*HD); dV=z.gemm(dxn,dWv,S,HV,NH*HD)
    # 2D-RoPE on Q,K
    dQr=z.alloc(S*NH*HD); z.launch("rope2d",(S,1,1),(1,1,1),[vp(dQ),vp(dQr),C.c_int(S),C.c_int(NH*HD),C.c_int(NH),C.c_int(HD),C.c_int(GW),C.c_float(THETA)])
    dKr=z.alloc(S*NH*HD); z.launch("rope2d",(S,1,1),(1,1,1),[vp(dK),vp(dKr),C.c_int(S),C.c_int(NH*HD),C.c_int(NH),C.c_int(HD),C.c_int(GW),C.c_float(THETA)])
    # full MHA
    dao=z.alloc(S*NH*HD); z.launch("mha",(1,1,1),(1,1,1),[vp(dQr),vp(dKr),vp(dV),vp(dao),C.c_int(S),C.c_int(NH),C.c_int(HD)])
    # out proj + residual
    do=z.gemm(dao,dWo,S,NH*HD,HV)
    dx1=z.alloc(S*HV); z.launch("addk",((S*HV+63)//64,1,1),(64,1,1),[vp(dX),vp(do),vp(dx1),C.c_int(S*HV)])
    # LayerNorm2 + MLP (fc1 -> GELU -> fc2)
    dxn2=z.alloc(S*HV); z.launch("layernorm",(S,1,1),(1,1,1),[vp(dx1),vp(dl2g),vp(dl2b),vp(dxn2),C.c_int(S),C.c_int(HV),C.c_float(EPS)])
    dh=z.gemm(dxn2,dWfc1,S,HV,IV)
    dhg=z.alloc(S*IV); z.launch("gelu",((S*IV+63)//64,1,1),(64,1,1),[vp(dh),vp(dhg),C.c_int(S*IV)])
    dmlp=z.gemm(dhg,dWfc2,S,IV,HV)
    dblk=z.alloc(S*HV); z.launch("addk",((S*HV+63)//64,1,1),(64,1,1),[vp(dx1),vp(dmlp),vp(dblk),C.c_int(S*HV)])
    # patch merger: spatial 2x2 merge -> LayerNorm -> fc1 -> GELU -> fc2
    dmrg=z.alloc(SM*MDIM); z.launch("smerge",(SM,1,1),(1,1,1),[vp(dblk),vp(dmrg),C.c_int(GH),C.c_int(GW),C.c_int(HV),C.c_int(MS)])
    dmrgn=z.alloc(SM*MDIM); z.launch("layernorm",(SM,1,1),(1,1,1),[vp(dmrg),vp(dmg),vp(dmb),vp(dmrgn),C.c_int(SM),C.c_int(MDIM),C.c_float(EPS)])
    dm1=z.gemm(dmrgn,dWm1,SM,MDIM,IM)
    dm1g=z.alloc(SM*IM); z.launch("gelu",((SM*IM+63)//64,1,1),(64,1,1),[vp(dm1),vp(dm1g),C.c_int(SM*IM)])
    dout=z.gemm(dm1g,dWm2,SM,IM,OUT)
    fo=z.get(dout, SM*OUT)
    return [fo[i*OUT:(i+1)*OUT] for i in range(SM)]

def main():
    ref = reference(); got = run_zluda()
    mxa = mxr = 0.0; nf = 0; rtol, atol = 1e-4, 1e-5; tot = SM*OUT
    for i in range(SM):
        for j in range(OUT):
            a = abs(got[i][j]-ref[i][j]); r = a/(abs(ref[i][j])+1e-12)
            mxa = max(mxa, a); mxr = max(mxr, r)
            if a > atol + rtol*abs(ref[i][j]): nf += 1
    print(f"Qwen3-VL ViT block + patch-merger through ZLUDA: max_abs={mxa:.3e} max_rel={mxr:.3e} n_fail={nf}/{tot}")
    print("VERDICT:", "PASS" if nf == 0 else "FAIL")
    return 0 if nf == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
